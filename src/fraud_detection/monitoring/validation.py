from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fraud_detection.schemas import ValidationResult

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["transaction_id", "transaction_dt", "transaction_amt", "product_cd", "card1"]

RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "transaction_amt": (0.0, 100_000.0),
    "transaction_dt": (0.0, 1e9),
    "card1": (1000, 20000),
    "addr1": (100, 600),
    "addr2": (10, 100),
}

VALID_PRODUCT_CODES = {"W", "H", "C", "S", "R"}
VALID_CARD4 = {"visa", "mastercard", "american express", "discover"}
VALID_CARD6 = {"debit", "credit", "charge card", "debit or credit"}


def validate_transaction(txn: dict[str, Any]) -> ValidationResult:
    """Validate a single transaction record for schema conformance and data quality."""
    errors: list[str] = []
    null_fields: list[str] = []
    out_of_range: list[str] = []

    txn_id = str(txn.get("transaction_id", "unknown"))

    for field in REQUIRED_FIELDS:
        if field not in txn or txn[field] is None:
            null_fields.append(field)
            errors.append(f"Required field '{field}' is missing or null")

    for field, (lo, hi) in RANGE_CHECKS.items():
        val = txn.get(field)
        if val is not None:
            try:
                fval = float(val)
                if fval < lo or fval > hi:
                    out_of_range.append(field)
                    errors.append(f"'{field}' value {fval} outside [{lo}, {hi}]")
            except (ValueError, TypeError):
                errors.append(f"'{field}' is not numeric: {val!r}")

    product_cd = txn.get("product_cd")
    if product_cd and product_cd not in VALID_PRODUCT_CODES:
        errors.append(f"Invalid product_cd: {product_cd!r}")

    card4 = txn.get("card4")
    if card4 and str(card4).lower() not in VALID_CARD4:
        errors.append(f"Invalid card4: {card4!r}")

    card6 = txn.get("card6")
    if card6 and str(card6).lower() not in VALID_CARD6:
        errors.append(f"Invalid card6: {card6!r}")

    return ValidationResult(
        transaction_id=txn_id,
        is_valid=len(errors) == 0,
        errors=errors,
        null_fields=null_fields,
        out_of_range_fields=out_of_range,
        validated_at=datetime.utcnow(),
    )


class BatchValidator:
    """Validates batches of transactions and tracks data quality metrics."""

    def __init__(self, max_null_rate: float = 0.1, max_error_rate: float = 0.05):
        self.max_null_rate = max_null_rate
        self.max_error_rate = max_error_rate
        self._total_count = 0
        self._error_count = 0
        self._null_counts: dict[str, int] = {}

    def validate_batch(
        self, transactions: list[dict[str, Any]]
    ) -> tuple[list[dict], list[dict], list[ValidationResult]]:
        """Validate a batch. Returns (valid_txns, rejected_txns, all_results)."""
        valid = []
        rejected = []
        results = []

        for txn in transactions:
            result = validate_transaction(txn)
            results.append(result)
            self._total_count += 1

            if result.is_valid:
                valid.append(txn)
            else:
                rejected.append(txn)
                self._error_count += 1
                for field in result.null_fields:
                    self._null_counts[field] = self._null_counts.get(field, 0) + 1

        if rejected:
            logger.warning(
                "Batch validation: %d/%d rejected (%.1f%%)",
                len(rejected), len(transactions),
                100 * len(rejected) / max(len(transactions), 1),
            )

        return valid, rejected, results

    @property
    def error_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._error_count / self._total_count

    @property
    def null_rates(self) -> dict[str, float]:
        if self._total_count == 0:
            return {}
        return {
            field: count / self._total_count
            for field, count in self._null_counts.items()
        }

    def check_quality_gate(self) -> tuple[bool, list[str]]:
        """Check if data quality is within acceptable thresholds."""
        issues: list[str] = []

        if self.error_rate > self.max_error_rate:
            issues.append(
                f"Error rate {self.error_rate:.2%} exceeds threshold {self.max_error_rate:.2%}"
            )

        for field, rate in self.null_rates.items():
            if rate > self.max_null_rate:
                issues.append(
                    f"Null rate for '{field}' is {rate:.2%} (threshold: {self.max_null_rate:.2%})"
                )

        return len(issues) == 0, issues

    def reset(self) -> None:
        self._total_count = 0
        self._error_count = 0
        self._null_counts.clear()
