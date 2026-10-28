from datetime import datetime

import pytest
from pydantic import ValidationError

from fraud_detection.schemas import (
    DriftAlert,
    DriftSeverity,
    FraudPrediction,
    Transaction,
    ValidationResult,
)


class TestTransaction:
    def test_valid_transaction(self):
        txn = Transaction(
            transaction_id="12345",
            transaction_dt=86400.0,
            transaction_amt=150.50,
            product_cd="W",
            card1=4567,
        )
        assert txn.transaction_id == "12345"
        assert txn.transaction_amt == 150.50

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError):
            Transaction(
                transaction_id="1",
                transaction_dt=0,
                transaction_amt=-10.0,
                product_cd="W",
                card1=1000,
            )

    def test_optional_fields_default_none(self):
        txn = Transaction(
            transaction_id="1",
            transaction_dt=0,
            transaction_amt=10,
            product_cd="W",
            card1=1000,
        )
        assert txn.card2 is None
        assert txn.addr1 is None
        assert txn.p_emaildomain is None

    def test_all_fields_populated(self):
        txn = Transaction(
            transaction_id="99",
            transaction_dt=172800.0,
            transaction_amt=500.0,
            product_cd="H",
            card1=5000,
            card2=200.0,
            card3=150.0,
            card4="visa",
            card5=226.0,
            card6="debit",
            addr1=315.0,
            addr2=87.0,
            dist1=19.0,
            dist2=None,
            p_emaildomain="gmail.com",
            r_emaildomain=None,
            c1=1.0,
            c2=1.0,
        )
        assert txn.card4 == "visa"
        assert txn.c1 == 1.0


class TestFraudPrediction:
    def test_valid_prediction(self):
        pred = FraudPrediction(
            transaction_id="123",
            fraud_probability=0.85,
            is_fraud=True,
            model_version="v1",
            latency_ms=3.2,
        )
        assert pred.is_fraud is True
        assert pred.fraud_probability == 0.85

    def test_probability_bounds(self):
        with pytest.raises(ValidationError):
            FraudPrediction(
                transaction_id="1",
                fraud_probability=1.5,
                is_fraud=True,
                model_version="v1",
                latency_ms=1.0,
            )


class TestDriftAlert:
    def test_valid_alert(self):
        alert = DriftAlert(
            alert_id="abc-123",
            feature_name="transaction_amt",
            drift_score=0.25,
            test_method="PSI+KS",
            severity=DriftSeverity.HIGH,
            reference_stats={"mean": 100.0, "std": 50.0},
            current_stats={"mean": 150.0, "std": 75.0},
            window_size=10000,
        )
        assert alert.severity == DriftSeverity.HIGH
        assert alert.retrain_triggered is False


class TestValidationResult:
    def test_valid_result(self):
        result = ValidationResult(
            transaction_id="1",
            is_valid=True,
        )
        assert result.errors == []
        assert result.null_fields == []

    def test_invalid_result_with_errors(self):
        result = ValidationResult(
            transaction_id="2",
            is_valid=False,
            errors=["Missing field 'card1'"],
            null_fields=["card1"],
        )
        assert not result.is_valid
        assert len(result.errors) == 1
