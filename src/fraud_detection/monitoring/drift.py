from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fraud_detection.schemas import DriftAlert, DriftSeverity

logger = logging.getLogger(__name__)


class DriftDetector:
    """Monitors feature and prediction drift using a sliding window approach."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: list[str],
        prediction_column: str = "fraud_probability",
        psi_threshold: float = 0.15,
        ks_threshold: float = 0.05,
        window_size: int = 10_000,
    ):
        self.reference = reference_data
        self.feature_columns = feature_columns
        self.prediction_column = prediction_column
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.window_size = window_size
        self._buffer: list[dict] = []

    def add_record(self, record: dict[str, Any]) -> None:
        self._buffer.append(record)
        if len(self._buffer) > self.window_size * 2:
            self._buffer = self._buffer[-self.window_size:]

    def add_batch(self, records: list[dict[str, Any]]) -> None:
        self._buffer.extend(records)
        if len(self._buffer) > self.window_size * 2:
            self._buffer = self._buffer[-self.window_size:]

    @property
    def current_window(self) -> pd.DataFrame:
        data = self._buffer[-self.window_size:]
        return pd.DataFrame(data)

    def check_drift(self) -> list[DriftAlert]:
        """Run drift checks on the current window. Returns alerts for drifted features."""
        if len(self._buffer) < self.window_size:
            logger.debug(
                "Buffer has %d records, need %d for drift check",
                len(self._buffer), self.window_size,
            )
            return []

        current = self.current_window
        alerts = []

        for col in self.feature_columns:
            if col not in current.columns or col not in self.reference.columns:
                continue

            ref_vals = self.reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 100 or len(cur_vals) < 100:
                continue

            psi_score = _calculate_psi(ref_vals, cur_vals)
            ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)

            if psi_score > self.psi_threshold or ks_pvalue < self.ks_threshold:
                severity = _classify_severity(psi_score, ks_pvalue)
                alerts.append(DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    detected_at=datetime.utcnow(),
                    feature_name=col,
                    drift_score=float(psi_score),
                    test_method="PSI+KS",
                    severity=severity,
                    reference_stats={
                        "mean": float(np.mean(ref_vals)),
                        "std": float(np.std(ref_vals)),
                        "median": float(np.median(ref_vals)),
                    },
                    current_stats={
                        "mean": float(np.mean(cur_vals)),
                        "std": float(np.std(cur_vals)),
                        "median": float(np.median(cur_vals)),
                    },
                    window_size=self.window_size,
                ))
                logger.warning(
                    "Drift detected in %s: PSI=%.4f, KS p=%.4e, severity=%s",
                    col, psi_score, ks_pvalue, severity.value,
                )

        return alerts

    def check_prediction_drift(self) -> DriftAlert | None:
        """Check for drift in model predictions specifically."""
        if len(self._buffer) < self.window_size:
            return None

        current = self.current_window
        if self.prediction_column not in current.columns:
            return None
        if self.prediction_column not in self.reference.columns:
            return None

        ref_preds = self.reference[self.prediction_column].dropna().values
        cur_preds = current[self.prediction_column].dropna().values

        if len(ref_preds) < 100 or len(cur_preds) < 100:
            return None

        psi_score = _calculate_psi(ref_preds, cur_preds)
        ks_stat, ks_pvalue = stats.ks_2samp(ref_preds, cur_preds)

        if psi_score > self.psi_threshold or ks_pvalue < self.ks_threshold:
            return DriftAlert(
                alert_id=str(uuid.uuid4()),
                detected_at=datetime.utcnow(),
                feature_name=self.prediction_column,
                drift_score=float(psi_score),
                test_method="PSI+KS",
                severity=_classify_severity(psi_score, ks_pvalue),
                reference_stats={
                    "mean": float(np.mean(ref_preds)),
                    "fraud_rate": float((ref_preds > 0.5).mean()),
                },
                current_stats={
                    "mean": float(np.mean(cur_preds)),
                    "fraud_rate": float((cur_preds > 0.5).mean()),
                },
                window_size=self.window_size,
            )
        return None

    def generate_evidently_report(self, output_path: str | None = None):
        """Generate a full Evidently data drift report."""
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        current = self.current_window
        cols = [c for c in self.feature_columns if c in current.columns and c in self.reference.columns]

        report = Report(metrics=[DataDriftPreset(columns=cols)])
        report.run(reference_data=self.reference[cols], current_data=current[cols])

        if output_path:
            report.save_html(output_path)
            logger.info("Evidently report saved to %s", output_path)

        return report


def _calculate_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two distributions."""
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, buckets + 1)

    ref_counts = np.histogram(reference, bins=bins)[0].astype(float)
    cur_counts = np.histogram(current, bins=bins)[0].astype(float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _classify_severity(psi_score: float, ks_pvalue: float) -> DriftSeverity:
    if psi_score > 0.5 or ks_pvalue < 1e-10:
        return DriftSeverity.CRITICAL
    elif psi_score > 0.25 or ks_pvalue < 1e-5:
        return DriftSeverity.HIGH
    elif psi_score > 0.15 or ks_pvalue < 0.01:
        return DriftSeverity.MEDIUM
    return DriftSeverity.LOW
