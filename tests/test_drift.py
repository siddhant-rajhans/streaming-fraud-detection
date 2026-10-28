import numpy as np
import pandas as pd
import pytest

from fraud_detection.monitoring.drift import DriftDetector, _calculate_psi, _classify_severity
from fraud_detection.schemas import DriftSeverity


class TestPSI:
    def test_identical_distributions(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 5000)
        psi = _calculate_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution_has_high_psi(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 5000)
        shifted = rng.normal(2, 1, 5000)
        psi = _calculate_psi(ref, shifted)
        assert psi > 0.2

    def test_wider_distribution(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 5000)
        wider = rng.normal(0, 3, 5000)
        psi = _calculate_psi(ref, wider)
        assert psi > 0.1


class TestSeverityClassification:
    def test_critical_high_psi(self):
        assert _classify_severity(0.6, 0.05) == DriftSeverity.CRITICAL

    def test_critical_low_pvalue(self):
        assert _classify_severity(0.1, 1e-12) == DriftSeverity.CRITICAL

    def test_high(self):
        assert _classify_severity(0.3, 0.05) == DriftSeverity.HIGH

    def test_medium(self):
        assert _classify_severity(0.18, 0.001) == DriftSeverity.MEDIUM

    def test_low(self):
        assert _classify_severity(0.05, 0.5) == DriftSeverity.LOW


class TestDriftDetector:
    @pytest.fixture
    def reference_data(self):
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "transaction_amt": rng.exponential(100, 1000),
            "card1": rng.randint(1000, 20000, 1000).astype(float),
        })

    def test_no_drift_with_similar_data(self, reference_data):
        detector = DriftDetector(
            reference_data=reference_data,
            feature_columns=["transaction_amt", "card1"],
            window_size=500,
        )
        rng = np.random.RandomState(99)
        for _ in range(600):
            detector.add_record({
                "transaction_amt": float(rng.exponential(100)),
                "card1": float(rng.randint(1000, 20000)),
            })

        alerts = detector.check_drift()
        assert len(alerts) == 0

    def test_detects_drift_with_shifted_data(self, reference_data):
        detector = DriftDetector(
            reference_data=reference_data,
            feature_columns=["transaction_amt"],
            window_size=500,
        )
        rng = np.random.RandomState(99)
        for _ in range(600):
            detector.add_record({
                "transaction_amt": float(rng.exponential(500)),
            })

        alerts = detector.check_drift()
        assert len(alerts) > 0
        assert alerts[0].feature_name == "transaction_amt"

    def test_insufficient_data_returns_empty(self, reference_data):
        detector = DriftDetector(
            reference_data=reference_data,
            feature_columns=["transaction_amt"],
            window_size=1000,
        )
        detector.add_record({"transaction_amt": 100.0})
        assert detector.check_drift() == []

    def test_add_batch(self, reference_data):
        detector = DriftDetector(
            reference_data=reference_data,
            feature_columns=["transaction_amt"],
            window_size=100,
        )
        records = [{"transaction_amt": float(i)} for i in range(200)]
        detector.add_batch(records)
        assert len(detector._buffer) == 200
