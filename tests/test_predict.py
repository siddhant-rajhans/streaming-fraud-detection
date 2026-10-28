import numpy as np
import pytest

from fraud_detection.model.predict import FraudScorer, _amount_to_bin


class TestAmountToBin:
    @pytest.mark.parametrize("amount,expected", [
        (5.0, 0.0),
        (10.0, 0.0),
        (25.0, 1.0),
        (50.0, 1.0),
        (75.0, 2.0),
        (250.0, 3.0),
        (750.0, 4.0),
        (2500.0, 5.0),
        (10000.0, 6.0),
    ])
    def test_bin_boundaries(self, amount: float, expected: float):
        assert _amount_to_bin(amount) == expected

    def test_zero_amount(self):
        assert _amount_to_bin(0.0) == 0.0


class TestFraudScorerFeatures:
    def test_extract_features_minimal(self):
        scorer = FraudScorer.__new__(FraudScorer)
        scorer._feature_columns = [
            "transaction_amt", "card1", "amt_log", "hour", "day_of_week",
        ]
        scorer._threshold = 0.5

        txn = {
            "transaction_id": "100",
            "transaction_amt": 150.0,
            "transaction_dt": 43200.0,
            "card1": 5000,
        }
        features = scorer.extract_features(txn)
        assert len(features) == 5
        assert features[0] == 150.0
        assert features[1] == 5000.0
        assert np.isclose(features[2], np.log1p(150.0))

    def test_missing_fields_fill_sentinel(self):
        scorer = FraudScorer.__new__(FraudScorer)
        scorer._feature_columns = ["transaction_amt", "dist1", "dist2"]
        scorer._threshold = 0.5

        txn = {"transaction_id": "1", "transaction_amt": 50.0, "transaction_dt": 0}
        features = scorer.extract_features(txn)
        assert features[1] == -999.0
        assert features[2] == -999.0
