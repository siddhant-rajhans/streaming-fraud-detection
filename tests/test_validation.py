import pytest

from fraud_detection.monitoring.validation import (
    BatchValidator,
    validate_transaction,
)


class TestValidateTransaction:
    def test_valid_transaction(self):
        txn = {
            "transaction_id": "123",
            "transaction_dt": 86400.0,
            "transaction_amt": 100.0,
            "product_cd": "W",
            "card1": 5000,
        }
        result = validate_transaction(txn)
        assert result.is_valid
        assert result.errors == []

    def test_missing_required_field(self):
        txn = {
            "transaction_dt": 86400.0,
            "transaction_amt": 100.0,
            "product_cd": "W",
            "card1": 5000,
        }
        result = validate_transaction(txn)
        assert not result.is_valid
        assert "transaction_id" in result.null_fields

    def test_out_of_range_amount(self):
        txn = {
            "transaction_id": "1",
            "transaction_dt": 86400.0,
            "transaction_amt": 200_000.0,
            "product_cd": "W",
            "card1": 5000,
        }
        result = validate_transaction(txn)
        assert not result.is_valid
        assert "transaction_amt" in result.out_of_range_fields

    def test_invalid_product_code(self):
        txn = {
            "transaction_id": "1",
            "transaction_dt": 86400.0,
            "transaction_amt": 100.0,
            "product_cd": "Z",
            "card1": 5000,
        }
        result = validate_transaction(txn)
        assert not result.is_valid
        assert any("product_cd" in e for e in result.errors)

    def test_invalid_card4(self):
        txn = {
            "transaction_id": "1",
            "transaction_dt": 86400.0,
            "transaction_amt": 100.0,
            "product_cd": "W",
            "card1": 5000,
            "card4": "diners",
        }
        result = validate_transaction(txn)
        assert not result.is_valid

    def test_zero_amount_is_valid(self):
        txn = {
            "transaction_id": "1",
            "transaction_dt": 86400.0,
            "transaction_amt": 0.0,
            "product_cd": "W",
            "card1": 5000,
        }
        result = validate_transaction(txn)
        assert result.is_valid


class TestBatchValidator:
    def test_all_valid(self):
        validator = BatchValidator()
        txns = [
            {
                "transaction_id": str(i),
                "transaction_dt": 86400.0,
                "transaction_amt": 100.0,
                "product_cd": "W",
                "card1": 5000,
            }
            for i in range(10)
        ]
        valid, rejected, results = validator.validate_batch(txns)
        assert len(valid) == 10
        assert len(rejected) == 0
        assert validator.error_rate == 0.0

    def test_mixed_batch(self):
        validator = BatchValidator()
        txns = [
            {
                "transaction_id": "1",
                "transaction_dt": 86400.0,
                "transaction_amt": 100.0,
                "product_cd": "W",
                "card1": 5000,
            },
            {
                "transaction_dt": 86400.0,
                "transaction_amt": 100.0,
                "product_cd": "W",
                "card1": 5000,
            },
        ]
        valid, rejected, _ = validator.validate_batch(txns)
        assert len(valid) == 1
        assert len(rejected) == 1
        assert validator.error_rate == 0.5

    def test_quality_gate_fails_on_high_error_rate(self):
        validator = BatchValidator(max_error_rate=0.01)
        bad_txns = [{"transaction_amt": -1.0} for _ in range(10)]
        validator.validate_batch(bad_txns)
        passed, issues = validator.check_quality_gate()
        assert not passed
        assert len(issues) > 0

    def test_reset(self):
        validator = BatchValidator()
        validator.validate_batch([{"bad": True}])
        validator.reset()
        assert validator.error_rate == 0.0
        assert validator.null_rates == {}
