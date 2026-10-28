from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fraud_detection.schemas import FraudPrediction

logger = logging.getLogger(__name__)


class FraudScorer:
    """Loads a trained XGBoost model and scores individual transactions."""

    def __init__(self, model_dir: Path = Path("models"), version: str = "latest"):
        import xgboost as xgb

        self.model_dir = Path(model_dir)
        self.version = version
        self._model: xgb.Booster | None = None
        self._feature_columns: list[str] = []
        self._threshold: float = 0.5
        self._load()

    def _load(self) -> None:
        model_path = self.model_dir / "fraud_model.json"
        meta_path = self.model_dir / "model_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        import xgboost as xgb

        self._model = xgb.Booster()
        self._model.load_model(str(model_path))

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._feature_columns = meta.get("feature_columns", [])
            self._threshold = meta.get("threshold", 0.5)

        logger.info(
            "Loaded model from %s (threshold=%.4f, %d features)",
            model_path, self._threshold, len(self._feature_columns),
        )

    def extract_features(self, txn: dict[str, Any]) -> np.ndarray:
        """Extract feature vector from a transaction dict, matching training pipeline order."""
        amt = float(txn.get("transaction_amt", 0))
        dt = float(txn.get("transaction_dt", 0))

        derived = {
            "amt_log": np.log1p(amt),
            "amt_bin": _amount_to_bin(amt),
            "hour": (dt % 86400) / 3600,
            "day_of_week": int((dt // 86400) % 7),
            "is_night": int((dt % 86400) / 3600 < 6 or (dt % 86400) / 3600 > 22),
            "is_weekend": int(int((dt // 86400) % 7) >= 5),
            "card1_freq": 1.0,
            "card1_freq_log": 0.0,
            "email_risk": 0.035,
        }

        features = []
        for col in self._feature_columns:
            if col in derived:
                features.append(derived[col])
            elif col in txn:
                val = txn[col]
                features.append(float(val) if not isinstance(val, str) else -999.0)
            else:
                features.append(-999.0)

        return np.array(features, dtype=np.float32)

    def score(self, txn: dict[str, Any]) -> FraudPrediction:
        """Score a single transaction and return a FraudPrediction."""
        start = time.perf_counter()

        features = self.extract_features(txn)
        import xgboost as xgb

        dmat = xgb.DMatrix(features.reshape(1, -1), feature_names=self._feature_columns)
        proba = float(self._model.predict(dmat)[0])

        latency_ms = (time.perf_counter() - start) * 1000

        return FraudPrediction(
            transaction_id=str(txn.get("transaction_id", "")),
            fraud_probability=proba,
            is_fraud=proba >= self._threshold,
            model_version=self.version,
            scored_at=datetime.utcnow(),
            latency_ms=latency_ms,
        )

    def score_batch(self, transactions: list[dict[str, Any]]) -> list[FraudPrediction]:
        """Score a batch of transactions."""
        start = time.perf_counter()

        import xgboost as xgb

        feature_matrix = np.array(
            [self.extract_features(txn) for txn in transactions],
            dtype=np.float32,
        )
        dmat = xgb.DMatrix(feature_matrix, feature_names=self._feature_columns)
        probas = self._model.predict(dmat)

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_txn_ms = elapsed_ms / max(len(transactions), 1)

        return [
            FraudPrediction(
                transaction_id=str(txn.get("transaction_id", "")),
                fraud_probability=float(p),
                is_fraud=float(p) >= self._threshold,
                model_version=self.version,
                scored_at=datetime.utcnow(),
                latency_ms=per_txn_ms,
            )
            for txn, p in zip(transactions, probas)
        ]


def _amount_to_bin(amt: float) -> float:
    if amt <= 10:
        return 0.0
    elif amt <= 50:
        return 1.0
    elif amt <= 100:
        return 2.0
    elif amt <= 500:
        return 3.0
    elif amt <= 1000:
        return 4.0
    elif amt <= 5000:
        return 5.0
    return 6.0
