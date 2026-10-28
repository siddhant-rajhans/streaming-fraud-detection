from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.model.train import train_model
from fraud_detection.monitoring.drift import DriftDetector
from fraud_detection.schemas import DriftAlert, DriftSeverity

logger = logging.getLogger(__name__)


class RetrainTrigger:
    """Orchestrates model retraining when drift is detected."""

    def __init__(
        self,
        model_dir: Path = Path("models"),
        data_dir: Path = Path("data"),
        archive_dir: Path = Path("models/archive"),
        min_auc_improvement: float = 0.005,
        min_records_for_retrain: int = 50_000,
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.archive_dir = Path(archive_dir)
        self.min_auc_improvement = min_auc_improvement
        self.min_records_for_retrain = min_records_for_retrain
        self._retrain_history: list[dict[str, Any]] = []

    def should_retrain(self, alerts: list[DriftAlert]) -> bool:
        """Decide whether drift alerts warrant a retrain."""
        if not alerts:
            return False

        critical = [a for a in alerts if a.severity == DriftSeverity.CRITICAL]
        if critical:
            logger.info("Critical drift detected in %d features, triggering retrain", len(critical))
            return True

        high = [a for a in alerts if a.severity == DriftSeverity.HIGH]
        if len(high) >= 3:
            logger.info("High drift in %d features, triggering retrain", len(high))
            return True

        medium_plus = [a for a in alerts if a.severity in (DriftSeverity.HIGH, DriftSeverity.MEDIUM)]
        if len(medium_plus) >= 5:
            logger.info("Widespread medium+ drift (%d features), triggering retrain", len(medium_plus))
            return True

        return False

    def collect_training_window(
        self,
        recent_data: pd.DataFrame,
        historical_path: Path | None = None,
    ) -> pd.DataFrame:
        """Assemble a training dataset from recent data and optionally historical data."""
        if historical_path and historical_path.exists():
            historical = pd.read_csv(historical_path)
            combined = pd.concat([historical, recent_data], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["transaction_id"] if "transaction_id" in combined.columns else None,
            )
            return combined
        return recent_data

    def archive_current_model(self) -> Path | None:
        """Move current model to archive with timestamp."""
        model_path = self.model_dir / "fraud_model.json"
        if not model_path.exists():
            return None

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / f"fraud_model_{ts}.json"
        shutil.copy2(model_path, archive_path)

        meta_path = self.model_dir / "model_meta.json"
        if meta_path.exists():
            shutil.copy2(meta_path, self.archive_dir / f"model_meta_{ts}.json")

        logger.info("Archived current model to %s", archive_path)
        return archive_path

    def retrain(
        self,
        transaction_path: Path,
        identity_path: Path | None = None,
        xgb_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a retrain cycle: archive old model, train new one, validate."""
        current_metrics = self._load_current_metrics()

        archived = self.archive_current_model()

        logger.info("Starting retrain on %s", transaction_path)
        new_metrics = train_model(
            transaction_path=transaction_path,
            identity_path=identity_path,
            output_dir=self.model_dir,
            xgb_params=xgb_params,
        )

        accepted = True
        if current_metrics and "auc" in current_metrics:
            auc_delta = new_metrics["auc"] - current_metrics["auc"]
            if auc_delta < -self.min_auc_improvement:
                logger.warning(
                    "New model AUC %.4f is worse than current %.4f (delta=%.4f), rolling back",
                    new_metrics["auc"], current_metrics["auc"], auc_delta,
                )
                self._rollback(archived)
                accepted = False

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "new_metrics": new_metrics,
            "previous_metrics": current_metrics,
            "archived_to": str(archived) if archived else None,
            "accepted": accepted,
        }
        self._retrain_history.append(result)

        if accepted:
            logger.info(
                "Retrain complete: AUC %.4f -> %.4f",
                current_metrics.get("auc", 0) if current_metrics else 0,
                new_metrics["auc"],
            )
        return result

    def _load_current_metrics(self) -> dict[str, float] | None:
        metrics_path = self.model_dir / "model_meta.json"
        if not metrics_path.exists():
            return None
        meta = json.loads(metrics_path.read_text())
        return meta.get("metrics")

    def _rollback(self, archive_path: Path | None) -> None:
        if archive_path and archive_path.exists():
            shutil.copy2(archive_path, self.model_dir / "fraud_model.json")
            meta_archive = archive_path.parent / archive_path.name.replace(
                "fraud_model_", "model_meta_"
            )
            if meta_archive.exists():
                shutil.copy2(meta_archive, self.model_dir / "model_meta.json")
            logger.info("Rolled back to archived model at %s", archive_path)

    @property
    def retrain_history(self) -> list[dict[str, Any]]:
        return list(self._retrain_history)
