"""Airflow DAG for scheduled drift checking and model retraining."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

MODEL_DIR = Path("/opt/fraud-detection/models")
DATA_DIR = Path("/opt/fraud-detection/data")
REFERENCE_DATA_PATH = DATA_DIR / "reference_data.parquet"
TRANSACTION_PATH = DATA_DIR / "train_transaction.csv"


def check_drift(**context) -> str:
    import pandas as pd
    from fraud_detection.monitoring.drift import DriftDetector
    from fraud_detection.model.train import get_feature_columns
    import json

    meta_path = MODEL_DIR / "model_meta.json"
    meta = json.loads(meta_path.read_text())
    feature_columns = meta["feature_columns"]

    reference = pd.read_parquet(REFERENCE_DATA_PATH)

    recent_predictions_path = DATA_DIR / "recent_predictions.parquet"
    if not recent_predictions_path.exists():
        return "skip_retrain"

    current = pd.read_parquet(recent_predictions_path)

    detector = DriftDetector(
        reference_data=reference,
        feature_columns=feature_columns,
        window_size=len(current),
    )
    detector.add_batch(current.to_dict("records"))
    alerts = detector.check_drift()

    context["ti"].xcom_push(key="drift_alerts", value=[a.model_dump() for a in alerts])
    context["ti"].xcom_push(key="n_alerts", value=len(alerts))

    critical_or_high = [
        a for a in alerts
        if a.severity.value in ("critical", "high")
    ]

    if len(critical_or_high) >= 1 or len(alerts) >= 5:
        return "retrain_model"
    return "skip_retrain"


def retrain_model(**context) -> None:
    from fraud_detection.retrain.trigger import RetrainTrigger

    trigger = RetrainTrigger(model_dir=MODEL_DIR, data_dir=DATA_DIR)
    result = trigger.retrain(
        transaction_path=TRANSACTION_PATH,
        identity_path=DATA_DIR / "train_identity.csv",
    )

    context["ti"].xcom_push(key="retrain_result", value=result)


def skip_retrain(**context) -> None:
    n_alerts = context["ti"].xcom_pull(key="n_alerts", task_ids="check_drift") or 0
    print(f"No retrain needed. {n_alerts} drift alert(s) below threshold.")


def log_results(**context) -> None:
    retrain_result = context["ti"].xcom_pull(key="retrain_result", task_ids="retrain_model")
    drift_alerts = context["ti"].xcom_pull(key="drift_alerts", task_ids="check_drift") or []

    print(f"Drift alerts: {len(drift_alerts)}")
    if retrain_result:
        print(f"Retrain accepted: {retrain_result.get('accepted')}")
        new_metrics = retrain_result.get("new_metrics", {})
        print(f"New model AUC: {new_metrics.get('auc', 'N/A')}")


with DAG(
    dag_id="fraud_model_retrain",
    default_args=default_args,
    description="Check for feature drift and retrain fraud model if needed",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "fraud", "retrain"],
) as dag:

    drift_check = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    skip = PythonOperator(
        task_id="skip_retrain",
        python_callable=skip_retrain,
    )

    log = PythonOperator(
        task_id="log_results",
        python_callable=log_results,
        trigger_rule="none_failed_min_one_success",
    )

    drift_check >> [retrain, skip]
    retrain >> log
    skip >> log
