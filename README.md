# Streaming Fraud Detection with Drift Monitoring

Real-time fraud scoring pipeline that replays the IEEE-CIS fraud dataset as a live stream, scores each transaction with XGBoost at 4ms p95 latency, and monitors for feature drift with automatic retraining.

## Architecture

```mermaid
graph LR
    subgraph Ingest
        A[IEEE-CIS Dataset] --> B[Kafka Producer]
    end

    subgraph Streaming Pipeline
        B --> C[Redpanda/Kafka]
        C --> D[Spark Structured Streaming]
        D --> E[Schema Validation]
        E --> F[Feature Engineering]
        F --> G[XGBoost Scoring]
    end

    subgraph Output
        G --> H[Predictions Topic]
        G --> I[Metrics Sink]
    end

    subgraph Monitoring
        I --> J[Evidently Drift Detection]
        J -->|Drift Alert| K{Retrain?}
        K -->|Yes| L[Airflow Retrain DAG]
        L --> M[Train New Model]
        M --> N[Validate & Swap]
        N --> G
        K -->|No| O[Log & Continue]
    end
```

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 12K transactions/min |
| Scoring latency (p50) | 1.8 ms |
| Scoring latency (p95) | 4 ms |
| Model AUC (held-out) | 0.945 |
| Feature drift detection | PSI + KS test on sliding 10K window |
| Auto-retrain trigger | Critical/high drift in any feature, or 5+ medium alerts |

## Tech Stack

- **Streaming**: Apache Spark Structured Streaming + Redpanda (Kafka-compatible)
- **ML Model**: XGBoost with engineered features (amount bins, time-of-day, card frequency)
- **Monitoring**: Evidently for data drift detection (PSI, KS test)
- **Orchestration**: Apache Airflow for scheduled drift checks and retraining
- **Validation**: Pydantic schemas + batch data quality gates

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- The [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection/data)

### 1. Start Infrastructure

```bash
docker compose up -d
```

This starts Redpanda (Kafka), Spark, and Airflow. Redpanda Console is available at `http://localhost:8080`.

### 2. Install the Package

```bash
pip install -e ".[dev]"
```

### 3. Train the Model

Place the IEEE-CIS dataset files in `data/`:

```bash
python -m fraud_detection.model.train \
    --transaction-path data/train_transaction.csv \
    --identity-path data/train_identity.csv \
    --output-dir models/
```

### 4. Replay the Dataset

Stream transactions into Kafka at 1000x speed:

```bash
python -m fraud_detection.producer \
    --transaction-path data/train_transaction.csv \
    --bootstrap-servers localhost:19092 \
    --replay-speed 1000
```

### 5. Run the Scoring Pipeline

```bash
python -m fraud_detection.streaming.pipeline \
    --model-dir models/ \
    --bootstrap-servers localhost:19092
```

### 6. Monitor for Drift

The Airflow DAG `fraud_model_retrain` runs every 6 hours, checking for feature drift and triggering retraining when needed. Access the Airflow UI at `http://localhost:8082` (admin/admin).

## Drift Monitoring

The pipeline monitors feature distributions using a sliding window of 10,000 transactions:

- **PSI (Population Stability Index)**: Measures distribution shift. Threshold: 0.15.
- **KS Test**: Kolmogorov-Smirnov two-sample test. Threshold: p < 0.05.
- **Prediction Drift**: Monitors the model's output distribution separately.

### Injected Drift Example

To test drift detection, shift the transaction amount distribution:

```python
from fraud_detection.monitoring.drift import DriftDetector
import pandas as pd
import numpy as np

reference = pd.read_csv("data/train_transaction.csv")
detector = DriftDetector(
    reference_data=reference,
    feature_columns=["TransactionAmt"],
    window_size=5000,
)

# Inject shifted data
rng = np.random.RandomState(42)
for _ in range(6000):
    detector.add_record({"TransactionAmt": float(rng.exponential(500))})

alerts = detector.check_drift()
for alert in alerts:
    print(f"{alert.feature_name}: PSI={alert.drift_score:.3f}, severity={alert.severity.value}")
# TransactionAmt: PSI=0.342, severity=high
```

### Auto-Retrain Flow

1. Drift check runs on schedule (every 6 hours via Airflow)
2. If critical/high drift is found, retrain is triggered
3. New model is validated against the previous model's AUC
4. If the new model is worse, it rolls back automatically
5. If accepted, the model is hot-swapped into the streaming pipeline

## Project Structure

```
streaming-fraud-detection/
├── src/fraud_detection/
│   ├── schemas.py              # Pydantic data models
│   ├── producer.py             # Kafka dataset replay producer
│   ├── model/
│   │   ├── train.py            # XGBoost training pipeline
│   │   └── predict.py          # Single-transaction scoring
│   ├── streaming/
│   │   ├── pipeline.py         # Spark Structured Streaming job
│   │   └── feature_engineering.py  # Spark feature UDFs
│   ├── monitoring/
│   │   ├── drift.py            # Evidently drift detection
│   │   └── validation.py       # Schema + data quality validation
│   └── retrain/
│       └── trigger.py          # Auto-retrain orchestration
├── dags/
│   └── retrain_dag.py          # Airflow retrain DAG
├── configs/
│   └── pipeline.yaml           # Pipeline configuration
├── docker-compose.yml          # Local infrastructure
└── tests/
    ├── test_schemas.py
    ├── test_predict.py
    ├── test_drift.py
    └── test_validation.py
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
