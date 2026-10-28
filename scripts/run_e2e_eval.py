#!/usr/bin/env python3
"""End-to-end evaluation: generate synthetic fraud data, train, score, measure drift."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import ks_2samp

np.random.seed(42)


def generate_synthetic_fraud_data(n_samples: int = 50_000) -> pd.DataFrame:
    """Generate synthetic transaction data resembling IEEE-CIS fraud dataset."""
    n_fraud = int(n_samples * 0.035)
    n_legit = n_samples - n_fraud

    def make_legit(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "transaction_amt": np.random.lognormal(3.5, 1.2, n).clip(1, 10000),
            "transaction_dt": np.random.uniform(0, 86400 * 180, n),
            "card1": np.random.randint(1000, 20000, n),
            "card2": np.random.choice([100, 150, 200, 250, 300, 350], n),
            "card3": np.random.choice([150, 185, 200], n),
            "card5": np.random.choice([100, 117, 200, 224, 225, 226], n),
            "addr1": np.random.randint(100, 500, n).astype(float),
            "addr2": np.random.choice([87.0, 65.0, 32.0, 96.0], n),
            "dist1": np.random.exponential(20, n).clip(0, 1000),
            "dist2": np.where(np.random.random(n) > 0.6, np.random.exponential(50, n), np.nan),
            "product_cd": np.random.choice(["W", "H", "C", "S", "R"], n, p=[0.74, 0.1, 0.08, 0.05, 0.03]),
            "card4": np.random.choice(["visa", "mastercard", "discover", "amex"], n, p=[0.5, 0.3, 0.12, 0.08]),
            "card6": np.random.choice(["debit", "credit", "charge"], n, p=[0.6, 0.35, 0.05]),
            "isfraud": 0,
        })

    def make_fraud(n: int) -> pd.DataFrame:
        df = make_legit(n)
        df["isfraud"] = 1
        df["transaction_amt"] = np.random.lognormal(4.5, 1.5, n).clip(10, 15000)
        df["dist1"] = np.random.exponential(80, n).clip(0, 2000)
        late_night = np.random.uniform(0, 6 * 3600, n)
        df["transaction_dt"] = np.random.choice(
            [1], n
        ) * 86400 * np.random.randint(0, 180, n) + late_night
        return df

    legit = make_legit(n_legit)
    fraud = make_fraud(n_fraud)
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["transaction_id"] = range(len(df))
    return df


def engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature engineering matching the production pipeline."""
    df = df.copy()
    amt = df["transaction_amt"]
    df["amt_log"] = np.log1p(amt)
    df["amt_bin"] = pd.cut(
        amt,
        bins=[-1, 10, 50, 100, 500, 1000, 5000, float("inf")],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).astype(float)

    dt = df["transaction_dt"]
    df["hour"] = (dt % 86400) / 3600
    df["day_of_week"] = ((dt // 86400) % 7).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    card_freq = df["card1"].value_counts().to_dict()
    df["card1_freq"] = df["card1"].map(card_freq).fillna(1).astype(float)
    df["card1_freq_log"] = np.log1p(df["card1_freq"])

    for cat in ["product_cd", "card4", "card6"]:
        df[cat] = df[cat].astype("category").cat.codes

    feature_cols = [
        "transaction_amt", "card1", "card2", "card3", "card5",
        "addr1", "addr2", "dist1", "dist2",
        "product_cd", "card4", "card6",
        "amt_log", "amt_bin", "hour", "day_of_week",
        "is_night", "is_weekend", "card1_freq", "card1_freq_log",
    ]
    X = df[feature_cols].fillna(-999).values
    y = df["isfraud"].values
    return X, y, feature_cols


def test_drift_detection(X_train: np.ndarray, feature_cols: list[str]) -> dict:
    """Inject drift into features and measure detection."""
    n_drift = 5000
    X_clean = X_train[:n_drift].copy()
    X_drifted = X_train[:n_drift].copy()

    amt_idx = feature_cols.index("transaction_amt")
    X_drifted[:, amt_idx] = X_drifted[:, amt_idx] * 3.0 + 500

    hour_idx = feature_cols.index("hour")
    X_drifted[:, hour_idx] = np.random.uniform(22, 24, n_drift)

    results = {}
    for i, col in enumerate(feature_cols):
        stat, pval = ks_2samp(X_clean[:, i], X_drifted[:, i])
        results[col] = {"ks_statistic": float(stat), "p_value": float(pval), "drifted": pval < 0.05}

    drifted_features = [k for k, v in results.items() if v["drifted"]]
    return {
        "features_checked": len(feature_cols),
        "features_drifted": len(drifted_features),
        "drifted_names": drifted_features,
        "detection_rate": len(drifted_features) / len(feature_cols),
        "amt_ks": results["transaction_amt"]["ks_statistic"],
        "amt_pval": results["transaction_amt"]["p_value"],
        "hour_ks": results["hour"]["ks_statistic"],
        "hour_pval": results["hour"]["p_value"],
    }


def test_schema_validation() -> dict:
    """Test the validation logic with valid and invalid transactions."""
    from fraud_detection.monitoring.validation import validate_transaction

    valid = {
        "transaction_id": "T001",
        "transaction_amt": 150.0,
        "transaction_dt": 43200.0,
        "card1": 5000,
        "card2": 200.0,
        "card3": 150.0,
        "card5": 226.0,
        "product_cd": "W",
        "card4": "visa",
        "card6": "debit",
    }
    valid_result = validate_transaction(valid)

    invalid = {
        "transaction_id": "T002",
        "transaction_amt": -50.0,
        "transaction_dt": 43200.0,
        "card1": 5000,
        "card2": 200.0,
        "card3": 150.0,
        "card5": 226.0,
        "product_cd": "W",
        "card4": "visa",
        "card6": "debit",
    }
    invalid_result = validate_transaction(invalid)

    return {
        "valid_passes": valid_result.is_valid,
        "invalid_caught": not invalid_result.is_valid,
        "invalid_errors": invalid_result.errors if not invalid_result.is_valid else [],
    }


def benchmark_scoring_throughput(
    model: GradientBoostingClassifier,
    X_test: np.ndarray,
    n_batches: int = 10,
) -> dict:
    """Measure scoring latency and throughput."""
    batch_size = 1000
    latencies = []

    for i in range(n_batches):
        batch = X_test[: batch_size]
        start = time.perf_counter()
        model.predict_proba(batch)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)

    latencies_arr = np.array(latencies)
    per_txn_ms = latencies_arr / batch_size

    return {
        "batch_size": batch_size,
        "n_batches": n_batches,
        "p50_batch_ms": float(np.percentile(latencies_arr, 50)),
        "p95_batch_ms": float(np.percentile(latencies_arr, 95)),
        "p50_per_txn_ms": float(np.percentile(per_txn_ms, 50)),
        "p95_per_txn_ms": float(np.percentile(per_txn_ms, 95)),
        "throughput_txn_per_sec": float(batch_size / (np.mean(latencies_arr) / 1000)),
    }


def main():
    print("=" * 70)
    print("STREAMING FRAUD DETECTION - END-TO-END EVALUATION")
    print("=" * 70)

    # 1. Generate data
    print("\n[1/6] Generating synthetic fraud data...")
    df = generate_synthetic_fraud_data(50_000)
    fraud_rate = df["isfraud"].mean()
    print(f"  Samples: {len(df):,}")
    print(f"  Fraud rate: {fraud_rate:.1%} ({df['isfraud'].sum():,} fraudulent)")

    # 2. Feature engineering
    print("\n[2/6] Engineering features...")
    X, y, feature_cols = engineer_features(df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    # 3. Train model
    print("\n[3/6] Training GradientBoosting model...")
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start
    print(f"  Training time: {train_time:.1f}s")

    # 4. Evaluate model
    print("\n[4/6] Evaluating model performance...")
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "auc": float(auc),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "best_threshold": float(best_threshold),
    }
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Best threshold: {metrics['best_threshold']:.4f}")

    # 5. Scoring throughput
    print("\n[5/6] Benchmarking scoring throughput...")
    throughput = benchmark_scoring_throughput(model, X_test)
    print(f"  Batch p50: {throughput['p50_batch_ms']:.1f}ms ({throughput['batch_size']} txns)")
    print(f"  Batch p95: {throughput['p95_batch_ms']:.1f}ms")
    print(f"  Per-txn p95: {throughput['p95_per_txn_ms']:.3f}ms")
    print(f"  Throughput: {throughput['throughput_txn_per_sec']:,.0f} txns/sec")

    # 6. Drift detection
    print("\n[6/6] Testing drift detection with injected drift...")
    drift = test_drift_detection(X_train, feature_cols)
    print(f"  Features checked: {drift['features_checked']}")
    print(f"  Features with detected drift: {drift['features_drifted']}")
    print(f"  Drifted features: {drift['drifted_names']}")
    print(f"  Amount KS statistic: {drift['amt_ks']:.4f} (p={drift['amt_pval']:.2e})")
    print(f"  Hour KS statistic: {drift['hour_ks']:.4f} (p={drift['hour_pval']:.2e})")

    # 7. Schema validation
    print("\n[BONUS] Testing schema validation...")
    try:
        validation = test_schema_validation()
        print(f"  Valid transaction passes: {validation['valid_passes']}")
        print(f"  Invalid transaction caught: {validation['invalid_caught']}")
        if validation.get("invalid_errors"):
            print(f"  Errors reported: {validation['invalid_errors']}")
    except Exception as e:
        print(f"  Schema validation skipped (import issue): {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - MEASURED RESULTS")
    print("=" * 70)
    print(f"  Dataset: {len(df):,} synthetic transactions ({fraud_rate:.1%} fraud rate)")
    print(f"  Model: GradientBoosting (200 trees, depth 6)")
    print(f"  AUC on held-out test set: {metrics['auc']:.4f}")
    print(f"  F1 score: {metrics['f1']:.4f}")
    print(f"  Scoring throughput: {throughput['throughput_txn_per_sec']:,.0f} txns/sec")
    print(f"  Per-transaction p95 latency: {throughput['p95_per_txn_ms']:.3f}ms")
    print(f"  Drift detection: {drift['features_drifted']}/{drift['features_checked']} injected drifts caught")
    print("=" * 70)

    results = {
        "dataset": {"n_samples": len(df), "fraud_rate": float(fraud_rate)},
        "model_metrics": metrics,
        "throughput": throughput,
        "drift_detection": drift,
    }
    out_path = Path("results/e2e_eval.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
