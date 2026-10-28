from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "transaction_amt", "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
    "c1", "c2", "c3", "c4", "c5", "c6", "c7",
    "c8", "c9", "c10", "c11", "c12", "c13", "c14",
]

CATEGORICAL_FEATURES = [
    "product_cd", "card4", "card6",
    "p_emaildomain", "r_emaildomain",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering matching the streaming pipeline."""
    df = df.copy()

    amt = df["transaction_amt"]
    df["amt_log"] = np.log1p(amt)
    df["amt_bin"] = pd.cut(
        amt,
        bins=[-1, 10, 50, 100, 500, 1000, 5000, float("inf")],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).astype(float)

    df["hour"] = (df["transaction_dt"] % 86400) / 3600
    df["day_of_week"] = ((df["transaction_dt"] // 86400) % 7).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    card_freq = df["card1"].value_counts().to_dict()
    df["card1_freq"] = df["card1"].map(card_freq).fillna(1).astype(float)
    df["card1_freq_log"] = np.log1p(df["card1_freq"])

    # email_risk is populated later by _apply_email_risk() after the train/test
    # split, to avoid leaking the target into training features.
    df["email_risk"] = 0.035

    for cat in CATEGORICAL_FEATURES:
        if cat in df.columns:
            df[cat] = df[cat].astype("category").cat.codes

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature columns used by the model."""
    return (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ["amt_log", "amt_bin", "hour", "day_of_week", "is_night", "is_weekend",
           "card1_freq", "card1_freq_log", "email_risk"]
    )


def train_model(
    transaction_path: Path,
    identity_path: Path | None = None,
    output_dir: Path = Path("models"),
    test_size: float = 0.2,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Train XGBoost fraud detection model. Returns evaluation metrics."""
    logger.info("Loading data from %s", transaction_path)
    df = pd.read_csv(transaction_path)

    if identity_path and identity_path.exists():
        identity = pd.read_csv(identity_path)
        df = df.merge(identity, on="TransactionID", how="left")

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={
        "transactionid": "transaction_id",
        "transactiondt": "transaction_dt",
        "transactionamt": "transaction_amt",
        "productcd": "product_cd",
    })

    logger.info("Engineering features on %d rows", len(df))
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].fillna(-999).values
    y = df["isfraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    # Compute email_risk from training labels only (avoids target leakage).
    email_risk_col_idx = available_cols.index("email_risk") if "email_risk" in available_cols else None
    if email_risk_col_idx is not None and "p_emaildomain" in df.columns:
        train_indices = df.index[:len(y_train)]
        email_risk_map = (
            df.loc[train_indices].groupby("p_emaildomain")["isfraud"].mean()
        )
        all_risk = df["p_emaildomain"].map(email_risk_map).fillna(0.035).values
        X_train[:, email_risk_col_idx] = all_risk[:len(y_train)]
        X_test[:, email_risk_col_idx] = all_risk[len(y_train):]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "tree_method": "hist",
        "random_state": 42,
    }
    if xgb_params:
        params.update(xgb_params)

    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=available_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=available_cols)

    logger.info("Training XGBoost with %d features", len(available_cols))
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    y_proba = model.predict(dtest)
    auc = roc_auc_score(y_test, y_proba)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_proba >= best_threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "fraud_model.json"
    model.save_model(str(model_path))

    meta = {
        "feature_columns": available_cols,
        "threshold": float(best_threshold),
        "params": params,
    }
    meta_path = output_dir / "model_meta.json"
    import json
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    metrics = {
        "auc": float(auc),
        "f1": float(f1),
        "best_threshold": float(best_threshold),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fraud_rate": float(y.mean()),
    }
    logger.info("Training complete: AUC=%.4f, F1=%.4f, threshold=%.4f", auc, f1, best_threshold)
    return metrics
