from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType


def add_amount_features(df: DataFrame) -> DataFrame:
    """Add log-transformed amount and binned amount features."""
    df = df.withColumn("amt_log", F.log1p(F.col("transaction_amt")))

    df = df.withColumn(
        "amt_bin",
        F.when(F.col("transaction_amt") <= 10, 0.0)
        .when(F.col("transaction_amt") <= 50, 1.0)
        .when(F.col("transaction_amt") <= 100, 2.0)
        .when(F.col("transaction_amt") <= 500, 3.0)
        .when(F.col("transaction_amt") <= 1000, 4.0)
        .when(F.col("transaction_amt") <= 5000, 5.0)
        .otherwise(6.0),
    )

    return df


def add_time_features(df: DataFrame) -> DataFrame:
    """Derive time-of-day and day-of-week features from transaction_dt."""
    df = df.withColumn(
        "hour",
        (F.col("transaction_dt") % F.lit(86400)) / F.lit(3600),
    )
    df = df.withColumn(
        "day_of_week",
        (F.floor(F.col("transaction_dt") / F.lit(86400)) % F.lit(7)).cast(IntegerType()),
    )
    df = df.withColumn(
        "is_night",
        F.when(
            (F.col("hour") < 6) | (F.col("hour") > 22), 1
        ).otherwise(0).cast(IntegerType()),
    )
    df = df.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week") >= 5, 1).otherwise(0).cast(IntegerType()),
    )

    return df


def add_card_frequency(df: DataFrame) -> DataFrame:
    """Use global-mean card frequency (1.0) as a streaming default.

    The batch training pipeline computes per-card1 frequencies from the full
    training set. In streaming, we default to 1.0 (neutral). A production
    system would broadcast a lookup table updated from the latest training
    window via a Spark broadcast variable or a side-input join.
    """
    # TODO: replace with a broadcast lookup from the training window
    df = df.withColumn("card1_freq", F.lit(1.0).cast(DoubleType()))
    df = df.withColumn("card1_freq_log", F.lit(0.0).cast(DoubleType()))
    return df


def add_email_risk(df: DataFrame) -> DataFrame:
    """Use the global fraud rate (0.035) as a streaming default for email risk.

    The batch training pipeline computes per-domain fraud rates from training
    labels. In streaming, we fall back to the population mean (3.5%). A
    production system would maintain a lookup table updated from the latest
    training window.
    """
    # TODO: replace with a broadcast lookup from the training window
    df = df.withColumn("email_risk", F.lit(0.035).cast(DoubleType()))
    return df


def encode_categoricals(df: DataFrame) -> DataFrame:
    """Encode categorical columns as integer codes for XGBoost."""
    cat_mappings = {
        "product_cd": {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4},
        "card4": {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3},
        "card6": {"debit": 0, "credit": 1, "charge card": 2, "debit or credit": 3},
    }

    for col, mapping in cat_mappings.items():
        if col in df.columns:
            map_expr = F.lit(-999.0)
            for key, val in mapping.items():
                map_expr = F.when(
                    F.lower(F.col(col)) == key, float(val)
                ).otherwise(map_expr)
            df = df.withColumn(col, map_expr.cast(DoubleType()))

    for col in ["p_emaildomain", "r_emaildomain"]:
        if col in df.columns:
            df = df.withColumn(col, F.lit(-999.0).cast(DoubleType()))

    return df


def engineer_all_features(df: DataFrame) -> DataFrame:
    """Apply the full feature engineering pipeline to a streaming DataFrame."""
    df = add_amount_features(df)
    df = add_time_features(df)
    df = add_card_frequency(df)
    df = add_email_risk(df)
    df = encode_categoricals(df)
    return df
