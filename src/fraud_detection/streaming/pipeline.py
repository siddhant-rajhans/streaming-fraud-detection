from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import xgboost as xgb
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from fraud_detection.streaming.feature_engineering import engineer_all_features

logger = logging.getLogger(__name__)

TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("transaction_dt", DoubleType(), False),
    StructField("transaction_amt", DoubleType(), False),
    StructField("product_cd", StringType(), False),
    StructField("card1", IntegerType(), False),
    StructField("card2", DoubleType(), True),
    StructField("card3", DoubleType(), True),
    StructField("card4", StringType(), True),
    StructField("card5", DoubleType(), True),
    StructField("card6", StringType(), True),
    StructField("addr1", DoubleType(), True),
    StructField("addr2", DoubleType(), True),
    StructField("dist1", DoubleType(), True),
    StructField("dist2", DoubleType(), True),
    StructField("p_emaildomain", StringType(), True),
    StructField("r_emaildomain", StringType(), True),
    *[StructField(f"c{i}", DoubleType(), True) for i in range(1, 15)],
])

OUTPUT_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("fraud_probability", DoubleType(), False),
    StructField("is_fraud", IntegerType(), False),
    StructField("model_version", StringType(), False),
])


def create_spark_session(app_name: str = "FraudDetection") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.backpressure.enabled", "true")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/fraud-checkpoint")
        .getOrCreate()
    )


def read_kafka_stream(
    spark: SparkSession,
    bootstrap_servers: str = "localhost:9092",
    topic: str = "transactions",
) -> DataFrame:
    """Read a streaming DataFrame from a Kafka topic."""
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw.select(
            F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("txn"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .select("txn.*", "kafka_timestamp")
    )

    return parsed


def _score_batch_udf(model_path: str, feature_columns: list[str], threshold: float):
    """Create a pandas UDF that scores a batch of transactions with XGBoost."""
    from pyspark.sql.functions import pandas_udf
    import pandas as pd

    @pandas_udf(DoubleType())
    def predict_fraud(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
        booster = xgb.Booster()
        booster.load_model(model_path)

        for batch in iterator:
            features = batch[feature_columns].fillna(-999.0).values.astype(np.float32)
            dmat = xgb.DMatrix(features, feature_names=feature_columns)
            probas = booster.predict(dmat)
            yield pd.Series(probas)

    return predict_fraud


def validate_transactions(df: DataFrame) -> DataFrame:
    """Filter out transactions that fail basic validation."""
    return (
        df
        .filter(F.col("transaction_id").isNotNull())
        .filter(F.col("transaction_amt") >= 0)
        .filter(F.col("transaction_amt") < 100_000)
        .filter(F.col("product_cd").isNotNull())
    )


def build_scoring_pipeline(
    spark: SparkSession,
    model_dir: Path = Path("models"),
    bootstrap_servers: str = "localhost:9092",
    input_topic: str = "transactions",
    output_topic: str = "predictions",
    model_version: str = "v1",
) -> Any:
    """Build and return the streaming query for the fraud scoring pipeline."""
    model_path = str(model_dir / "fraud_model.json")
    meta_path = model_dir / "model_meta.json"

    meta = json.loads(meta_path.read_text())
    feature_columns = meta["feature_columns"]
    threshold = meta["threshold"]

    raw_stream = read_kafka_stream(spark, bootstrap_servers, input_topic)

    validated = validate_transactions(raw_stream)

    featured = engineer_all_features(validated)

    available_cols = [c for c in feature_columns if c in featured.columns]
    feature_df = featured.select(
        "transaction_id",
        *[F.col(c).cast(DoubleType()).alias(c) for c in available_cols],
    ).fillna(-999.0)

    predict_udf = _score_batch_udf(model_path, available_cols, threshold)

    scored = (
        feature_df
        .withColumn("fraud_probability", predict_udf(F.struct(*available_cols)))
        .withColumn("is_fraud", F.when(F.col("fraud_probability") >= threshold, 1).otherwise(0))
        .withColumn("model_version", F.lit(model_version))
        .select("transaction_id", "fraud_probability", "is_fraud", "model_version")
    )

    output_value = F.to_json(F.struct(
        "transaction_id", "fraud_probability", "is_fraud", "model_version",
    ))

    query = (
        scored
        .select(
            F.col("transaction_id").alias("key"),
            output_value.alias("value"),
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("topic", output_topic)
        .option("checkpointLocation", f"/tmp/fraud-checkpoint/{output_topic}")
        .outputMode("append")
        .trigger(processingTime="1 second")
        .start()
    )

    logger.info("Streaming pipeline started: %s -> %s", input_topic, output_topic)
    return query


def run_pipeline(
    model_dir: Path = Path("models"),
    bootstrap_servers: str = "localhost:9092",
    input_topic: str = "transactions",
    output_topic: str = "predictions",
) -> None:
    """Start the streaming pipeline and block until termination."""
    spark = create_spark_session()
    query = build_scoring_pipeline(
        spark, model_dir, bootstrap_servers, input_topic, output_topic,
    )
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted, shutting down")
        query.stop()
    finally:
        spark.stop()
