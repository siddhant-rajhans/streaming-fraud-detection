from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from fraud_detection.schemas import Transaction

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "transactions"


def load_ieee_cis_data(
    transaction_path: Path,
    identity_path: Path | None = None,
) -> pd.DataFrame:
    """Load and merge IEEE-CIS transaction and identity tables."""
    transactions = pd.read_csv(transaction_path)

    if identity_path and identity_path.exists():
        identity = pd.read_csv(identity_path)
        transactions = transactions.merge(identity, on="TransactionID", how="left")

    transactions.columns = [c.lower().replace("transaction", "transaction_") if c.startswith("Transaction") else c.lower() for c in transactions.columns]
    transactions = transactions.rename(columns={
        "transaction_id": "transaction_id",
        "transaction_dt": "transaction_dt",
        "transaction_amt": "transaction_amt",
        "productcd": "product_cd",
        "p_emaildomain": "p_emaildomain",
        "r_emaildomain": "r_emaildomain",
    })

    return transactions


def create_producer(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    retries: int = 5,
    retry_delay: float = 2.0,
) -> KafkaProducer:
    """Create a Kafka producer with retry logic."""
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=1,
            )
            logger.info("Connected to Kafka at %s", bootstrap_servers)
            return producer
        except NoBrokersAvailable:
            if attempt < retries - 1:
                logger.warning(
                    "Kafka not available, retrying in %.1fs (%d/%d)",
                    retry_delay, attempt + 1, retries,
                )
                time.sleep(retry_delay)
            else:
                raise


def row_to_transaction_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a transaction dictionary suitable for Kafka."""
    txn_fields = {
        "transaction_id": str(int(row.get("transaction_id", 0))),
        "transaction_dt": float(row.get("transaction_dt", 0)),
        "transaction_amt": float(row.get("transaction_amt", 0)),
        "product_cd": str(row.get("product_cd", "W")),
        "card1": int(row.get("card1", 0)),
    }

    optional_fields = [
        "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "dist2",
        "p_emaildomain", "r_emaildomain",
    ]
    for field in optional_fields:
        val = row.get(field)
        if pd.notna(val):
            txn_fields[field] = val if isinstance(val, str) else float(val)

    for i in range(1, 15):
        key = f"c{i}"
        val = row.get(key)
        if pd.notna(val):
            txn_fields[key] = float(val)

    return txn_fields


def stream_transactions(
    df: pd.DataFrame,
    replay_speed: float = 1.0,
) -> Iterator[dict]:
    """Yield transaction dicts, pausing between rows to simulate real-time replay."""
    df = df.sort_values("transaction_dt").reset_index(drop=True)

    prev_dt = None
    for _, row in df.iterrows():
        txn = row_to_transaction_dict(row)
        current_dt = row["transaction_dt"]

        if prev_dt is not None and replay_speed > 0:
            dt_diff = (current_dt - prev_dt) / replay_speed
            if 0 < dt_diff < 5.0:
                time.sleep(dt_diff)

        prev_dt = current_dt
        yield txn


def replay_dataset(
    transaction_path: Path,
    identity_path: Path | None = None,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    topic: str = DEFAULT_TOPIC,
    replay_speed: float = 1000.0,
    max_records: int | None = None,
) -> int:
    """Replay the IEEE-CIS dataset as Kafka events. Returns count of records sent."""
    df = load_ieee_cis_data(transaction_path, identity_path)
    if max_records:
        df = df.head(max_records)

    producer = create_producer(bootstrap_servers)
    count = 0

    try:
        for txn in stream_transactions(df, replay_speed):
            producer.send(
                topic,
                key=txn["transaction_id"],
                value=txn,
            )
            count += 1
            if count % 10000 == 0:
                logger.info("Sent %d transactions", count)
                producer.flush()
    finally:
        producer.flush()
        producer.close()

    logger.info("Replay complete: %d transactions sent to topic '%s'", count, topic)
    return count
