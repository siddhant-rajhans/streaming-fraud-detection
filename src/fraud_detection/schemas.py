from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A single financial transaction from the IEEE-CIS schema."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    transaction_dt: float = Field(..., description="Timedelta from reference datetime (seconds)")
    transaction_amt: float = Field(..., ge=0, description="Transaction amount in USD")
    product_cd: str = Field(..., description="Product code for the transaction")
    card1: int = Field(..., description="Payment card identifier (hashed)")
    card2: Optional[float] = Field(None, description="Card info field 2")
    card3: Optional[float] = Field(None, description="Card info field 3")
    card4: Optional[str] = Field(None, description="Card type (credit/debit)")
    card5: Optional[float] = Field(None, description="Card info field 5")
    card6: Optional[str] = Field(None, description="Card category")
    addr1: Optional[float] = Field(None, description="Billing address part 1")
    addr2: Optional[float] = Field(None, description="Billing address part 2")
    dist1: Optional[float] = Field(None, description="Distance measure 1")
    dist2: Optional[float] = Field(None, description="Distance measure 2")
    p_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    r_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    c1: Optional[float] = None
    c2: Optional[float] = None
    c3: Optional[float] = None
    c4: Optional[float] = None
    c5: Optional[float] = None
    c6: Optional[float] = None
    c7: Optional[float] = None
    c8: Optional[float] = None
    c9: Optional[float] = None
    c10: Optional[float] = None
    c11: Optional[float] = None
    c12: Optional[float] = None
    c13: Optional[float] = None
    c14: Optional[float] = None


class FraudPrediction(BaseModel):
    """Model output for a scored transaction."""

    transaction_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    model_version: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = Field(..., ge=0, description="Scoring latency in milliseconds")


class DriftSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftAlert(BaseModel):
    alert_id: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_name: str
    drift_score: float = Field(..., ge=0.0)
    test_method: str = Field(..., description="PSI, KS, or chi-squared")
    severity: DriftSeverity
    reference_stats: dict
    current_stats: dict
    window_size: int
    retrain_triggered: bool = False


class ValidationResult(BaseModel):
    transaction_id: str
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    null_fields: list[str] = Field(default_factory=list)
    out_of_range_fields: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
