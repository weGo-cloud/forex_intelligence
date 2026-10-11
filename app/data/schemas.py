"""Pydantic schemas for the market-data domain.

`CanonicalCandle` is the one shape every provider's raw data gets
normalized into before validation and persistence. Strategy,
backtesting, and other future modules should consume canonical data
(or ORM rows), never provider-specific structures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from app.data.timeframes import Timeframe


class CanonicalCandle(BaseModel):
    """A single OHLCV candle in canonical internal form.

    Timestamps are always UTC and timezone-aware; prices are always
    `Decimal` to avoid float rounding error creeping into stored market
    data. This is the shape produced by normalization and consumed by
    validation and persistence.
    """

    model_config = ConfigDict(frozen=True)

    instrument_symbol: str
    timeframe: Timeframe
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Optional[Decimal] = None
    source: Optional[str] = None

    @field_validator("timestamp")
    @classmethod
    def _require_utc_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (UTC); naive datetimes are rejected")
        return value.astimezone(timezone.utc)


class ValidationIssue(BaseModel):
    """A single data-quality problem found in a candle or batch."""

    code: str
    message: str
    candle_timestamp: Optional[datetime] = None


class GapIssue(BaseModel):
    """A larger-than-expected interval between two consecutive candles.

    This is informational only. Weekend/holiday closures produce gaps
    that are entirely legitimate; this record exists so a human (or a
    later milestone) can investigate, not so the pipeline can reject or
    manufacture data automatically.
    """

    after: datetime
    before: datetime
    expected_interval_minutes: int
    missing_periods: int


class InstrumentCreate(BaseModel):
    """Input shape for registering a new instrument."""

    symbol: str
    asset_class: str = "forex"
    base_currency: str
    quote_currency: str
    is_active: bool = True
    extra_metadata: dict[str, Any] = {}


class InstrumentRead(BaseModel):
    """Output shape for an instrument, safe to return over the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    asset_class: str
    base_currency: str
    quote_currency: str
    is_active: bool


class CandleRead(BaseModel):
    """Output shape for a stored candle, safe to return over the API."""

    model_config = ConfigDict(from_attributes=True)

    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Optional[Decimal] = None
    source: Optional[str] = None


class IngestionResult(BaseModel):
    """Summary of a single ingestion run, returned to the caller."""

    provider_name: str
    instrument_symbol: str
    timeframe: Timeframe
    requested_start: datetime
    requested_end: datetime
    received: int
    accepted: int
    rejected: int
    duplicates: int
    gaps: list[GapIssue] = []
    rejection_reasons: list[ValidationIssue] = []
    status: str
    run_id: Optional[int] = None
