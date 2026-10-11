"""ORM models for the market-data schema.

Scope for M1: instruments, candles, and ingestion provenance only.
No strategy/backtesting/probability/ranking tables are introduced here
(see docs/M1_DATA_FOUNDATION.md for the reasoning).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """A DateTime column that is always UTC and always timezone-aware.

    Different backends behave differently here: PostgreSQL's
    TIMESTAMPTZ preserves tzinfo across a round trip, but SQLite (used
    in the test suite) does not. Rather than let application code
    special-case the backend, this type normalizes every value to UTC
    before storage and re-attaches UTC tzinfo on load unconditionally,
    so `candle.timestamp.tzinfo` is always UTC regardless of which
    database is in use.

    Naive datetimes are rejected outright rather than assumed to be UTC
    — the canonical timestamp policy for this system is "UTC and
    explicit," not "UTC by convention."
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are not allowed here; timestamps must be UTC-aware.")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class AssetClass(str, enum.Enum):
    """Supported instrument asset classes.

    Only FOREX is used in M1. The field is a plain string column (not a
    DB-level enum constraint) so a future milestone can add asset
    classes without a schema migration to alter a fixed enum type.
    """

    FOREX = "forex"


class IngestionStatus(str, enum.Enum):
    """Overall outcome of a single ingestion run."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class Instrument(Base):
    """A tradable instrument (e.g. an EUR/USD forex pair)."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False, default=AssetClass.FOREX.value)
    base_currency: Mapped[str] = mapped_column(String(8), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(8), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    candles: Mapped[list["Candle"]] = relationship(back_populates="instrument", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Instrument(id={self.id}, symbol={self.symbol!r})"


class Candle(Base):
    """A single OHLCV candle, uniquely identified by (instrument, timeframe, timestamp)."""

    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("instrument_id", "timeframe", "timestamp", name="uq_candle_identity"),
        Index("ix_candle_lookup", "instrument_id", "timeframe", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(4), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    open: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    # Nullable: spot forex has no centralized-exchange volume. Whatever
    # (if anything) the selected provider supplies is stored as-is; this
    # column makes no claim about what "volume" means for a given source.
    volume: Mapped[Optional[Decimal]] = mapped_column(Numeric(24, 8), nullable=True)

    source: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow)

    instrument: Mapped["Instrument"] = relationship(back_populates="candles")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Candle(instrument_id={self.instrument_id}, timeframe={self.timeframe}, timestamp={self.timestamp!r})"


class IngestionRun(Base):
    """Provenance record for a single provider -> ingestion pipeline pass.

    Exists so datasets are reproducible and auditable: for any stored
    candle range you can find which provider supplied it, when, and
    what was accepted/rejected/flagged along the way.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider_name: Mapped[str] = mapped_column(String(64), nullable=False)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(4), nullable=False)

    requested_start: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    requested_end: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    run_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow)

    received_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accepted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(String(16), nullable=False)
    # Capped, JSON-serialized list of rejection reasons and gap reports.
    # Deliberately not a normalized child table: at M1 scale this is
    # simpler to query and more than sufficient for investigation, and
    # avoids introducing a table whose shape would need to anticipate
    # every future quality-check type.
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    instrument: Mapped["Instrument"] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"IngestionRun(id={self.id}, provider={self.provider_name!r}, status={self.status!r})"
