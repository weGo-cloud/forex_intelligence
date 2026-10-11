"""Application-level access to stored market data.

Callers (ingestion pipeline, API routes, future strategy/backtesting
code) interact with these repositories using domain concepts —
instrument, timeframe, date range — and never touch SQLAlchemy/
PostgreSQL details directly. This keeps database infrastructure
separable from market-data domain logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.models import Candle, IngestionRun, Instrument
from app.data.schemas import CanonicalCandle, InstrumentCreate
from app.data.timeframes import Timeframe


class InstrumentRepository:
    """Persistence access for `Instrument` rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_symbol(self, symbol: str) -> Optional[Instrument]:
        stmt = select(Instrument).where(Instrument.symbol == symbol)
        return self._session.scalar(stmt)

    def get_or_create(self, data: InstrumentCreate) -> Instrument:
        existing = self.get_by_symbol(data.symbol)
        if existing is not None:
            return existing

        instrument = Instrument(
            symbol=data.symbol,
            asset_class=data.asset_class,
            base_currency=data.base_currency,
            quote_currency=data.quote_currency,
            is_active=data.is_active,
            extra_metadata=data.extra_metadata,
        )
        self._session.add(instrument)
        self._session.flush()
        return instrument

    def list_active(self) -> Sequence[Instrument]:
        stmt = select(Instrument).where(Instrument.is_active.is_(True)).order_by(Instrument.symbol)
        return self._session.scalars(stmt).all()


class CandleRepository:
    """Persistence and query access for `Candle` rows.

    Duplicate prevention is implemented at the application level (query
    existing keys, skip matches) rather than relying on a
    PostgreSQL-specific `ON CONFLICT` clause. This keeps the same code
    path correct against both PostgreSQL in production and SQLite in
    the test suite. The DB-level unique constraint on
    (instrument_id, timeframe, timestamp) remains as a safety net
    against races between concurrent ingestion runs.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_existing_timestamps(
        self,
        instrument_id: int,
        timeframe: Timeframe,
        timestamps: Sequence[datetime],
    ) -> set[datetime]:
        if not timestamps:
            return set()
        stmt = select(Candle.timestamp).where(
            Candle.instrument_id == instrument_id,
            Candle.timeframe == timeframe.value,
            Candle.timestamp.in_(timestamps),
        )
        return set(self._session.scalars(stmt).all())

    def bulk_insert(self, instrument_id: int, candles: list[CanonicalCandle]) -> tuple[int, int]:
        """Insert candles not already present for this instrument/timeframe.

        `candles` must all share the same timeframe. Returns
        (inserted_count, duplicate_count). Safe to call repeatedly with
        overlapping ranges — re-ingesting an already-stored range
        inserts nothing new and reports the overlap as duplicates.
        """
        if not candles:
            return 0, 0

        timeframe = candles[0].timeframe
        timestamps = [c.timestamp for c in candles]
        existing = self.get_existing_timestamps(instrument_id, timeframe, timestamps)

        inserted = 0
        duplicates = 0
        for candle in candles:
            if candle.timestamp in existing:
                duplicates += 1
                continue
            self._session.add(
                Candle(
                    instrument_id=instrument_id,
                    timeframe=candle.timeframe.value,
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    source=candle.source,
                )
            )
            existing.add(candle.timestamp)
            inserted += 1

        self._session.flush()
        return inserted, duplicates

    def query(
        self,
        instrument_id: int,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Return stored candles for `[start, end)`, ascending by timestamp."""
        stmt = (
            select(Candle)
            .where(
                Candle.instrument_id == instrument_id,
                Candle.timeframe == timeframe.value,
                Candle.timestamp >= start,
                Candle.timestamp < end,
            )
            .order_by(Candle.timestamp)
        )
        return list(self._session.scalars(stmt).all())


class IngestionRunRepository:
    """Persistence access for ingestion provenance records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, **fields: Any) -> IngestionRun:
        run = IngestionRun(**fields)
        self._session.add(run)
        self._session.flush()
        return run
