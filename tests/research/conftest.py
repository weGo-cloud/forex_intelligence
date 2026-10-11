from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base
from app.data import models as _data_models  # noqa: F401  (registers tables on Base.metadata)
from app.data.ingestion import run_ingestion
from app.data.providers.base import ProviderCandle
from app.data.providers.mock import MockMarketDataProvider
from app.data.schemas import InstrumentCreate
from app.data.timeframes import Timeframe
from app.research import registry as _research_models  # noqa: F401  (registers tables on Base.metadata)

import app.strategies.orb  # noqa: F401  (registers "orb" in the strategy registry)
import app.strategies.rsi_divergence  # noqa: F401  (registers "rsi_divergence")
import app.strategies.mean_reversion  # noqa: F401  (registers "bb_stoch_mean_reversion")
import app.strategies.liquidity_sweep_fvg  # noqa: F401  (registers "liquidity_sweep_fvg")


def utc(*args, **kwargs) -> datetime:
    return datetime(*args, tzinfo=timezone.utc, **kwargs)


@pytest.fixture
def db_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def orb_breakout_candles(start: datetime, count: int = 40) -> list[ProviderCandle]:
    """A repeating clean-breakout-and-run pattern, long enough to
    generate several trades across a multi-day/multi-window range -
    used by sweep/walk-forward/robustness tests that need more than one
    trade to exercise their aggregation logic."""
    candles: list[ProviderCandle] = []
    price = 34000.0
    for i in range(count):
        cycle_pos = i % 7
        pattern = [
            (0, 10, -5, 5),
            (5, 15, -2, 8),
            (8, 12, 0, 6),
            (6, 30, 4, 28),
            (28, 60, 26, 55),
            (55, 80, 50, 75),
            (75, 100, 70, 95),
        ][cycle_pos]
        o, h, low, c = (price + pattern[0], price + pattern[1], price + pattern[2], price + pattern[3])
        candles.append(
            ProviderCandle(
                raw_timestamp=start + timedelta(minutes=5 * i),
                raw_open=o,
                raw_high=h,
                raw_low=low,
                raw_close=c,
                raw_volume=1000,
            )
        )
        if cycle_pos == 6:
            price = c  # carry the level forward so each cycle starts near where the last ended
    return candles


@pytest.fixture
def seeded_eurusd(db_session: Session):
    """Seeds enough 5-minute EURUSD candles to run ORB backtests across
    several days - returns (start, end) of the seeded range."""
    start = utc(2024, 1, 2, 0, 0)
    candles = orb_breakout_candles(start, count=40 * 6 * 10)  # ~10 "days" worth of cycles
    end = start + timedelta(minutes=5 * len(candles))
    provider = MockMarketDataProvider(fixed_candles=candles)
    result = run_ingestion(
        db_session,
        provider,
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        start=start,
        end=end,
        instrument_defaults=InstrumentCreate(
            symbol="EURUSD", asset_class="forex", base_currency="EUR", quote_currency="USD"
        ),
    )
    assert result.status == "success"
    return start, end
