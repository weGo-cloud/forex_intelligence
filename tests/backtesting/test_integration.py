"""End-to-end test: M1 ingestion pipeline -> M2 data bridge -> strategy
-> engine -> metrics, using the mock provider against an in-memory
SQLite database (same approach M1's own ingestion tests use). This is
the test that proves M2 actually composes with M1 as built, not just
against an idealized interface.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from app.backtesting.config import BacktestConfig
from app.backtesting.results import BacktestResult
from app.backtesting.runner import RunSpec, run_backtest
from app.data.ingestion import run_ingestion
from app.data.providers.base import ProviderCandle
from app.data.providers.mock import MockMarketDataProvider
from app.data.schemas import InstrumentCreate
from app.data.timeframes import Timeframe
from app.strategies.orb import ORBConfig, ORBStrategy
from tests.backtesting.conftest import utc


def _orb_breakout_candles() -> list[ProviderCandle]:
    """5-minute US30 candles for one NY session (2024-01-02, winter =
    UTC-5, so 09:30 NY = 14:30 UTC): a clean 15-minute opening range
    followed by an unambiguous close-based breakout, then a clean run to
    the take-profit level so the integration test has a deterministic,
    non-degenerate trade to inspect end to end."""
    start = utc(2024, 1, 2, 14, 30)
    # (open, high, low, close) per 5-minute bar
    bars = [
        (34000, 34010, 33995, 34005),  # opening range bar 1
        (34005, 34015, 33998, 34008),  # opening range bar 2
        (34008, 34012, 34000, 34006),  # opening range bar 3 (OR high=34015, low=33995)
        (34006, 34030, 34004, 34028),  # breakout bar: close 34028 > OR high 34015
        (34028, 34060, 34026, 34055),  # continuation toward TP
        (34055, 34080, 34050, 34075),
        (34075, 34100, 34070, 34095),
    ]
    candles = []
    for i, (o, h, low, c) in enumerate(bars):
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
    return candles


def test_ingested_data_flows_through_orb_backtest_end_to_end(db_session: Session) -> None:
    provider = MockMarketDataProvider(fixed_candles=_orb_breakout_candles())
    start = utc(2024, 1, 2, 14, 30)
    end = utc(2024, 1, 2, 15, 10)

    ingestion_result = run_ingestion(
        db_session,
        provider,
        symbol="US30",
        timeframe=Timeframe.M5,
        start=start,
        end=end,
        instrument_defaults=InstrumentCreate(symbol="US30", asset_class="index", base_currency="US30", quote_currency="USD"),
    )
    assert ingestion_result.status == "success"
    assert ingestion_result.accepted == 7

    strategy = ORBStrategy(
        ORBConfig(
            instrument_symbol="US30",
            opening_range_minutes=15,
            execution_timeframe_minutes=5,
            risk_reward=2.0,
        )
    )
    backtest_config = BacktestConfig(instrument_symbol="US30", spread_points=1.0, risk_per_trade_pct=1.0)
    run_spec = RunSpec(symbol="US30", timeframe=Timeframe.M5, start=start, end=end)

    result = run_backtest(db_session, strategy, run_spec, backtest_config)

    assert isinstance(result, BacktestResult)
    assert result.metrics.instrument_symbol == "US30"
    assert result.metrics.strategy_name == "orb"
    assert result.metrics.trade_count == 1
    assert result.trades[0].direction == "long"
    assert result.trades[0].strategy_params["risk_reward"] == 2.0
    # Reproducibility requirement: the exact config used travels with the result.
    assert result.backtest_config["spread_points"] == 1.0
    assert len(result.assumptions) > 0


def test_unknown_symbol_raises_rather_than_silently_returning_empty(db_session: Session) -> None:
    from app.backtesting.data_access import UnknownInstrumentError

    strategy = ORBStrategy(ORBConfig(instrument_symbol="GBP/JPY"))
    run_spec = RunSpec(
        symbol="GBP/JPY",
        timeframe=Timeframe.M5,
        start=utc(2024, 1, 1),
        end=utc(2024, 1, 2),
    )
    backtest_config = BacktestConfig(instrument_symbol="GBP/JPY")

    try:
        run_backtest(db_session, strategy, run_spec, backtest_config)
        assert False, "expected UnknownInstrumentError"
    except UnknownInstrumentError:
        pass
