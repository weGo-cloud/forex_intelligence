#!/usr/bin/env python3
"""Example: run an ORB backtest against ingested mock data.

Usage:
    python scripts/run_example_backtest.py

Uses the mock provider (no real data vendor or network access required)
purely to demonstrate the M1 -> M2 flow end to end: ingest a small
deterministic candle set, run the ORB strategy through the backtest
engine, and print the standardized result. This is a demonstration
script, not a research tool — a real research run should load a real
provider's historical data over a meaningful date range.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from app.backtesting.config import BacktestConfig
from app.backtesting.runner import RunSpec, run_backtest
from app.core.database import get_session_factory
from app.data.ingestion import run_ingestion
from app.data.providers.mock import MockMarketDataProvider
from app.data.schemas import InstrumentCreate
from app.data.timeframes import Timeframe
from app.strategies.orb import ORBConfig, ORBStrategy


def main() -> int:
    session_factory = get_session_factory()
    session = session_factory()

    try:
        start = datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 18, 0, tzinfo=timezone.utc)

        provider = MockMarketDataProvider(base_price=34000.0)
        ingestion_result = run_ingestion(
            session,
            provider,
            symbol="US30",
            timeframe=Timeframe.M5,
            start=start,
            end=end,
            instrument_defaults=InstrumentCreate(
                symbol="US30", asset_class="index", base_currency="US30", quote_currency="USD"
            ),
        )
        print(f"Ingestion: {ingestion_result.status}, accepted={ingestion_result.accepted}")

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

        result = run_backtest(session, strategy, run_spec, backtest_config)

        print(f"\nStrategy: {result.metrics.strategy_name}")
        print(f"Trades: {result.metrics.trade_count}")
        print(f"Win rate: {result.metrics.win_rate}")
        print(f"Expectancy: {result.metrics.expectancy}")
        print(f"Total return: {result.metrics.total_return_pct:.2f}%")
        print(f"Max drawdown: {result.metrics.max_drawdown_pct:.2f}%")
        print(f"\nAssumptions:")
        for a in result.assumptions:
            print(f"  - {a}")

        session.commit()
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
