#!/usr/bin/env python3
"""The fastest real path from a CSV to a backtest result. No database,
no FastAPI, no registry, no experiment log — just: load data, run
strategy, print numbers. Everything this deliberately skips can be
added later IF it turns out you need it; none of it is required to get
a real answer tonight.

Usage:
    python scripts/quick_backtest.py --csv path/to/data.csv --strategy orb

CSV must have columns: timestamp, open, high, low, close (volume optional).
`timestamp` must be parseable by pandas and either already UTC or naive
(naive is assumed UTC — check this assumption against your data source;
getting timezone wrong silently shifts your session boundaries for ORB).

Where to actually get a CSV tonight, free, no signup hassle:
  - Yahoo Finance via `yfinance` (pip install yfinance) — has daily/intraday
    data for indices like ^DJI (US30-ish), ^GSPC, ^NDX, and most FX pairs
    as e.g. "EURUSD=X". Intraday history is limited to the last ~60 days
    on the free tier, which is fine for a first real backtest.
  - Dukascopy / HistData.com — free historical FX/CFD tick or minute data,
    manual CSV export, no API key needed.
Pick whichever gets you a CSV in the next 10 minutes. Perfect data
provenance is a later problem.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.metrics import compute_metrics
from app.strategies.base import MarketContext
from app.strategies.orb import ORBConfig, ORBStrategy
from app.strategies.rsi_divergence import RSIDivergenceConfig, RSIDivergenceStrategy


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])]
    df = df.astype(float)

    if df.index.has_duplicates:
        dupes = df.index.duplicated().sum()
        print(f"WARNING: {dupes} duplicate timestamps found - dropping duplicates, keeping first.", file=sys.stderr)
        df = df[~df.index.duplicated(keep="first")]

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--strategy", choices=["orb", "rsi_divergence"], required=True)
    parser.add_argument("--symbol", default="INSTRUMENT")
    parser.add_argument("--risk-reward", type=float, default=2.0)
    parser.add_argument("--spread-points", type=float, default=0.0, help="Set this to something realistic for your instrument, not 0 - see docs/M2_BACKTESTING_ENGINE.md")
    args = parser.parse_args()

    df = load_csv(args.csv)
    print(f"Loaded {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    if args.strategy == "orb":
        strategy = ORBStrategy(ORBConfig(instrument_symbol=args.symbol, risk_reward=args.risk_reward))
    else:
        strategy = RSIDivergenceStrategy(RSIDivergenceConfig(instrument_symbol=args.symbol, risk_reward=args.risk_reward))

    market = MarketContext(instrument_symbol=args.symbol, execution_df=df)
    signals = strategy.generate_signals(market)
    print(f"Generated {len(signals)} signals")

    if not signals:
        print("\nNo signals generated. Before assuming the strategy is bad:")
        print("  - Check your CSV's timeframe matches what the strategy expects (ORB defaults to 5-minute bars)")
        print("  - Check enough bars exist per session for the opening range / swing detection to fire at all")
        return 0

    config = BacktestConfig(instrument_symbol=args.symbol, spread_points=args.spread_points, risk_per_trade_pct=1.0)
    trades = BacktestEngine(config).run(df, signals, strategy.name, strategy.params())
    print(f"Executed {len(trades)} trades (some signals may be skipped - see BacktestEngine docstring)")

    if not trades:
        print("\nSignals were generated but zero trades executed - likely every signal landed on")
        print("the last available bar with no next bar to fill on. This is a data-range problem,")
        print("not a strategy problem.")
        return 0

    metrics = compute_metrics(trades, strategy.name, args.symbol, config.initial_equity, df.index[0].to_pydatetime(), df.index[-1].to_pydatetime())

    print(f"\n--- {strategy.name} on {args.symbol} ---")
    print(f"Trades:         {metrics.trade_count}")
    print(f"Win rate:       {metrics.win_rate}")
    print(f"Expectancy:     {metrics.expectancy}")
    print(f"Profit factor:  {metrics.profit_factor}")
    print(f"Total return:   {metrics.total_return_pct:.2f}%")
    print(f"Max drawdown:   {metrics.max_drawdown_pct:.2f}%")
    print(f"Sharpe (trade-level, not annualized): {metrics.sharpe_ratio}")

    print("\nFirst 5 trades (inspect these against your raw data before trusting the summary numbers):")
    for t in trades[:5]:
        print(f"  {t.entry_time} {t.direction} @ {t.entry_price:.4f} -> {t.exit_time} @ {t.exit_price:.4f} "
              f"({t.exit_reason}, P&L {t.net_pnl:+.2f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
