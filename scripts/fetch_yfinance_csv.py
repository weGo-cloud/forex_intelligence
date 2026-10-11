#!/usr/bin/env python3
"""Download real historical price data and save it as a CSV in the
exact shape scripts/quick_backtest.py expects. This is the "where do I
get a CSV" answer — Yahoo Finance, via the `yfinance` package, free,
no signup, no API key.

Usage:
    python scripts/fetch_yfinance_csv.py --ticker EURUSD=X --period 5y --interval 1d --out data.csv

Common tickers:
    EURUSD=X   Euro / US Dollar
    GBPUSD=X   British Pound / US Dollar
    ^GSPC      S&P 500 index
    ^DJI       Dow Jones (closest free proxy to "US30")
    GC=F       Gold futures

`--interval` note: Yahoo Finance only keeps INTRADAY data (1m/5m/15m/etc)
for the last ~60 days, no matter how far back you ask. Daily data ("1d")
goes back years with no such limit. For a first backtest, use daily
data with the RSI Divergence strategy (it doesn't need session/intraday
logic) - it's the simplest path to a real, meaningful result tonight.
Save the ORB strategy (which needs intraday bars and a specific session)
for once you've got one working result and want a second data point.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True, help="e.g. EURUSD=X, ^GSPC, ^DJI")
    parser.add_argument("--period", default="5y", help="How far back: 1y, 2y, 5y, max, etc.")
    parser.add_argument("--interval", default="1d", help="1d for daily (recommended to start). Intraday only works for the last ~60 days.")
    parser.add_argument("--out", default="data.csv")
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Run: pip install yfinance", file=sys.stderr)
        return 1

    print(f"Downloading {args.ticker} ({args.period}, {args.interval} bars)...")
    df = yf.download(args.ticker, period=args.period, interval=args.interval, progress=False)

    if df.empty:
        print(f"No data returned for {args.ticker!r}. Check the ticker symbol and --period/--interval combination.", file=sys.stderr)
        return 1

    # yfinance can return either flat columns or a MultiIndex (when it
    # thinks you might be asking for multiple tickers) - flatten defensively.
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index.name = "timestamp"
    df = df[["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])]

    df.to_csv(args.out)
    print(f"Saved {len(df)} rows to {args.out}")
    print(f"Range: {df.index[0]} -> {df.index[-1]}")
    print(f"\nNext step:\n  python scripts/quick_backtest.py --csv {args.out} --strategy rsi_divergence --symbol {args.ticker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
