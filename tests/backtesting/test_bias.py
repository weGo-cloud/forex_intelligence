"""Tests specifically designed to catch accidental future-data usage,
per the M2 spec's explicit requirement for a dedicated bias-test suite.

Each test follows the same pattern: run the system on data up to time T,
then run it again on a longer series that shares an identical prefix up
to T but differs afterward, and assert the result up to T is byte-for-
byte identical. If a result at time T changed because of what happens
after T, that is look-ahead bias by definition.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.backtesting.data_access import align_higher_timeframe
from app.strategies.base import MarketContext
from app.strategies.orb import ORBConfig, ORBStrategy
from app.strategies.rsi_divergence import RSIDivergenceConfig, RSIDivergenceStrategy
from app.strategies.mean_reversion import BBStochMeanReversionConfig, BBStochMeanReversionStrategy


def _make_htf_leakage_frame():
    htf_idx = pd.to_datetime(["2024-01-01 09:00", "2024-01-01 10:00"], utc=True)
    htf_df = pd.DataFrame({"close": [100.0, 105.0]}, index=htf_idx)
    base_idx = pd.date_range("2024-01-01 09:00", "2024-01-01 10:30", freq="5min", tz="UTC")
    return base_idx, htf_df


class TestHigherTimeframeAlignmentBias:
    def test_unclosed_htf_candle_is_not_visible(self) -> None:
        base_idx, htf_df = _make_htf_leakage_frame()
        aligned = align_higher_timeframe(base_idx, htf_df, htf_minutes=60)
        # The 09:00 1H candle covers [09:00, 10:00) and is not closed until 10:00.
        for ts in base_idx[base_idx < pd.Timestamp("2024-01-01 10:00", tz="UTC")]:
            assert pd.isna(aligned.loc[ts, "close_htf"])

    def test_htf_candle_visible_exactly_at_its_close(self) -> None:
        base_idx, htf_df = _make_htf_leakage_frame()
        aligned = align_higher_timeframe(base_idx, htf_df, htf_minutes=60)
        assert aligned.loc[pd.Timestamp("2024-01-01 10:00", tz="UTC"), "close_htf"] == 100.0
        # Still the 09:00 candle's value at 10:05 - the 10:00 candle hasn't closed yet.
        assert aligned.loc[pd.Timestamp("2024-01-01 10:05", tz="UTC"), "close_htf"] == 100.0


def _orb_frame(breakout_high: float) -> pd.DataFrame:
    """5-minute bars covering a 15-minute opening range (09:30-09:45 NY)
    followed by bars that may or may not break out, parameterized so
    tests can extend the tail without changing the shared prefix."""
    idx = pd.date_range("2024-01-02 14:30", periods=12, freq="5min", tz="UTC")  # 14:30 UTC = 09:30 America/New_York (winter)
    data = {
        "open": [100.0] * 12,
        "high": [101.0, 102.0, 100.5] + [breakout_high] + [100.5] * 8,
        "low": [99.0] * 12,
        "close": [100.0] * 12,
    }
    return pd.DataFrame(data, index=idx)


class TestORBSignalBias:
    def test_signal_before_extension_point_unaffected_by_future_data(self) -> None:
        cfg = ORBConfig(instrument_symbol="US30", opening_range_minutes=15, execution_timeframe_minutes=5)
        strategy = ORBStrategy(cfg)

        short_df = _orb_frame(breakout_high=105.0)
        long_df = pd.concat(
            [short_df, _extend_flat(short_df, extra_bars=20, breakout_high=999.0)]
        )

        signals_short = strategy.generate_signals(MarketContext("US30", short_df))
        signals_long = strategy.generate_signals(MarketContext("US30", long_df))

        signals_short_in_common_range = [s for s in signals_short if s.signal_time <= short_df.index[-1]]
        signals_long_in_common_range = [s for s in signals_long if s.signal_time <= short_df.index[-1]]
        assert signals_short_in_common_range == signals_long_in_common_range


def _extend_flat(df: pd.DataFrame, extra_bars: int, breakout_high: float) -> pd.DataFrame:
    idx = pd.date_range(df.index[-1] + pd.Timedelta(minutes=5), periods=extra_bars, freq="5min", tz="UTC")
    data = {
        "open": [100.0] * extra_bars,
        "high": [breakout_high] + [100.5] * (extra_bars - 1),
        "low": [99.0] * extra_bars,
        "close": [100.0] * extra_bars,
    }
    return pd.DataFrame(data, index=idx)


class TestRSIDivergenceSignalBias:
    def test_signal_confirmation_time_unaffected_by_future_data(self) -> None:
        idx = pd.date_range("2024-01-01 00:00", periods=40, freq="5min", tz="UTC")
        base_prices = [100 + (i % 6) for i in range(40)]
        base_prices[10] = 130  # swing high #1
        base_prices[25] = 140  # swing high #2 (higher high in price)
        df = pd.DataFrame(
            {
                "open": base_prices,
                "high": [p + 1 for p in base_prices],
                "low": [p - 1 for p in base_prices],
                "close": base_prices,
            },
            index=idx,
        )

        cfg = RSIDivergenceConfig(instrument_symbol="US30", rsi_period=5, swing_left_bars=3, swing_right_bars=3, min_pivot_separation=5)
        strategy = RSIDivergenceStrategy(cfg)

        signals_full = strategy.generate_signals(MarketContext("US30", df))
        # Truncate 5 bars after the point where the second pivot would be confirmed.
        cutoff = 25 + 3 + 5
        truncated_df = df.iloc[:cutoff]
        signals_truncated = strategy.generate_signals(MarketContext("US30", truncated_df))

        common_cutoff_time = truncated_df.index[-1]
        full_in_common = [s for s in signals_full if s.signal_time <= common_cutoff_time]
        trunc_in_common = [s for s in signals_truncated if s.signal_time <= common_cutoff_time]
        assert full_in_common == trunc_in_common


def _ranging_frame(n: int, seed: int = 7):
    import numpy as np

    rng = np.random.default_rng(seed)
    mean = 100.0
    x = np.zeros(n)
    x[0] = mean
    for i in range(1, n):
        x[i] = x[i - 1] + 0.35 * (mean - x[i - 1]) + rng.normal(0, 0.5)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(x, index=idx)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


class TestBBStochMeanReversionSignalBias:
    def test_signal_before_truncation_point_unaffected_by_future_data(self) -> None:
        """Every indicator this strategy uses (Bollinger Bands, Stochastic,
        ADX, ATR) is a backward-looking rolling/recursive computation - the
        generic truncation test is sufficient here (no confirmation-lag
        concept like ORB's opening range or RSI divergence's pivots)."""
        full_df = _ranging_frame(400)
        cfg = BBStochMeanReversionConfig(instrument_symbol="EURUSD", adx_ranging_threshold=25)
        strategy = BBStochMeanReversionStrategy(cfg)

        cutoff = 300
        truncated_df = full_df.iloc[:cutoff]

        signals_full = strategy.generate_signals(MarketContext("EURUSD", full_df))
        signals_truncated = strategy.generate_signals(MarketContext("EURUSD", truncated_df))

        common_cutoff_time = truncated_df.index[-1]
        full_in_common = [s for s in signals_full if s.signal_time <= common_cutoff_time]
        trunc_in_common = [s for s in signals_truncated if s.signal_time <= common_cutoff_time]
        assert full_in_common == trunc_in_common


class TestLiquiditySweepFVGSignalBias:
    def test_signal_before_truncation_point_unaffected_by_future_data(self) -> None:
        """Daily bias and 1H fakeout detection only ever read already-CLOSED
        higher-timeframe candles (align_higher_timeframe's own guarantee -
        see its docstring/tests), and FVG zones only use confirmed_pos <=
        the current bar. The generic truncation test still applies: nothing
        about a signal at or before the truncation point should change when
        more data is appended afterward."""
        from app.backtesting.data_access import align_higher_timeframe
        from app.strategies.liquidity_sweep_fvg import LiquiditySweepFVGConfig, LiquiditySweepFVGStrategy

        daily_idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"], utc=True)
        daily_df = pd.DataFrame(
            {"open": [100.0, 100.0], "high": [102.0, 108.0], "low": [98.0, 99.0], "close": [99.0, 107.0]},
            index=daily_idx,
        )
        hourly_idx = pd.date_range("2024-01-03 12:00", periods=6, freq="1h", tz="UTC")
        hourly_df = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [101.0, 101.0, 101.0, 101.0, 102.0, 103.0],
                "low": [99.5, 97.0, 99.5, 99.5, 99.0, 100.0],
                "close": [100.0, 100.0, 100.0, 100.0, 101.0, 102.0],
            },
            index=hourly_idx,
        )
        exec_idx = pd.date_range("2024-01-03 12:00", periods=6 * 12, freq="5min", tz="UTC")
        close = pd.Series(100.0, index=exec_idx)
        high = close + 0.3
        low = close - 0.3
        high.iloc[25], low.iloc[25] = 100.4, 99.8
        high.iloc[26], low.iloc[26] = 101.5, 100.5
        high.iloc[27], low.iloc[27] = 102.5, 102.0
        close.iloc[27] = 102.3
        full_df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=exec_idx)

        cutoff = 40
        truncated_df = full_df.iloc[:cutoff]

        ctx_1d_full = align_higher_timeframe(exec_idx, daily_df, htf_minutes=24 * 60)
        ctx_1h_full = align_higher_timeframe(exec_idx, hourly_df, htf_minutes=60)
        ctx_1d_trunc = align_higher_timeframe(truncated_df.index, daily_df, htf_minutes=24 * 60)
        ctx_1h_trunc = align_higher_timeframe(truncated_df.index, hourly_df, htf_minutes=60)

        cfg = LiquiditySweepFVGConfig(instrument_symbol="EURUSD")
        strategy = LiquiditySweepFVGStrategy(cfg)

        signals_full = strategy.generate_signals(
            MarketContext("EURUSD", full_df, context={"1D": ctx_1d_full, "1H": ctx_1h_full})
        )
        signals_truncated = strategy.generate_signals(
            MarketContext("EURUSD", truncated_df, context={"1D": ctx_1d_trunc, "1H": ctx_1h_trunc})
        )

        common_cutoff_time = truncated_df.index[-1]
        full_in_common = [s for s in signals_full if s.signal_time <= common_cutoff_time]
        trunc_in_common = [s for s in signals_truncated if s.signal_time <= common_cutoff_time]
        assert full_in_common == trunc_in_common
