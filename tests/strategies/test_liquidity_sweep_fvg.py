from __future__ import annotations

import pandas as pd
import pytest

from app.backtesting.data_access import align_higher_timeframe
from app.strategies.base import MarketContext
from app.strategies.liquidity_sweep_fvg import LiquiditySweepFVGConfig, LiquiditySweepFVGStrategy


def _build_market(
    daily_open, daily_close, daily_high=None, daily_low=None,
    hourly_low_sweep=97.0, hourly_close_reject=100.0, add_fvg_and_pullback=True,
) -> MarketContext:
    """Builds a minimal but complete multi-timeframe scenario: 2 daily
    candles (only the 2nd/most recent is "yesterday" as of exec day),
    6 hourly candles covering the NY session, and a 5M execution frame
    with an optional constructed bullish FVG + pullback right after the
    fakeout hour closes."""
    daily_high = daily_high if daily_high is not None else max(daily_open, daily_close) + 5
    daily_low = daily_low if daily_low is not None else min(daily_open, daily_close) - 5

    daily_idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"], utc=True)
    daily_df = pd.DataFrame(
        {"open": [100.0, daily_open], "high": [102.0, daily_high], "low": [98.0, daily_low], "close": [99.0, daily_close]},
        index=daily_idx,
    )

    hourly_idx = pd.date_range("2024-01-03 12:00", periods=6, freq="1h", tz="UTC")
    hourly_df = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [101.0, 101.0, 101.0, 101.0, 102.0, 103.0],
            "low": [99.5, hourly_low_sweep, 99.5, 99.5, 99.0, 100.0],
            "close": [100.0, hourly_close_reject, 100.0, 100.0, 101.0, 102.0],
        },
        index=hourly_idx,
    )

    exec_idx = pd.date_range("2024-01-03 12:00", periods=6 * 12, freq="5min", tz="UTC")
    close = pd.Series(100.0, index=exec_idx)
    high = close + 0.3
    low = close - 0.3
    if add_fvg_and_pullback:
        high.iloc[25], low.iloc[25] = 100.4, 99.8
        high.iloc[26], low.iloc[26] = 101.5, 100.5
        high.iloc[27], low.iloc[27] = 102.5, 102.0
        close.iloc[27] = 102.3
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=exec_idx)

    ctx_1d = align_higher_timeframe(exec_idx, daily_df, htf_minutes=24 * 60)
    ctx_1h = align_higher_timeframe(exec_idx, hourly_df, htf_minutes=60)
    return MarketContext("EURUSD", df, context={"1D": ctx_1d, "1H": ctx_1h})


class TestConfigValidation:
    def test_rejects_session_end_before_start(self) -> None:
        from datetime import time

        with pytest.raises(ValueError):
            LiquiditySweepFVGConfig(instrument_symbol="EURUSD", ny_session_start=time(11, 0), ny_session_end=time(8, 0))

    def test_rejects_too_small_lookback(self) -> None:
        with pytest.raises(ValueError):
            LiquiditySweepFVGConfig(instrument_symbol="EURUSD", fvg_lookback_bars=2)

    def test_rejects_non_positive_max_trades(self) -> None:
        with pytest.raises(ValueError):
            LiquiditySweepFVGConfig(instrument_symbol="EURUSD", max_trades_per_day=0)


class TestRequiresContext:
    def test_raises_without_required_context_frames(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0}, index=idx)
        strategy = LiquiditySweepFVGStrategy(LiquiditySweepFVGConfig(instrument_symbol="EURUSD"))
        with pytest.raises(ValueError):
            strategy.generate_signals(MarketContext("EURUSD", df, context={}))


class TestSignalGeneration:
    def test_bullish_bias_sweep_and_fvg_pullback_produces_long_signal(self) -> None:
        market = _build_market(daily_open=100.0, daily_close=107.0, daily_high=108.0, daily_low=99.0)
        strategy = LiquiditySweepFVGStrategy(LiquiditySweepFVGConfig(instrument_symbol="EURUSD"))
        signals = strategy.generate_signals(market)

        assert len(signals) == 1
        assert signals[0].direction == "long"
        assert signals[0].stop_loss == 97.0  # the swept 1H low
        assert signals[0].take_profit == 108.0  # yesterday's high

    def test_bearish_bias_sweep_produces_short_signal(self) -> None:
        # Yesterday bearish (close < open); sweep the daily HIGH instead, reject back below.
        market = _build_market(
            daily_open=107.0, daily_close=100.0, daily_high=108.0, daily_low=99.0,
            hourly_low_sweep=99.5, hourly_close_reject=100.0, add_fvg_and_pullback=False,
        )
        # Manually build a bearish sweep: hourly high sweeps above prev_day_high(108), closes back below.
        hourly_idx = pd.date_range("2024-01-03 12:00", periods=6, freq="1h", tz="UTC")
        hourly_df = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [101.0, 109.0, 101.0, 101.0, 102.0, 103.0],
                "low": [99.5, 100.0, 99.5, 99.5, 99.0, 100.0],
                "close": [100.0, 100.0, 100.0, 100.0, 101.0, 102.0],
            },
            index=hourly_idx,
        )
        exec_idx = pd.date_range("2024-01-03 12:00", periods=6 * 12, freq="5min", tz="UTC")
        close = pd.Series(100.0, index=exec_idx)
        high = close + 0.3
        low = close - 0.3
        # construct a bearish FVG + pullback after the fakeout hour (bar 24) closes
        high.iloc[25], low.iloc[25] = 100.2, 99.6
        high.iloc[26], low.iloc[26] = 99.5, 98.5
        high.iloc[27], low.iloc[27] = 98.0, 97.5
        close.iloc[27] = 97.7
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=exec_idx)

        daily_idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"], utc=True)
        daily_df = pd.DataFrame(
            {"open": [100.0, 107.0], "high": [102.0, 108.0], "low": [98.0, 99.0], "close": [99.0, 100.0]},
            index=daily_idx,
        )
        ctx_1d = align_higher_timeframe(exec_idx, daily_df, htf_minutes=24 * 60)
        ctx_1h = align_higher_timeframe(exec_idx, hourly_df, htf_minutes=60)
        market = MarketContext("EURUSD", df, context={"1D": ctx_1d, "1H": ctx_1h})

        strategy = LiquiditySweepFVGStrategy(LiquiditySweepFVGConfig(instrument_symbol="EURUSD"))
        signals = strategy.generate_signals(market)
        assert len(signals) == 1
        assert signals[0].direction == "short"
        assert signals[0].stop_loss == 109.0  # the swept 1H high
        assert signals[0].take_profit == 99.0  # yesterday's low

    def test_flat_daily_candle_produces_no_bias_and_no_signals(self) -> None:
        market = _build_market(daily_open=100.0, daily_close=100.0)  # flat day: no bias
        strategy = LiquiditySweepFVGStrategy(LiquiditySweepFVGConfig(instrument_symbol="EURUSD"))
        assert strategy.generate_signals(market) == []

    def test_no_sweep_no_signal(self) -> None:
        # bullish bias but the "sweep" hourly candle never actually goes below prev day low
        market = _build_market(
            daily_open=100.0, daily_close=107.0, daily_high=108.0, daily_low=99.0,
            hourly_low_sweep=99.6, hourly_close_reject=100.0,
        )
        strategy = LiquiditySweepFVGStrategy(LiquiditySweepFVGConfig(instrument_symbol="EURUSD"))
        assert strategy.generate_signals(market) == []

    def test_sweep_without_matching_fvg_produces_no_signal(self) -> None:
        market = _build_market(daily_open=100.0, daily_close=107.0, daily_high=108.0, daily_low=99.0, add_fvg_and_pullback=False)
        strategy = LiquiditySweepFVGStrategy(LiquiditySweepFVGConfig(instrument_symbol="EURUSD"))
        assert strategy.generate_signals(market) == []

    def test_max_trades_per_day_limits_signal_count(self) -> None:
        market = _build_market(daily_open=100.0, daily_close=107.0, daily_high=108.0, daily_low=99.0)
        cfg = LiquiditySweepFVGConfig(instrument_symbol="EURUSD", max_trades_per_day=1)
        signals = LiquiditySweepFVGStrategy(cfg).generate_signals(market)
        assert len(signals) == 1

    def test_params_reports_every_config_field(self) -> None:
        cfg = LiquiditySweepFVGConfig(instrument_symbol="EURUSD")
        strategy = LiquiditySweepFVGStrategy(cfg)
        params = strategy.params()
        for field_name in cfg.__dataclass_fields__:
            assert field_name in params
