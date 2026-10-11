from __future__ import annotations

import pandas as pd
import pytest

from app.strategies.base import MarketContext
from app.strategies.orb import ORBConfig, ORBStrategy


def _session_frame(highs, lows, closes=None):
    idx = pd.date_range("2024-01-02 14:30", periods=len(highs), freq="5min", tz="UTC")  # 09:30 America/New_York
    closes = closes or highs
    return pd.DataFrame({"open": [h - 0.5 for h in highs], "high": highs, "low": lows, "close": closes}, index=idx)


class TestORBConfig:
    def test_rejects_non_multiple_opening_range(self) -> None:
        with pytest.raises(ValueError):
            ORBConfig(instrument_symbol="US30", opening_range_minutes=17, execution_timeframe_minutes=5)

    def test_rejects_trading_window_not_longer_than_range(self) -> None:
        with pytest.raises(ValueError):
            ORBConfig(instrument_symbol="US30", opening_range_minutes=15, trading_window_minutes=15)


class TestORBSignals:
    def test_long_breakout_detected_after_range_closes(self) -> None:
        # 3 bars = 15min opening range (100.5-99), then a bar breaking above range high.
        highs = [101.0, 100.0, 100.0, 105.0, 100.0]
        lows = [99.0, 99.0, 99.0, 99.0, 99.0]
        df = _session_frame(highs, lows)
        cfg = ORBConfig(instrument_symbol="US30", opening_range_minutes=15, execution_timeframe_minutes=5)
        signals = ORBStrategy(cfg).generate_signals(MarketContext("US30", df))

        assert len(signals) == 1
        assert signals[0].direction == "long"
        assert signals[0].stop_loss == 99.0  # opposite side of the opening range
        assert signals[0].signal_time == df.index[3]

    def test_short_breakout_detected(self) -> None:
        highs = [101.0, 100.0, 100.0, 100.0, 100.0]
        lows = [99.0, 99.0, 99.0, 90.0, 99.0]
        df = _session_frame(highs, lows, closes=[100.5, 99.5, 99.5, 90.5, 99.5])
        cfg = ORBConfig(instrument_symbol="US30", opening_range_minutes=15, execution_timeframe_minutes=5)
        signals = ORBStrategy(cfg).generate_signals(MarketContext("US30", df))

        assert len(signals) == 1
        assert signals[0].direction == "short"
        assert signals[0].stop_loss == 101.0

    def test_max_trades_per_session_limits_signal_count(self) -> None:
        # Two separate breakout-worthy bars after the opening range.
        highs = [101.0, 100.0, 100.0, 105.0, 100.0, 106.0]
        lows = [99.0] * 6
        df = _session_frame(highs, lows)
        cfg = ORBConfig(
            instrument_symbol="US30", opening_range_minutes=15, execution_timeframe_minutes=5, max_trades_per_session=1
        )
        signals = ORBStrategy(cfg).generate_signals(MarketContext("US30", df))
        assert len(signals) == 1

    def test_no_signal_when_no_breakout(self) -> None:
        highs = [100.5] * 6
        lows = [99.5] * 6
        df = _session_frame(highs, lows)
        cfg = ORBConfig(instrument_symbol="US30", opening_range_minutes=15, execution_timeframe_minutes=5)
        signals = ORBStrategy(cfg).generate_signals(MarketContext("US30", df))
        assert signals == []

    def test_wick_breakout_definition_is_configurable(self) -> None:
        # Close stays inside the range, but the wick pokes above it.
        highs = [101.0, 100.0, 100.0, 105.0, 100.0]
        lows = [99.0] * 5
        df = _session_frame(highs, lows, closes=[100.5, 99.5, 99.5, 99.5, 99.5])

        cfg_close = ORBConfig(instrument_symbol="US30", opening_range_minutes=15, execution_timeframe_minutes=5, breakout_definition="close")
        assert ORBStrategy(cfg_close).generate_signals(MarketContext("US30", df)) == []

        cfg_wick = ORBConfig(instrument_symbol="US30", opening_range_minutes=15, execution_timeframe_minutes=5, breakout_definition="wick")
        signals = ORBStrategy(cfg_wick).generate_signals(MarketContext("US30", df))
        assert len(signals) == 1

    def test_params_reflects_exact_config(self) -> None:
        cfg = ORBConfig(instrument_symbol="US30", risk_reward=3.0)
        params = ORBStrategy(cfg).params()
        assert params["risk_reward"] == 3.0
        assert params["instrument_symbol"] == "US30"
