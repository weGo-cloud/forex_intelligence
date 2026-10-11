from __future__ import annotations

import pandas as pd
import pytest

from app.strategies.base import MarketContext
from app.strategies.rsi_divergence import RSIDivergenceConfig, RSIDivergenceStrategy


def _oscillating_frame(n: int, spike_positions: dict[int, float]) -> pd.DataFrame:
    """A flat baseline with specific bars pushed to given price levels to
    create controllable, isolated swing points. Flat (not periodic)
    baseline deliberately: a periodic zigzag baseline creates its own
    "noise" swing points at every cycle, which would contaminate the
    nearest-prior-pivot pairing the strategy uses and make tests
    unpredictable for reasons unrelated to what's being tested."""
    base = [100.0] * n
    for pos, price in spike_positions.items():
        base[pos] = price
    return pd.DataFrame(
        {"open": base, "high": [p + 0.5 for p in base], "low": [p - 0.5 for p in base], "close": base},
        index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
    )


class TestRSIDivergenceConfig:
    def test_rejects_htf_filter_without_timeframe(self) -> None:
        with pytest.raises(ValueError):
            RSIDivergenceConfig(instrument_symbol="US30", htf_filter_enabled=True, htf_timeframe=None)

    def test_rejects_short_swing_windows(self) -> None:
        with pytest.raises(ValueError):
            RSIDivergenceConfig(instrument_symbol="US30", swing_left_bars=0)


class TestBearishDivergenceDetection:
    def test_higher_high_price_lower_high_rsi_triggers_short(self) -> None:
        # Two swing highs: first modest, second a higher price high but
        # reached via smaller/weaker up-moves so RSI at the second peak
        # is lower than at the first - classic bearish divergence setup.
        n = 40
        df = _oscillating_frame(n, {10: 130.0, 25: 140.0})
        cfg = RSIDivergenceConfig(
            instrument_symbol="US30", rsi_period=5, swing_left_bars=3, swing_right_bars=3, min_pivot_separation=5
        )
        signals = RSIDivergenceStrategy(cfg).generate_signals(MarketContext("US30", df))

        shorts = [s for s in signals if s.direction == "short"]
        assert len(shorts) >= 1
        sig = shorts[0]
        assert sig.meta["pivot2_price"] > sig.meta["pivot1_price"]
        assert sig.meta["pivot2_rsi"] < sig.meta["pivot1_rsi"]
        # Signal must not be emitted before the second pivot is confirmed (index 25 + right_bars 3 = 28).
        assert sig.signal_time >= df.index[28]


class TestBullishDivergenceDetection:
    def test_lower_low_price_higher_low_rsi_triggers_long(self) -> None:
        n = 40
        df = _oscillating_frame(n, {10: 70.0, 25: 60.0})
        cfg = RSIDivergenceConfig(
            instrument_symbol="US30", rsi_period=5, swing_left_bars=3, swing_right_bars=3, min_pivot_separation=5
        )
        signals = RSIDivergenceStrategy(cfg).generate_signals(MarketContext("US30", df))

        longs = [s for s in signals if s.direction == "long"]
        assert len(longs) >= 1
        sig = longs[0]
        assert sig.meta["pivot2_price"] < sig.meta["pivot1_price"]
        assert sig.meta["pivot2_rsi"] > sig.meta["pivot1_rsi"]


class TestNoDivergence:
    def test_no_signal_on_flat_oscillation(self) -> None:
        df = _oscillating_frame(40, {})
        cfg = RSIDivergenceConfig(instrument_symbol="US30", rsi_period=5, swing_left_bars=3, swing_right_bars=3)
        signals = RSIDivergenceStrategy(cfg).generate_signals(MarketContext("US30", df))
        assert signals == []

    def test_insufficient_data_returns_no_signals(self) -> None:
        df = _oscillating_frame(5, {})
        cfg = RSIDivergenceConfig(instrument_symbol="US30", rsi_period=14)
        signals = RSIDivergenceStrategy(cfg).generate_signals(MarketContext("US30", df))
        assert signals == []


class TestMaxTradesPerDay:
    def test_daily_cap_is_enforced(self) -> None:
        n = 60
        df = _oscillating_frame(n, {5: 130.0, 12: 140.0, 20: 150.0, 28: 160.0})
        cfg = RSIDivergenceConfig(
            instrument_symbol="US30",
            rsi_period=5,
            swing_left_bars=2,
            swing_right_bars=2,
            min_pivot_separation=3,
            max_trades_per_day=1,
        )
        signals = RSIDivergenceStrategy(cfg).generate_signals(MarketContext("US30", df))
        by_day: dict = {}
        for s in signals:
            by_day.setdefault(s.signal_time.date(), 0)
            by_day[s.signal_time.date()] += 1
        assert all(count <= 1 for count in by_day.values())
