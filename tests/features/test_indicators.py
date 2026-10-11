from __future__ import annotations

import pandas as pd
import pytest

from app.features.indicators import adx, atr, bollinger_bands, detect_fvg, detect_swing_points, rsi, stochastic_oscillator


class TestRSI:
    def test_matches_classic_wilder_example(self) -> None:
        # Standard textbook closes/expected values for period=14 Wilder RSI.
        closes = pd.Series(
            [
                44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
                46.22, 45.64,
            ]
        )
        result = rsi(closes, period=14)
        assert pd.isna(result.iloc[13])
        assert result.iloc[14] == pytest.approx(70.46, abs=0.05)
        assert result.iloc[15] == pytest.approx(66.25, abs=0.05)

    def test_bounded_zero_to_hundred(self) -> None:
        closes = pd.Series([100 + i * 0.3 for i in range(50)])
        result = rsi(closes, period=14)
        assert result.dropna().between(0, 100).all()

    def test_flat_series_avg_loss_zero_gives_100(self) -> None:
        closes = pd.Series([100.0] + [100.0 + i for i in range(1, 20)])  # strictly rising after first bar
        result = rsi(closes, period=14)
        assert result.iloc[14] == 100.0

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            rsi(pd.Series([1.0, 2.0, 3.0]), period=0)

    def test_only_depends_on_past_data(self) -> None:
        """Causality: RSI at position i must be identical whether or not
        later rows exist in the input series."""
        closes = pd.Series([100 + (i % 5) * 0.7 for i in range(40)])
        full = rsi(closes, period=14)
        truncated = rsi(closes.iloc[:25], period=14)
        pd.testing.assert_series_equal(full.iloc[:25], truncated, check_names=False)


class TestSwingPoints:
    def _index(self, n: int) -> pd.DatetimeIndex:
        return pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

    def test_detects_swing_high_with_correct_confirmation_lag(self) -> None:
        idx = self._index(20)
        high = pd.Series([1, 2, 3, 10, 4, 3, 2, 1, 2, 3, 4, 5, 20, 6, 5, 4, 3, 2, 1, 1], index=idx, dtype=float)
        low = high - 0.5

        points = detect_swing_points(high, low, left_bars=3, right_bars=3)
        highs = [p for p in points if p.kind == "high"]

        assert any(p.index_pos == 3 and p.confirmed_pos == 6 for p in highs)
        assert any(p.index_pos == 12 and p.confirmed_pos == 15 for p in highs)

    def test_confirmed_pivot_is_unaffected_by_future_data(self) -> None:
        """The core anti-lookahead guarantee: a pivot's existence and
        price must not depend on data after its confirmation bar."""
        idx = self._index(20)
        high = pd.Series([1, 2, 3, 10, 4, 3, 2, 1, 2, 3, 4, 5, 20, 6, 5, 4, 3, 2, 1, 1], index=idx, dtype=float)
        low = high - 0.5

        full = detect_swing_points(high, low, left_bars=3, right_bars=3)
        truncated = detect_swing_points(high.iloc[:7], low.iloc[:7], left_bars=3, right_bars=3)

        full_confirmed_by_7 = [(p.index_pos, p.kind, p.price) for p in full if p.confirmed_pos <= 6]
        truncated_all = [(p.index_pos, p.kind, p.price) for p in truncated]
        assert full_confirmed_by_7 == truncated_all

    def test_rejects_invalid_window(self) -> None:
        idx = self._index(5)
        s = pd.Series(range(5), index=idx, dtype=float)
        with pytest.raises(ValueError):
            detect_swing_points(s, s, left_bars=0, right_bars=3)

    def test_no_points_when_series_too_short(self) -> None:
        idx = self._index(4)
        s = pd.Series([1, 2, 3, 4], index=idx, dtype=float)
        assert detect_swing_points(s, s, left_bars=3, right_bars=3) == []


class TestBollingerBands:
    def test_middle_band_matches_simple_moving_average(self) -> None:
        closes = pd.Series([float(i) for i in range(1, 31)])
        result = bollinger_bands(closes, period=5, num_std=2.0)
        assert result["bb_middle"].iloc[9] == pytest.approx(closes.iloc[5:10].mean())

    def test_bands_are_symmetric_around_middle(self) -> None:
        closes = pd.Series([100, 102, 99, 105, 101, 98, 103, 107, 95, 110], dtype=float)
        result = bollinger_bands(closes, period=5, num_std=2.0)
        row = result.iloc[-1]
        assert (row["bb_upper"] - row["bb_middle"]) == pytest.approx(row["bb_middle"] - row["bb_lower"])

    def test_nan_before_window_is_full(self) -> None:
        closes = pd.Series([float(i) for i in range(10)])
        result = bollinger_bands(closes, period=5)
        assert result["bb_middle"].iloc[:4].isna().all()
        assert result["bb_middle"].iloc[4:].notna().all()

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            bollinger_bands(pd.Series([1.0, 2.0]), period=1)


class TestStochasticOscillator:
    def test_bounded_zero_to_hundred(self) -> None:
        idx = pd.date_range("2024-01-01", periods=60, freq="5min", tz="UTC")
        close = pd.Series([100 + 5 * ((i % 10) - 5) for i in range(60)], index=idx, dtype=float)
        high = close + 1
        low = close - 1
        result = stochastic_oscillator(high, low, close, k_period=14, smooth_k=3, d_period=3)
        assert result["stoch_k"].dropna().between(0, 100).all()
        assert result["stoch_d"].dropna().between(0, 100).all()

    def test_pins_to_100_at_the_top_of_a_strict_uptrend(self) -> None:
        idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
        close = pd.Series(range(100, 120), index=idx, dtype=float)
        result = stochastic_oscillator(close, close, close, k_period=14, smooth_k=1, d_period=1)
        assert result["stoch_k"].iloc[-1] == pytest.approx(100.0)

    def test_flat_range_is_nan_not_zero_or_hundred(self) -> None:
        idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
        close = pd.Series([100.0] * 20, index=idx)
        result = stochastic_oscillator(close, close, close, k_period=14, smooth_k=1, d_period=1)
        assert pd.isna(result["stoch_k"].iloc[-1])

    def test_rejects_invalid_periods(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            stochastic_oscillator(s, s, s, k_period=0)


class TestATR:
    def test_first_bar_true_range_is_high_minus_low(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
        high = pd.Series([10, 12, 13, 11, 14], index=idx, dtype=float)
        low = pd.Series([8, 9, 11, 9, 12], index=idx, dtype=float)
        close = pd.Series([9, 11, 12, 10, 13], index=idx, dtype=float)
        result = atr(high, low, close, period=3)
        assert result.iloc[:2].isna().all()
        assert result.iloc[2] == pytest.approx((2 + 3 + 2) / 3)  # simple mean of the first 3 TRs: [10-8, max(12-9,|12-9|,|9-9|)=3, max(13-11,|13-11|,|11-11|)=2]

    def test_non_negative(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="5min", tz="UTC")
        high = pd.Series([100 + (i % 5) for i in range(30)], index=idx, dtype=float)
        low = high - 2
        close = high - 1
        result = atr(high, low, close, period=14)
        assert (result.dropna() >= 0).all()


class TestADX:
    def test_strong_uptrend_produces_high_adx(self) -> None:
        idx = pd.date_range("2024-01-01", periods=60, freq="5min", tz="UTC")
        close = pd.Series([100 + i for i in range(60)], index=idx, dtype=float)
        high = close + 1
        low = close - 1
        result = adx(high, low, close, period=14)
        assert result["adx"].dropna().iloc[-1] > 40

    def test_flat_market_produces_low_adx(self) -> None:
        idx = pd.date_range("2024-01-01", periods=60, freq="5min", tz="UTC")
        close = pd.Series([100.0] * 60, index=idx)
        high = close + 0.1
        low = close - 0.1
        result = adx(high, low, close, period=14)
        assert result["adx"].dropna().iloc[-1] < 5

    def test_rejects_invalid_period(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            adx(s, s, s, period=0)


class TestDetectFVG:
    def _index(self, n: int) -> pd.DatetimeIndex:
        return pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

    def test_detects_bullish_gap(self) -> None:
        idx = self._index(3)
        high = pd.Series([10.0, 15.0, 20.0], index=idx)  # candle1 high=10
        low = pd.Series([8.0, 14.0, 12.0], index=idx)     # candle3 low=12 > candle1 high=10
        zones = detect_fvg(high, low)
        assert len(zones) == 1
        z = zones[0]
        assert z.kind == "bullish"
        assert z.zone_low == 10.0
        assert z.zone_high == 12.0
        assert z.index_pos == 1
        assert z.confirmed_pos == 2

    def test_detects_bearish_gap(self) -> None:
        idx = self._index(3)
        high = pd.Series([20.0, 15.0, 9.0], index=idx)   # candle3 high=9 < candle1 low=10
        low = pd.Series([10.0, 8.0, 5.0], index=idx)
        zones = detect_fvg(high, low)
        assert len(zones) == 1
        z = zones[0]
        assert z.kind == "bearish"
        assert z.zone_low == 9.0
        assert z.zone_high == 10.0

    def test_no_gap_when_ranges_overlap(self) -> None:
        idx = self._index(3)
        high = pd.Series([10.0, 15.0, 11.0], index=idx)
        low = pd.Series([8.0, 9.0, 9.5], index=idx)
        assert detect_fvg(high, low) == []

    def test_min_gap_size_filters_small_gaps(self) -> None:
        idx = self._index(3)
        high = pd.Series([10.0, 15.0, 10.2], index=idx)
        low = pd.Series([8.0, 14.0, 10.1], index=idx)  # gap of 0.1
        assert detect_fvg(high, low, min_gap_size=0.0) != []
        assert detect_fvg(high, low, min_gap_size=1.0) == []

    def test_confirmed_zone_unaffected_by_future_data(self) -> None:
        idx = self._index(6)
        high = pd.Series([10.0, 15.0, 20.0, 5.0, 6.0, 7.0], index=idx)
        low = pd.Series([8.0, 14.0, 12.0, 3.0, 4.0, 5.0], index=idx)
        full = detect_fvg(high, low)
        truncated = detect_fvg(high.iloc[:3], low.iloc[:3])
        full_confirmed_by_3 = [(z.index_pos, z.kind, z.zone_low, z.zone_high) for z in full if z.confirmed_pos <= 2]
        trunc_all = [(z.index_pos, z.kind, z.zone_low, z.zone_high) for z in truncated]
        assert full_confirmed_by_3 == trunc_all

    def test_rejects_negative_min_gap_size(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            detect_fvg(s, s, min_gap_size=-1.0)
