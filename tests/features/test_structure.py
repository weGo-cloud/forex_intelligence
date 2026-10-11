from __future__ import annotations

import pandas as pd
import pytest

from app.features.indicators import SwingPoint, detect_swing_points
from app.features.structure import label_swings, range_compression_ratio, structural_trend_strength_series


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")


def _swing(index_pos, confirmed_pos, price, kind, idx):
    return SwingPoint(
        index_pos=index_pos,
        confirmed_pos=confirmed_pos,
        timestamp=idx[index_pos],
        confirmed_at=idx[confirmed_pos],
        price=price,
        kind=kind,
    )


class TestLabelSwings:
    def test_first_swing_of_each_kind_is_unlabeled(self) -> None:
        idx = _idx(10)
        swings = [_swing(2, 5, 100.0, "high", idx), _swing(4, 7, 90.0, "low", idx)]
        labeled = label_swings(swings)
        assert all(ls.label is None for ls in labeled)

    def test_second_high_labeled_relative_to_first(self) -> None:
        idx = _idx(20)
        swings = [
            _swing(2, 5, 100.0, "high", idx),
            _swing(10, 13, 105.0, "high", idx),  # higher than 100 -> higher_high
        ]
        labeled = label_swings(swings)
        second = [ls for ls in labeled if ls.swing.index_pos == 10][0]
        assert second.label == "higher_high"

    def test_lower_high_detected(self) -> None:
        idx = _idx(20)
        swings = [_swing(2, 5, 100.0, "high", idx), _swing(10, 13, 95.0, "high", idx)]
        labeled = label_swings(swings)
        second = [ls for ls in labeled if ls.swing.index_pos == 10][0]
        assert second.label == "lower_high"

    def test_available_at_is_the_later_confirmed_pos(self) -> None:
        idx = _idx(20)
        swings = [_swing(2, 5, 100.0, "high", idx), _swing(10, 13, 105.0, "high", idx)]
        labeled = label_swings(swings)
        second = [ls for ls in labeled if ls.swing.index_pos == 10][0]
        assert second.available_at == 13


class TestStructuralTrendStrengthSeries:
    def test_all_higher_highs_and_lows_gives_positive_one(self) -> None:
        idx = _idx(30)
        swings = [
            _swing(2, 5, 100.0, "high", idx),
            _swing(4, 7, 95.0, "low", idx),
            _swing(10, 13, 105.0, "high", idx),  # higher_high
            _swing(12, 15, 98.0, "low", idx),  # higher_low
        ]
        result = structural_trend_strength_series(idx, swings, lookback_count=4)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_nan_before_first_labeled_swing(self) -> None:
        idx = _idx(30)
        swings = [_swing(2, 5, 100.0, "high", idx)]  # only one - never gets a label
        result = structural_trend_strength_series(idx, swings, lookback_count=4)
        assert result.isna().all()

    def test_value_not_visible_before_available_at(self) -> None:
        idx = _idx(30)
        swings = [
            _swing(2, 5, 100.0, "high", idx),
            _swing(10, 13, 105.0, "high", idx),
        ]
        result = structural_trend_strength_series(idx, swings, lookback_count=4)
        assert pd.isna(result.iloc[12])  # bar 12 is before confirmed_pos=13
        assert result.iloc[13] == pytest.approx(1.0)

    def test_empty_swings_all_nan(self) -> None:
        idx = _idx(10)
        result = structural_trend_strength_series(idx, [], lookback_count=4)
        assert result.isna().all()

    def test_rejects_invalid_lookback(self) -> None:
        with pytest.raises(ValueError):
            structural_trend_strength_series(_idx(5), [], lookback_count=0)


class TestRangeCompressionRatio:
    def test_ratio_near_one_for_stable_range(self) -> None:
        idx = _idx(150)
        close = pd.Series([100.0] * 150, index=idx)
        high = close + 1.0
        low = close - 1.0
        result = range_compression_ratio(high, low, close, period=20, baseline_period=100)
        assert result.dropna().apply(lambda x: x == pytest.approx(1.0)).all()

    def test_ratio_below_one_for_recent_compression(self) -> None:
        idx = _idx(150)
        high = pd.Series([101.0] * 130 + [100.2] * 20, index=idx)
        low = pd.Series([99.0] * 130 + [99.8] * 20, index=idx)
        close = pd.Series([100.0] * 150, index=idx)
        result = range_compression_ratio(high, low, close, period=20, baseline_period=100)
        assert result.iloc[-1] < 1.0

    def test_rejects_invalid_periods(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            range_compression_ratio(s, s, s, period=1, baseline_period=10)
