from __future__ import annotations

import pandas as pd
import pytest

from app.features.trend import directional_return, ma_alignment, ma_slope, moving_average


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")


class TestMovingAverage:
    def test_matches_manual_mean(self) -> None:
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=_idx(5))
        result = moving_average(close, period=3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_nan_before_warmup(self) -> None:
        close = pd.Series([1.0, 2.0, 3.0], index=_idx(3))
        result = moving_average(close, period=3)
        assert result.iloc[:2].isna().all()

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            moving_average(pd.Series([1.0]), period=0)


class TestMAAlignment:
    def test_positive_when_fast_above_slow(self) -> None:
        idx = _idx(3)
        fast = pd.Series([10.0, 11.0, 12.0], index=idx)
        slow = pd.Series([9.0, 9.0, 9.0], index=idx)
        result = ma_alignment(fast, slow)
        assert (result == 1.0).all()

    def test_negative_when_fast_below_slow(self) -> None:
        idx = _idx(2)
        fast = pd.Series([5.0, 5.0], index=idx)
        slow = pd.Series([9.0, 9.0], index=idx)
        assert (ma_alignment(fast, slow) == -1.0).all()

    def test_zero_when_equal(self) -> None:
        idx = _idx(2)
        s = pd.Series([5.0, 5.0], index=idx)
        assert (ma_alignment(s, s) == 0.0).all()

    def test_nan_propagates(self) -> None:
        idx = _idx(2)
        fast = pd.Series([float("nan"), 5.0], index=idx)
        slow = pd.Series([5.0, 5.0], index=idx)
        result = ma_alignment(fast, slow)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 0.0


class TestMASlope:
    def test_matches_manual_fractional_change(self) -> None:
        ma = pd.Series([100.0, 100.0, 100.0, 110.0, 121.0], index=_idx(5))
        result = ma_slope(ma, lookback=2)
        assert result.iloc[4] == pytest.approx((121.0 - 100.0) / 100.0)

    def test_nan_before_warmup(self) -> None:
        ma = pd.Series([100.0, 101.0, 102.0], index=_idx(3))
        result = ma_slope(ma, lookback=2)
        assert result.iloc[:2].isna().all()

    def test_zero_prior_value_is_nan_not_inf(self) -> None:
        ma = pd.Series([0.0, 5.0, 10.0], index=_idx(3))
        result = ma_slope(ma, lookback=2)
        assert pd.isna(result.iloc[2]) or not (result.iloc[2] == float("inf"))

    def test_rejects_invalid_lookback(self) -> None:
        with pytest.raises(ValueError):
            ma_slope(pd.Series([1.0, 2.0]), lookback=0)


class TestDirectionalReturn:
    def test_matches_manual_percent_change(self) -> None:
        close = pd.Series([100.0, 105.0, 110.0, 121.0], index=_idx(4))
        result = directional_return(close, period=3)
        assert result.iloc[3] == pytest.approx((121.0 - 100.0) / 100.0)

    def test_nan_before_warmup(self) -> None:
        close = pd.Series([100.0, 105.0], index=_idx(2))
        result = directional_return(close, period=3)
        assert result.isna().all()

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            directional_return(pd.Series([1.0]), period=0)
