from __future__ import annotations

import pandas as pd
import pytest

from app.features.volatility import (
    normalized_atr,
    rolling_percentile_rank,
    rolling_return_volatility,
    volatility_ratio,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")


class TestNormalizedATR:
    def test_matches_manual_ratio(self) -> None:
        idx = _idx(3)
        atr = pd.Series([1.0, 2.0, 3.0], index=idx)
        close = pd.Series([100.0, 100.0, 100.0], index=idx)
        result = normalized_atr(atr, close)
        assert result.iloc[1] == pytest.approx(0.02)

    def test_zero_close_is_nan_not_inf(self) -> None:
        idx = _idx(1)
        atr = pd.Series([1.0], index=idx)
        close = pd.Series([0.0], index=idx)
        result = normalized_atr(atr, close)
        assert pd.isna(result.iloc[0])


class TestRollingReturnVolatility:
    def test_zero_for_constant_series(self) -> None:
        close = pd.Series([100.0] * 20, index=_idx(20))
        result = rolling_return_volatility(close, period=10)
        assert result.dropna().eq(0.0).all()

    def test_nonzero_for_varying_series(self) -> None:
        idx = _idx(20)
        close = pd.Series([100 + (i % 3) for i in range(20)], index=idx, dtype=float)
        result = rolling_return_volatility(close, period=10)
        assert (result.dropna() > 0).any()

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            rolling_return_volatility(pd.Series([1.0, 2.0]), period=1)


class TestVolatilityRatio:
    def test_ratio_of_one_for_constant_normalized_atr(self) -> None:
        idx = _idx(30)
        natr = pd.Series([0.01] * 30, index=idx)
        result = volatility_ratio(natr, baseline_period=10)
        assert result.dropna().apply(lambda x: x == pytest.approx(1.0)).all()

    def test_above_one_when_recent_spike(self) -> None:
        idx = _idx(30)
        values = [0.01] * 29 + [0.05]
        natr = pd.Series(values, index=idx)
        result = volatility_ratio(natr, baseline_period=10)
        assert result.iloc[-1] > 1.0

    def test_rejects_invalid_baseline(self) -> None:
        with pytest.raises(ValueError):
            volatility_ratio(pd.Series([1.0, 2.0]), baseline_period=1)


class TestRollingPercentileRank:
    def test_max_value_ranks_100(self) -> None:
        idx = _idx(10)
        s = pd.Series(range(10), index=idx, dtype=float)
        result = rolling_percentile_rank(s, window=10)
        assert result.iloc[-1] == pytest.approx(100.0)

    def test_min_value_ranks_low(self) -> None:
        idx = _idx(10)
        s = pd.Series(list(range(9, -1, -1)), index=idx, dtype=float)  # descending: last value is the min
        result = rolling_percentile_rank(s, window=10)
        assert result.iloc[-1] == pytest.approx(10.0)  # itself is the only value <= itself, out of 10

    def test_nan_before_full_window(self) -> None:
        idx = _idx(5)
        s = pd.Series(range(5), index=idx, dtype=float)
        result = rolling_percentile_rank(s, window=10)
        assert result.isna().all()

    def test_bounded_zero_to_hundred(self) -> None:
        idx = _idx(50)
        import numpy as np

        s = pd.Series(np.random.default_rng(1).normal(0, 1, 50), index=idx)
        result = rolling_percentile_rank(s, window=20)
        assert result.dropna().between(0, 100).all()

    def test_rejects_invalid_window(self) -> None:
        with pytest.raises(ValueError):
            rolling_percentile_rank(pd.Series([1.0, 2.0]), window=1)
