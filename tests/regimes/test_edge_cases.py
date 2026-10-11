from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.regimes.classifier import RegimeClassifier
from app.regimes.models import StructureState, TrendState, VolatilityState


def _idx(n: int, tz="UTC") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="5min", tz=tz)


class TestEmptyAndInsufficientData:
    def test_empty_dataframe_raises_not_crashes_silently(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        with pytest.raises(ValueError):
            RegimeClassifier().classify_series(df, "EURUSD", "5M")

    def test_single_bar_produces_unknown_not_an_exception(self) -> None:
        idx = _idx(1)
        df = pd.DataFrame({"open": [1.0], "high": [1.01], "low": [0.99], "close": [1.0]}, index=idx)
        contexts = RegimeClassifier().classify_series(df, "EURUSD", "5M")
        assert len(contexts) == 1
        assert contexts[0].trend_state == TrendState.UNKNOWN


class TestMissingColumns:
    def test_missing_high_raises(self) -> None:
        idx = _idx(10)
        df = pd.DataFrame({"open": 1.0, "low": 0.99, "close": 1.0}, index=idx)
        with pytest.raises(ValueError, match="high"):
            RegimeClassifier().classify_series(df, "EURUSD", "5M")

    def test_missing_close_raises(self) -> None:
        idx = _idx(10)
        df = pd.DataFrame({"open": 1.0, "high": 1.01, "low": 0.99}, index=idx)
        with pytest.raises(ValueError, match="close"):
            RegimeClassifier().classify_series(df, "EURUSD", "5M")


class TestConstantPrices:
    def test_flat_market_does_not_crash_and_is_not_falsely_trending(self) -> None:
        idx = _idx(300)
        df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
        contexts = RegimeClassifier().classify_series(df, "EURUSD", "5M")
        # Zero true range everywhere means ADX/ATR are degenerate (0 or NaN) -
        # never a false UPTREND/DOWNTREND call from a perfectly flat market.
        assert all(c.trend_state in (TrendState.NEUTRAL, TrendState.UNKNOWN) for c in contexts)


class TestVolatilitySpike:
    def test_sudden_spike_does_not_crash(self) -> None:
        idx = _idx(200)
        close = pd.Series([100.0] * 150 + [100.0 + i * 5 for i in range(50)], index=idx)
        high = close + 0.2
        low = close - 0.2
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
        contexts = RegimeClassifier().classify_series(df, "EURUSD", "5M")
        assert len(contexts) == 200
        assert contexts[-1].volatility_state in (VolatilityState.LOW, VolatilityState.NORMAL, VolatilityState.HIGH)


class TestNoConfirmedSwings:
    def test_monotonic_series_with_no_swings_still_produces_context(self) -> None:
        idx = _idx(200)
        close = pd.Series([100.0 + 0.01 * i for i in range(200)], index=idx)  # strictly increasing - no pivots at all
        high = close + 0.05
        low = close - 0.05
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
        contexts = RegimeClassifier().classify_series(df, "EURUSD", "5M")
        # No swings ever confirm -> structural_trend_strength stays NaN forever -> UNKNOWN, not invented.
        assert all(c.structure_state == StructureState.UNKNOWN for c in contexts)


class TestDuplicateAndNonMonotonicTimestamps:
    def test_duplicate_timestamps_do_not_crash(self) -> None:
        idx = _idx(20).tolist()
        idx[10] = idx[9]  # duplicate
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.01, "low": 0.99, "close": np.linspace(1, 1.2, 20)},
            index=pd.DatetimeIndex(idx),
        )
        # Should not raise inside the regime engine itself (data validation
        # is the data layer's job per the mission - this only checks the
        # regime engine doesn't crash ungracefully on it).
        RegimeClassifier().classify_series(df, "EURUSD", "5M")

    def test_non_monotonic_timestamps_do_not_crash(self) -> None:
        idx = _idx(20).tolist()
        idx[5], idx[6] = idx[6], idx[5]  # swap two timestamps out of order
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.01, "low": 0.99, "close": np.linspace(1, 1.2, 20)},
            index=pd.DatetimeIndex(idx),
        )
        RegimeClassifier().classify_series(df, "EURUSD", "5M")


class TestTimezoneHandling:
    def test_timezone_aware_index_works(self) -> None:
        df = pd.DataFrame(
            {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}, index=_idx(200, tz="UTC")
        )
        contexts = RegimeClassifier().classify_series(df, "EURUSD", "5M")
        assert len(contexts) == 200

    def test_timezone_naive_index_raises_in_session_labeling(self) -> None:
        """Per this project's established rule (naive timestamps are
        rejected, not silently assumed UTC - see M1's UTCDateTime),
        session labeling - the one place regimes code calls
        tz_convert - must fail loudly on a naive index rather than
        silently mislabeling sessions."""
        naive_idx = pd.date_range("2024-01-01", periods=200, freq="5min")  # no tz
        df = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}, index=naive_idx)
        with pytest.raises(Exception):
            RegimeClassifier().classify_series(df, "EURUSD", "5M")
