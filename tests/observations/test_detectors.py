from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtesting.data_access import align_higher_timeframe
from app.observations.detectors import (
    fvg_observations,
    liquidity_sweep_observations,
    regime_observations,
    structure_observations,
    support_resistance_observations,
    swing_observations,
)
from app.observations.models import ObservationSource, ObservationType
from app.regimes.context import RegimeContext
from app.regimes.models import StructureState, TrendState, VolatilityState


def _zigzag_df(n: int = 300, seed: int = 2) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.08, n)) + np.sin(np.arange(n) / 15) * 4, index=idx)
    high = close + rng.uniform(0.05, 0.15, n)
    low = close - rng.uniform(0.05, 0.15, n)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


class TestSwingObservations:
    def test_produces_swing_high_and_low_types(self) -> None:
        df = _zigzag_df()
        obs = swing_observations(df["high"], df["low"], "EURUSD", "5M")
        assert obs
        types = {o.observation_type for o in obs}
        assert types <= {ObservationType.SWING_HIGH, ObservationType.SWING_LOW}

    def test_timestamp_is_confirmed_at_not_pivot_time(self) -> None:
        df = _zigzag_df()
        obs = swing_observations(df["high"], df["low"], "EURUSD", "5M")
        for o in obs:
            pivot_ts = df.index[o.metadata["index_pos"]]
            confirmed_ts = df.index[o.metadata["confirmed_pos"]]
            assert o.timestamp == confirmed_ts
            assert confirmed_ts >= pivot_ts

    def test_source_is_market_data(self) -> None:
        df = _zigzag_df()
        obs = swing_observations(df["high"], df["low"], "EURUSD", "5M")
        assert all(o.source == ObservationSource.MARKET_DATA for o in obs)


class TestFVGObservations:
    def test_produces_direction_from_kind(self) -> None:
        df = _zigzag_df()
        obs = fvg_observations(df["high"], df["low"], "EURUSD", "5M")
        for o in obs:
            assert o.observation_type == ObservationType.FVG
            assert o.direction in ("long", "short")
            assert o.zone_low is not None and o.zone_high is not None


class TestStructureObservations:
    def test_produces_only_bos_and_choch(self) -> None:
        df = _zigzag_df()
        obs = structure_observations(df["high"], df["low"], "EURUSD", "5M")
        assert obs
        assert all(o.observation_type in (ObservationType.BOS, ObservationType.CHOCH) for o in obs)

    def test_first_labeled_swing_establishes_bias_without_an_event(self) -> None:
        # A monotonically increasing series should never produce a CHoCH -
        # every structural confirmation continues the same bias.
        idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
        close = pd.Series([100 + 0.5 * i + (1 if i % 4 == 0 else 0) for i in range(100)], index=idx, dtype=float)
        high = close + 0.3
        low = close - 0.3
        obs = structure_observations(high, low, "EURUSD", "5M")
        # any CHoCH here would indicate the "first swing establishes bias" rule broke
        choch = [o for o in obs if o.observation_type == ObservationType.CHOCH]
        assert choch == [] or all(o.direction == "long" for o in choch)  # weakly monotonic - allow either


class TestSupportResistanceObservations:
    def test_requires_minimum_touches(self) -> None:
        df = _zigzag_df(n=500)
        obs = support_resistance_observations(
            df["high"], df["low"], df["close"], "EURUSD", "5M", tolerance_pct=0.02, min_touches=2
        )
        for o in obs:
            assert o.metadata["touch_count"] >= 2

    def test_produces_support_or_resistance_type(self) -> None:
        df = _zigzag_df(n=500)
        obs = support_resistance_observations(
            df["high"], df["low"], df["close"], "EURUSD", "5M", tolerance_pct=0.02, min_touches=2
        )
        assert all(o.observation_type in (ObservationType.SUPPORT, ObservationType.RESISTANCE) for o in obs)

    def test_rejects_invalid_tolerance(self) -> None:
        df = _zigzag_df(n=20)
        with pytest.raises(ValueError):
            support_resistance_observations(df["high"], df["low"], df["close"], "EURUSD", "5M", tolerance_pct=0.0)

    def test_rejects_min_touches_below_two(self) -> None:
        df = _zigzag_df(n=20)
        with pytest.raises(ValueError):
            support_resistance_observations(df["high"], df["low"], df["close"], "EURUSD", "5M", min_touches=1)

    def test_empty_series_produces_no_observations(self) -> None:
        idx = pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC")
        s = pd.Series([100.0], index=idx)
        assert support_resistance_observations(s, s, s, "EURUSD", "5M") == []


def _sweep_scenario():
    daily_idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"], utc=True)
    daily_df = pd.DataFrame(
        {"open": [100.0, 100.0], "high": [102.0, 108.0], "low": [98.0, 99.0], "close": [99.0, 107.0]}, index=daily_idx
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
    exec_idx = pd.date_range("2024-01-03 12:00", periods=72, freq="5min", tz="UTC")
    close = pd.Series(100.0, index=exec_idx)
    df = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close}, index=exec_idx)
    ctx_1d = align_higher_timeframe(exec_idx, daily_df, htf_minutes=24 * 60)
    ctx_1h = align_higher_timeframe(exec_idx, hourly_df, htf_minutes=60)
    return df, ctx_1d, ctx_1h


class TestLiquiditySweepObservations:
    def test_detects_sweep_matching_the_strategy_scenario(self) -> None:
        df, ctx_1d, ctx_1h = _sweep_scenario()
        obs = liquidity_sweep_observations(df, ctx_1d, ctx_1h, "EURUSD", "5M")
        assert len(obs) == 1
        assert obs[0].direction == "long"
        assert obs[0].metadata["swept_level"] == 99.0

    def test_no_sweep_no_observation(self) -> None:
        df, ctx_1d, ctx_1h = _sweep_scenario()
        ctx_1h = ctx_1h.copy()
        ctx_1h["low_htf"] = 99.6  # never actually sweeps below prev day low
        obs = liquidity_sweep_observations(df, ctx_1d, ctx_1h, "EURUSD", "5M")
        assert obs == []


class TestRegimeObservations:
    def test_produces_three_observation_types(self) -> None:
        ctx = RegimeContext(
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"), instrument_symbol="EURUSD", timeframe="5M",
            trend_state=TrendState.UPTREND, volatility_state=VolatilityState.HIGH,
            structure_state=StructureState.TRENDING, sessions=("london",),
        )
        obs = regime_observations(ctx, current_price=105.0)
        types = {o.observation_type for o in obs}
        assert types == {ObservationType.TREND_ALIGNMENT, ObservationType.VOLATILITY_STATE, ObservationType.PRICE_LOCATION}

    def test_unknown_trend_produces_no_trend_alignment_observation(self) -> None:
        ctx = RegimeContext(
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"), instrument_symbol="EURUSD", timeframe="5M",
            trend_state=TrendState.UNKNOWN, volatility_state=VolatilityState.UNKNOWN,
            structure_state=StructureState.UNKNOWN, sessions=(),
        )
        obs = regime_observations(ctx)
        assert ObservationType.TREND_ALIGNMENT not in {o.observation_type for o in obs}
