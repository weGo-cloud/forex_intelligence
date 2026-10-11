from __future__ import annotations

import math

import pytest

from app.regimes.config import RegimeClassifierConfig
from app.regimes.detectors import classify_structure, classify_trend, classify_volatility
from app.regimes.models import StructureState, TrendState, VolatilityState

CFG = RegimeClassifierConfig()
NAN = float("nan")


class TestClassifyTrend:
    def test_uptrend_when_all_signals_agree(self) -> None:
        result = classify_trend(close=110, slow_ma=100, slow_ma_slope=0.01, plus_di=30, minus_di=10, adx=25, config=CFG)
        assert result == TrendState.UPTREND

    def test_downtrend_when_all_signals_agree(self) -> None:
        result = classify_trend(close=90, slow_ma=100, slow_ma_slope=-0.01, plus_di=10, minus_di=30, adx=25, config=CFG)
        assert result == TrendState.DOWNTREND

    def test_neutral_when_adx_below_threshold(self) -> None:
        result = classify_trend(close=110, slow_ma=100, slow_ma_slope=0.01, plus_di=30, minus_di=10, adx=5, config=CFG)
        assert result == TrendState.NEUTRAL

    def test_neutral_when_votes_split(self) -> None:
        # price above MA (bullish, +1) but flat slope (0) and -DI > +DI (bearish, -1):
        # 1 vote each way - neither reaches the 2-vote majority required.
        result = classify_trend(close=110, slow_ma=100, slow_ma_slope=0.0, plus_di=10, minus_di=30, adx=25, config=CFG)
        assert result == TrendState.NEUTRAL

    def test_unknown_when_any_input_is_nan(self) -> None:
        result = classify_trend(close=110, slow_ma=100, slow_ma_slope=NAN, plus_di=30, minus_di=10, adx=25, config=CFG)
        assert result == TrendState.UNKNOWN

    def test_unknown_when_adx_is_nan(self) -> None:
        result = classify_trend(close=110, slow_ma=100, slow_ma_slope=0.01, plus_di=30, minus_di=10, adx=NAN, config=CFG)
        assert result == TrendState.UNKNOWN

    def test_deterministic(self) -> None:
        args = dict(close=110, slow_ma=100, slow_ma_slope=0.01, plus_di=30, minus_di=10, adx=25, config=CFG)
        assert classify_trend(**args) == classify_trend(**args)


class TestClassifyVolatility:
    def test_low_below_threshold(self) -> None:
        assert classify_volatility(10.0, CFG) == VolatilityState.LOW

    def test_high_above_threshold(self) -> None:
        assert classify_volatility(90.0, CFG) == VolatilityState.HIGH

    def test_normal_in_between(self) -> None:
        assert classify_volatility(50.0, CFG) == VolatilityState.NORMAL

    def test_boundary_values_are_inclusive(self) -> None:
        assert classify_volatility(CFG.volatility_low_percentile, CFG) == VolatilityState.LOW
        assert classify_volatility(CFG.volatility_high_percentile, CFG) == VolatilityState.HIGH

    def test_unknown_when_nan(self) -> None:
        assert classify_volatility(NAN, CFG) == VolatilityState.UNKNOWN


class TestClassifyStructure:
    def test_trending_when_strong_structure_and_expansion(self) -> None:
        result = classify_structure(structural_trend_strength=0.8, range_compression_ratio=1.2, config=CFG)
        assert result == StructureState.TRENDING

    def test_ranging_when_weak_structure_and_compression(self) -> None:
        result = classify_structure(structural_trend_strength=0.1, range_compression_ratio=0.5, config=CFG)
        assert result == StructureState.RANGING

    def test_transition_when_signals_disagree(self) -> None:
        # strong swing structure but compressed range - mixed evidence
        result = classify_structure(structural_trend_strength=0.9, range_compression_ratio=0.5, config=CFG)
        assert result == StructureState.TRANSITION

    def test_structure_independent_of_trend_direction(self) -> None:
        up = classify_structure(structural_trend_strength=0.9, range_compression_ratio=1.5, config=CFG)
        down = classify_structure(structural_trend_strength=-0.9, range_compression_ratio=1.5, config=CFG)
        assert up == down == StructureState.TRENDING

    def test_unknown_when_nan(self) -> None:
        assert classify_structure(NAN, 1.0, CFG) == StructureState.UNKNOWN
        assert classify_structure(0.5, NAN, CFG) == StructureState.UNKNOWN
