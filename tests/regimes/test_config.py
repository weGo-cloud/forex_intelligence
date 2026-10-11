from __future__ import annotations

import pytest

from app.regimes.config import RegimeClassifierConfig


class TestValidation:
    def test_default_config_is_valid(self) -> None:
        RegimeClassifierConfig()  # should not raise

    def test_rejects_fast_ma_not_less_than_slow(self) -> None:
        with pytest.raises(ValueError):
            RegimeClassifierConfig(trend_fast_ma_period=50, trend_slow_ma_period=50)

    def test_rejects_invalid_min_agreeing_votes(self) -> None:
        with pytest.raises(ValueError):
            RegimeClassifierConfig(trend_min_agreeing_votes=4)

    def test_rejects_invalid_volatility_percentiles(self) -> None:
        with pytest.raises(ValueError):
            RegimeClassifierConfig(volatility_low_percentile=80, volatility_high_percentile=20)

    def test_rejects_zero_swing_bars(self) -> None:
        with pytest.raises(ValueError):
            RegimeClassifierConfig(structure_swing_left_bars=0)

    def test_rejects_out_of_range_structure_threshold(self) -> None:
        with pytest.raises(ValueError):
            RegimeClassifierConfig(structure_trend_strength_threshold=1.5)

    def test_multiple_configs_are_independent(self) -> None:
        a = RegimeClassifierConfig(trend_adx_threshold=15.0)
        b = RegimeClassifierConfig(trend_adx_threshold=30.0)
        assert a.trend_adx_threshold == 15.0
        assert b.trend_adx_threshold == 30.0
