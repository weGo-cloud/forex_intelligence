from __future__ import annotations

import pytest

from app.research.models import ExperimentDefinition, ExperimentType
from tests.research.conftest import utc


def _base_kwargs(**overrides):
    kwargs = dict(
        strategy_name="orb",
        strategy_params={"opening_range_minutes": 15, "risk_reward": 2.0},
        instrument_symbol="EURUSD",
        timeframe="5M",
        test_period_start=utc(2024, 1, 1),
        test_period_end=utc(2024, 1, 10),
    )
    kwargs.update(overrides)
    return kwargs


class TestDeterministicSerialization:
    def test_canonical_json_is_stable_across_calls(self) -> None:
        definition = ExperimentDefinition(**_base_kwargs())
        assert definition.canonical_json() == definition.canonical_json()

    def test_dict_insertion_order_does_not_affect_hash(self) -> None:
        d1 = ExperimentDefinition(**_base_kwargs(strategy_params={"a": 1, "b": 2}))
        d2 = ExperimentDefinition(**_base_kwargs(strategy_params={"b": 2, "a": 1}))
        assert d1.config_hash() == d2.config_hash()

    def test_equivalent_configurations_hash_identically(self) -> None:
        d1 = ExperimentDefinition(**_base_kwargs())
        d2 = ExperimentDefinition(**_base_kwargs())
        assert d1.config_hash() == d2.config_hash()

    def test_name_tags_and_created_at_do_not_affect_hash(self) -> None:
        d1 = ExperimentDefinition(**_base_kwargs())
        d2 = ExperimentDefinition(
            **_base_kwargs(experiment_name="my run", tags={"team": "quant"}, created_at=utc(2099, 1, 1))
        )
        assert d1.config_hash() == d2.config_hash()


class TestHashSensitivity:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"strategy_name": "rsi_divergence"},
            {"strategy_params": {"opening_range_minutes": 15, "risk_reward": 3.0}},
            {"instrument_symbol": "GBPUSD"},
            {"timeframe": "1H"},
            {"test_period_start": utc(2024, 1, 2)},
            {"test_period_end": utc(2024, 1, 11)},
            {"initial_equity": 20_000.0},
            {"risk_per_trade_pct": 2.0},
            {"spread_points": 1.0},
            {"commission_per_trade": 5.0},
            {"same_bar_sl_tp_policy": "take_profit_first"},
            {"max_holding_bars": 50},
            {"allow_short": False},
            {"data_source": "yfinance"},
            {"execution_engine": "vectorbt"},
            {"experiment_type": ExperimentType.WALK_FORWARD},
            {"context_timeframes": {"1H": "1H"}},
        ],
    )
    def test_materially_different_configuration_changes_hash(self, overrides) -> None:
        base = ExperimentDefinition(**_base_kwargs())
        changed = ExperimentDefinition(**_base_kwargs(**overrides))
        assert base.config_hash() != changed.config_hash()


class TestValidation:
    def test_naive_datetime_is_rejected(self) -> None:
        import datetime as dt

        with pytest.raises(ValueError):
            ExperimentDefinition(**_base_kwargs(test_period_start=dt.datetime(2024, 1, 1)))

    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ExperimentDefinition(**_base_kwargs(test_period_start=utc(2024, 1, 10), test_period_end=utc(2024, 1, 1)))

    def test_with_overrides_does_not_mutate_original(self) -> None:
        original = ExperimentDefinition(**_base_kwargs())
        derived = original.with_overrides(strategy_params={"risk_reward": 99.0})
        assert original.strategy_params["risk_reward"] == 2.0
        assert derived.strategy_params["risk_reward"] == 99.0
        assert original.config_hash() != derived.config_hash()
