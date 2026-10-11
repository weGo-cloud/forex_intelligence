from __future__ import annotations

from sqlalchemy.orm import Session

from app.research.models import ExperimentDefinition, ExperimentStatus, ExperimentType
from app.research.registry import ExperimentRegistry
from app.research.sweep import _combinations, run_parameter_sweep


def _orb_definition(start, end) -> ExperimentDefinition:
    return ExperimentDefinition(
        strategy_name="orb",
        strategy_params={
            "instrument_symbol": "EURUSD",
            "opening_range_minutes": 15,
            "execution_timeframe_minutes": 5,
            "risk_reward": 2.0,
        },
        instrument_symbol="EURUSD",
        timeframe="5M",
        test_period_start=start,
        test_period_end=end,
    )


class TestCombinations:
    def test_cartesian_product_correctness(self) -> None:
        combos = _combinations({"a": [1, 2], "b": [10, 20]})
        assert len(combos) == 4
        assert {"a": 1, "b": 10} in combos
        assert {"a": 2, "b": 20} in combos

    def test_deterministic_ordering(self) -> None:
        grid = {"risk_reward": [1.5, 2.0], "opening_range_minutes": [10, 15]}
        assert _combinations(grid) == _combinations(grid)

    def test_duplicate_values_are_deduplicated(self) -> None:
        combos = _combinations({"a": [1, 1, 2]})
        assert combos == [{"a": 1}, {"a": 2}]

    def test_empty_grid_produces_one_empty_combination(self) -> None:
        assert _combinations({}) == [{}]


class TestRunParameterSweep:
    def test_creates_independent_child_experiments_with_parent_link(self, db_session: Session, seeded_eurusd) -> None:
        start, end = seeded_eurusd
        registry = ExperimentRegistry(db_session)
        base = _orb_definition(start, end)

        result = run_parameter_sweep(db_session, registry, base, {"risk_reward": [1.5, 2.0, 2.5]})

        parent = registry.get_or_raise(result.parent_experiment_id)
        assert parent.experiment_type == ExperimentType.PARAMETER_SWEEP.value
        assert parent.status == ExperimentStatus.COMPLETED.value
        assert parent.result_summary["combination_count"] == 3

        children = registry.children(result.parent_experiment_id)
        assert len(children) == 3
        risk_rewards = sorted(c.config["strategy_params"]["risk_reward"] for c in children)
        assert risk_rewards == [1.5, 2.0, 2.5]
        for child in children:
            assert child.experiment_type == ExperimentType.SINGLE_BACKTEST.value
            assert child.parent_experiment_id == result.parent_experiment_id

    def test_does_not_mutate_base_definition(self, db_session: Session, seeded_eurusd) -> None:
        start, end = seeded_eurusd
        registry = ExperimentRegistry(db_session)
        base = _orb_definition(start, end)
        original_hash = base.config_hash()

        run_parameter_sweep(db_session, registry, base, {"risk_reward": [1.5, 2.5]})

        assert base.config_hash() == original_hash

    def test_sweep_is_strategy_agnostic(self, db_session: Session, seeded_eurusd) -> None:
        """The sweep engine never references risk_reward or any other
        ORB-specific name in its own code - this test just confirms an
        arbitrary parameter name flows through untouched."""
        start, end = seeded_eurusd
        registry = ExperimentRegistry(db_session)
        base = _orb_definition(start, end)

        result = run_parameter_sweep(db_session, registry, base, {"opening_range_minutes": [10, 15, 20]})
        children = registry.children(result.parent_experiment_id)
        assert sorted(c.config["strategy_params"]["opening_range_minutes"] for c in children) == [10, 15, 20]
