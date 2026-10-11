from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.research.robustness import MIN_TRADES_FOR_STABILITY_JUDGEMENT, classify_stability
from app.research.models import ExperimentDefinition
from app.research.registry import ExperimentRegistry
from app.research.robustness import run_robustness_analysis
from sqlalchemy.orm import Session


@dataclass
class _FakeMetrics:
    trade_count: int
    total_return_pct: float


@dataclass
class _FakeResult:
    metrics: _FakeMetrics


@dataclass
class _FakeOutcome:
    result: Optional[_FakeResult]


@dataclass
class _FakeChild:
    parameter_combination: dict
    outcome: _FakeOutcome


def _child(trade_count: int, total_return_pct: float) -> _FakeChild:
    return _FakeChild(
        parameter_combination={},
        outcome=_FakeOutcome(result=_FakeResult(metrics=_FakeMetrics(trade_count, total_return_pct))),
    )


class TestClassifyStability:
    def test_insufficient_data_when_any_neighbor_has_too_few_trades(self) -> None:
        children = [_child(20, 5.0), _child(MIN_TRADES_FOR_STABILITY_JUDGEMENT - 1, 5.0)]
        label, _ = classify_stability(children)
        assert label == "insufficient_data"

    def test_insufficient_data_when_a_neighbor_failed_to_run(self) -> None:
        children = [_child(20, 5.0), _FakeChild(parameter_combination={}, outcome=_FakeOutcome(result=None))]
        label, _ = classify_stability(children)
        assert label == "insufficient_data"

    def test_isolated_when_signs_disagree_across_majority(self) -> None:
        children = [_child(20, 5.0), _child(20, -5.0), _child(20, -4.0)]
        label, _ = classify_stability(children)
        assert label == "isolated"

    def test_stable_when_returns_are_consistent_same_sign(self) -> None:
        children = [_child(20, 5.0), _child(20, 5.2), _child(20, 4.8)]
        label, _ = classify_stability(children)
        assert label == "stable"

    def test_sensitive_when_variability_exceeds_mean_magnitude(self) -> None:
        children = [_child(20, 1.0), _child(20, 20.0), _child(20, 0.5)]
        label, _ = classify_stability(children)
        assert label == "sensitive"


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


def test_run_robustness_analysis_end_to_end(db_session: Session, seeded_eurusd) -> None:
    start, end = seeded_eurusd
    registry = ExperimentRegistry(db_session)
    base = _orb_definition(start, end)

    result = run_robustness_analysis(db_session, registry, base, "risk_reward", [1.5, 1.75, 2.0, 2.25, 2.5])

    assert result.stability_label in {"stable", "sensitive", "isolated", "insufficient_data"}
    parent = registry.get_or_raise(result.parent_experiment_id)
    assert parent.result_summary["stability_label"] == result.stability_label
    assert len(result.children) == 5


def test_run_robustness_analysis_requires_at_least_two_values(db_session: Session, seeded_eurusd) -> None:
    start, end = seeded_eurusd
    registry = ExperimentRegistry(db_session)
    base = _orb_definition(start, end)

    import pytest

    with pytest.raises(ValueError):
        run_robustness_analysis(db_session, registry, base, "risk_reward", [2.0])
