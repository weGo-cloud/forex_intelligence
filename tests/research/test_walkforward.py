from __future__ import annotations

from sqlalchemy.orm import Session

from app.research.models import ExperimentDefinition, ExperimentStatus, ExperimentType
from app.research.registry import ExperimentRegistry
from app.research.runner import run_walk_forward_experiment


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


def test_walk_forward_creates_one_child_per_window_with_correct_test_period(
    db_session: Session, seeded_eurusd
) -> None:
    start, end = seeded_eurusd
    registry = ExperimentRegistry(db_session)
    base = _orb_definition(start, end)

    result = run_walk_forward_experiment(db_session, registry, base, train_days=2, test_days=2)

    parent = registry.get_or_raise(result.parent_experiment_id)
    assert parent.experiment_type == ExperimentType.WALK_FORWARD.value
    assert parent.status == ExperimentStatus.COMPLETED.value
    assert parent.result_summary["window_count"] == len(result.children)
    assert len(result.children) >= 1

    for child_result in result.children:
        child = registry.get_or_raise(child_result.outcome.record.experiment_id)
        assert child.parent_experiment_id == result.parent_experiment_id
        assert child.config["test_period_start"] == child_result.window.test_start.isoformat()
        assert child.config["test_period_end"] == child_result.window.test_end.isoformat()
        # Child test period must be the WINDOW's test range, never the overall base range.
        assert child.config["test_period_start"] != base.test_period_start.isoformat() or child_result.window.window_index == 0


def test_windows_do_not_overlap_and_train_precedes_test(db_session: Session, seeded_eurusd) -> None:
    start, end = seeded_eurusd
    registry = ExperimentRegistry(db_session)
    base = _orb_definition(start, end)

    result = run_walk_forward_experiment(db_session, registry, base, train_days=2, test_days=1, step_days=1)

    for child_result in result.children:
        window = child_result.window
        assert window.train_end == window.test_start
        assert window.test_start < window.test_end

    for a, b in zip(result.children, result.children[1:]):
        assert a.window.test_end <= b.window.test_start


def test_no_windows_fit_raises_value_error(db_session: Session, seeded_eurusd) -> None:
    start, end = seeded_eurusd
    registry = ExperimentRegistry(db_session)
    base = _orb_definition(start, end)

    import pytest

    with pytest.raises(ValueError):
        run_walk_forward_experiment(db_session, registry, base, train_days=1000, test_days=1000)
