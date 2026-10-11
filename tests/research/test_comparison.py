from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.research.comparison import IncompatibleExperimentsError, compare_experiments
from app.research.experiment import execute_experiment
from app.research.models import ExperimentDefinition
from app.research.registry import ExperimentRegistry


def _orb_definition(start, end, **overrides) -> ExperimentDefinition:
    kwargs = dict(
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
    kwargs.update(overrides)
    return ExperimentDefinition(**kwargs)


def test_compare_returns_rows_in_requested_order(db_session: Session, seeded_eurusd) -> None:
    start, end = seeded_eurusd
    registry = ExperimentRegistry(db_session)
    a = execute_experiment(db_session, registry, _orb_definition(start, end)).record
    b = execute_experiment(
        db_session, registry, _orb_definition(start, end, strategy_params={
            "instrument_symbol": "EURUSD", "opening_range_minutes": 15,
            "execution_timeframe_minutes": 5, "risk_reward": 3.0,
        })
    ).record

    rows = compare_experiments(registry, [b.experiment_id, a.experiment_id])
    assert [r.experiment_id for r in rows] == [b.experiment_id, a.experiment_id]


def test_compare_rejects_incompatible_instruments(db_session: Session, seeded_eurusd) -> None:
    start, end = seeded_eurusd
    registry = ExperimentRegistry(db_session)
    a = execute_experiment(db_session, registry, _orb_definition(start, end)).record
    other = registry.create(_orb_definition(start, end, instrument_symbol="GBPUSD"))

    with pytest.raises(IncompatibleExperimentsError):
        compare_experiments(registry, [a.experiment_id, other.experiment_id])


def test_compare_passes_through_none_for_undefined_metrics_never_zero(db_session: Session) -> None:
    registry = ExperimentRegistry(db_session)
    from tests.research.conftest import utc

    definition = _orb_definition(utc(2024, 1, 1), utc(2024, 1, 2), instrument_symbol="ZZZUNKNOWN",
                                  strategy_params={"instrument_symbol": "ZZZUNKNOWN"})
    failed = execute_experiment(db_session, registry, definition).record

    rows = compare_experiments(registry, [failed.experiment_id])
    assert rows[0].trade_count is None
    assert rows[0].total_return_pct is None
