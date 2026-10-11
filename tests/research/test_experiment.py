from __future__ import annotations

from sqlalchemy.orm import Session

from app.research.experiment import execute_experiment
from app.research.models import ExperimentDefinition, ExperimentStatus
from app.research.registry import ExperimentRegistry
from tests.research.conftest import utc


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


def test_execute_experiment_completes_and_persists_summary(db_session: Session, seeded_eurusd) -> None:
    start, end = seeded_eurusd
    registry = ExperimentRegistry(db_session)
    outcome = execute_experiment(db_session, registry, _orb_definition(start, end))

    assert outcome.record.status == ExperimentStatus.COMPLETED.value
    assert outcome.result is not None
    assert outcome.record.result_summary["trade_count"] == outcome.result.metrics.trade_count
    # Full trade log is available on the returned result but not duplicated into the registry row.
    assert "trades" not in outcome.record.result_summary


def test_execute_experiment_marks_unknown_instrument_as_failed_not_zero_metrics(db_session: Session) -> None:
    registry = ExperimentRegistry(db_session)
    definition = ExperimentDefinition(
        strategy_name="orb",
        strategy_params={"instrument_symbol": "ZZZUNKNOWN"},
        instrument_symbol="ZZZUNKNOWN",
        timeframe="5M",
        test_period_start=utc(2024, 1, 1),
        test_period_end=utc(2024, 1, 2),
    )
    outcome = execute_experiment(db_session, registry, definition)

    assert outcome.record.status == ExperimentStatus.FAILED.value
    assert outcome.record.result_summary is None
    assert "unknown_instrument" in outcome.record.failure_reason
    assert outcome.result is None
