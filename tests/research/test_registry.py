from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.research.models import ExperimentDefinition, ExperimentStatus, ExperimentType
from app.research.registry import ExperimentImmutableError, ExperimentRegistry
from tests.research.conftest import utc


def _definition(**overrides) -> ExperimentDefinition:
    kwargs = dict(
        strategy_name="orb",
        strategy_params={"risk_reward": 2.0},
        instrument_symbol="EURUSD",
        timeframe="5M",
        test_period_start=utc(2024, 1, 1),
        test_period_end=utc(2024, 1, 10),
    )
    kwargs.update(overrides)
    return ExperimentDefinition(**kwargs)


class TestCreateAndRetrieve:
    def test_create_assigns_sequential_id_and_pending_status(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        record = registry.create(_definition())
        assert record.experiment_id == "exp_000001"
        assert record.status == ExperimentStatus.PENDING.value
        assert record.config_hash == _definition().config_hash()

    def test_get_returns_none_for_unknown_id(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        assert registry.get("exp_999999") is None

    def test_get_or_raise_raises_key_error(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        with pytest.raises(KeyError):
            registry.get_or_raise("exp_999999")


class TestListing:
    def test_list_filters_by_strategy_and_instrument(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        registry.create(_definition(strategy_name="orb", instrument_symbol="EURUSD"))
        registry.create(_definition(strategy_name="rsi_divergence", instrument_symbol="GBPUSD"))

        assert [r.strategy_name for r in registry.list(strategy_name="orb")] == ["orb"]
        assert [r.instrument_symbol for r in registry.list(instrument_symbol="GBPUSD")] == ["GBPUSD"]

    def test_list_ordering_is_deterministic(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        for _ in range(5):
            registry.create(_definition())
        ids = [r.experiment_id for r in registry.list()]
        assert ids == sorted(ids)


class TestStatusTransitions:
    def test_mark_running_then_completed(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        record = registry.create(_definition())
        registry.mark_running(record.experiment_id)
        completed = registry.mark_completed(record.experiment_id, {"trade_count": 5})
        assert completed.status == ExperimentStatus.COMPLETED.value
        assert completed.result_summary == {"trade_count": 5}
        assert completed.completed_at is not None

    def test_mark_failed_persists_reason_without_fake_metrics(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        record = registry.create(_definition())
        registry.mark_running(record.experiment_id)
        failed = registry.mark_failed(record.experiment_id, "execution_error: boom")
        assert failed.status == ExperimentStatus.FAILED.value
        assert failed.failure_reason == "execution_error: boom"
        assert failed.result_summary is None


class TestImmutability:
    def test_completed_experiment_cannot_be_recompleted(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        record = registry.create(_definition())
        registry.mark_completed(record.experiment_id, {"trade_count": 1})
        with pytest.raises(ExperimentImmutableError):
            registry.mark_completed(record.experiment_id, {"trade_count": 2})

    def test_completed_experiment_cannot_be_marked_failed(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        record = registry.create(_definition())
        registry.mark_completed(record.experiment_id, {})
        with pytest.raises(ExperimentImmutableError):
            registry.mark_failed(record.experiment_id, "late failure")

    def test_completed_experiment_config_hash_is_stable(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        definition = _definition()
        record = registry.create(definition)
        registry.mark_completed(record.experiment_id, {})
        reloaded = registry.get_or_raise(record.experiment_id)
        assert reloaded.config_hash == definition.config_hash()


class TestParentChild:
    def test_children_returns_only_direct_children_in_order(self, db_session: Session) -> None:
        registry = ExperimentRegistry(db_session)
        parent = registry.create(_definition(experiment_type=ExperimentType.PARAMETER_SWEEP))
        child1 = registry.create(_definition(parent_experiment_id=parent.experiment_id))
        child2 = registry.create(_definition(parent_experiment_id=parent.experiment_id))
        registry.create(_definition())  # unrelated experiment

        children = registry.children(parent.experiment_id)
        assert [c.experiment_id for c in children] == [child1.experiment_id, child2.experiment_id]
