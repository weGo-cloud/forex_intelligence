from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.setups.models import InvalidationCondition, Setup, SetupStatus, SetupType
from app.setups.state_machine import InvalidTransitionError, is_valid_transition, transition

TS = datetime(2024, 1, 1, tzinfo=timezone.utc)
TS2 = datetime(2024, 1, 2, tzinfo=timezone.utc)


def _setup(status: SetupStatus) -> Setup:
    return Setup(
        id="test:1", instrument_symbol="EURUSD", timeframe="5M", direction="long",
        setup_type=SetupType.LIQUIDITY_SWEEP_REVERSAL, created_at=TS, updated_at=TS,
        observations=(), levels=(), confirmations=(), invalidation_conditions=(),
        status=status,
    )


VALID_TRANSITIONS = [
    (SetupStatus.DETECTED, SetupStatus.WATCH),
    (SetupStatus.WATCH, SetupStatus.CONFIRMING),
    (SetupStatus.CONFIRMING, SetupStatus.VALID),
    (SetupStatus.VALID, SetupStatus.ENTRY_CONDITION_MET),
    (SetupStatus.DETECTED, SetupStatus.INVALIDATED),
    (SetupStatus.WATCH, SetupStatus.INVALIDATED),
    (SetupStatus.CONFIRMING, SetupStatus.INVALIDATED),
    (SetupStatus.VALID, SetupStatus.INVALIDATED),
    (SetupStatus.ENTRY_CONDITION_MET, SetupStatus.INVALIDATED),
    (SetupStatus.DETECTED, SetupStatus.EXPIRED),
    (SetupStatus.WATCH, SetupStatus.EXPIRED),
    (SetupStatus.VALID, SetupStatus.EXPIRED),
]

INVALID_TRANSITIONS = [
    (SetupStatus.DETECTED, SetupStatus.ENTRY_CONDITION_MET),
    (SetupStatus.DETECTED, SetupStatus.CONFIRMING),
    (SetupStatus.DETECTED, SetupStatus.VALID),
    (SetupStatus.INVALIDATED, SetupStatus.VALID),
    (SetupStatus.INVALIDATED, SetupStatus.DETECTED),
    (SetupStatus.INVALIDATED, SetupStatus.WATCH),
    (SetupStatus.EXPIRED, SetupStatus.ENTRY_CONDITION_MET),
    (SetupStatus.EXPIRED, SetupStatus.VALID),
    (SetupStatus.WATCH, SetupStatus.DETECTED),
    (SetupStatus.VALID, SetupStatus.WATCH),
    (SetupStatus.VALID, SetupStatus.CONFIRMING),
    (SetupStatus.ENTRY_CONDITION_MET, SetupStatus.VALID),
    (SetupStatus.ENTRY_CONDITION_MET, SetupStatus.WATCH),
    (SetupStatus.DETECTED, SetupStatus.DETECTED),  # same-state is not a transition
]


class TestValidTransitions:
    @pytest.mark.parametrize("current,target", VALID_TRANSITIONS)
    def test_is_valid(self, current, target) -> None:
        assert is_valid_transition(current, target) is True

    @pytest.mark.parametrize("current,target", VALID_TRANSITIONS)
    def test_transition_succeeds(self, current, target) -> None:
        setup = _setup(current)
        result = transition(setup, target, at=TS2)
        assert result.status == target
        assert result.updated_at == TS2
        assert setup.status == current  # original untouched (frozen)


class TestInvalidTransitions:
    @pytest.mark.parametrize("current,target", INVALID_TRANSITIONS)
    def test_is_invalid(self, current, target) -> None:
        assert is_valid_transition(current, target) is False

    @pytest.mark.parametrize("current,target", INVALID_TRANSITIONS)
    def test_transition_raises(self, current, target) -> None:
        setup = _setup(current)
        with pytest.raises(InvalidTransitionError):
            transition(setup, target, at=TS2)


class TestTransitionReason:
    def test_reason_is_recorded_in_metadata(self) -> None:
        setup = _setup(SetupStatus.DETECTED)
        result = transition(setup, SetupStatus.WATCH, at=TS2, reason="price approaching level")
        assert result.metadata["last_transition_reason"] == "price approaching level"

    def test_no_reason_leaves_metadata_unchanged(self) -> None:
        setup = _setup(SetupStatus.DETECTED)
        result = transition(setup, SetupStatus.WATCH, at=TS2)
        assert "last_transition_reason" not in result.metadata
