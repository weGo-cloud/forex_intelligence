from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.observations.models import Observation, ObservationSource, ObservationType
from app.setups.detection import SetupDetectionConfig, SetupDetector
from app.setups.models import SetupStatus, SetupType
from app.setups.state_machine import transition

TS = datetime(2024, 1, 1, tzinfo=timezone.utc)
TS2 = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
TS3 = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)


def _sweep(ts=TS, direction="long", confidence=1.0) -> Observation:
    return Observation(
        observation_type=ObservationType.LIQUIDITY_SWEEP, instrument_symbol="EURUSD", timeframe="5M",
        timestamp=ts, source=ObservationSource.MARKET_DATA, direction=direction, price=97.0,
        confidence=confidence, metadata={"swept_level": 99.0},
    )


def _bos(ts=TS2, direction="long") -> Observation:
    return Observation(
        observation_type=ObservationType.BOS, instrument_symbol="EURUSD", timeframe="5M",
        timestamp=ts, source=ObservationSource.MARKET_DATA, direction=direction, price=101.0,
    )


def _support(ts=TS, price=99.0) -> Observation:
    return Observation(
        observation_type=ObservationType.SUPPORT, instrument_symbol="EURUSD", timeframe="5M",
        timestamp=ts, source=ObservationSource.MARKET_DATA, price=price,
    )


class TestSetupCreation:
    def test_confirmed_sweep_produces_valid_setup(self) -> None:
        setups = SetupDetector().detect_liquidity_sweep_setups([_sweep()], [_bos()], "EURUSD", "5M")
        assert len(setups) == 1
        assert setups[0].status == SetupStatus.VALID
        assert setups[0].setup_type == SetupType.LIQUIDITY_SWEEP_REVERSAL
        assert len(setups[0].observations) == 2

    def test_unconfirmed_sweep_produces_detected_setup(self) -> None:
        setups = SetupDetector().detect_liquidity_sweep_setups([_sweep()], [], "EURUSD", "5M")
        assert len(setups) == 1
        assert setups[0].status == SetupStatus.DETECTED
        assert len(setups[0].observations) == 1


class TestSetupDetection:
    def test_detect_combines_both_patterns(self) -> None:
        detector = SetupDetector()
        setups = detector.detect(
            "EURUSD", "5M", sweep_observations=[_sweep()], level_observations=[_support()],
            structure_observations=[_bos()],
        )
        types = {s.setup_type for s in setups}
        assert SetupType.LIQUIDITY_SWEEP_REVERSAL in types
        assert SetupType.SUPPORT_RESISTANCE_REACTION in types

    def test_confirmation_must_be_after_trigger_not_before(self) -> None:
        # BOS occurs BEFORE the sweep - must not count as confirmation.
        early_bos = _bos(ts=datetime(2023, 12, 31, tzinfo=timezone.utc))
        setups = SetupDetector().detect_liquidity_sweep_setups([_sweep()], [early_bos], "EURUSD", "5M")
        assert setups[0].status == SetupStatus.DETECTED

    def test_wrong_direction_confirmation_ignored(self) -> None:
        setups = SetupDetector().detect_liquidity_sweep_setups(
            [_sweep(direction="long")], [_bos(direction="short")], "EURUSD", "5M"
        )
        assert setups[0].status == SetupStatus.DETECTED

    def test_earliest_confirmation_chosen_when_multiple(self) -> None:
        later_bos = _bos(ts=TS3)
        earlier_bos = _bos(ts=TS2)
        setups = SetupDetector().detect_liquidity_sweep_setups([_sweep()], [later_bos, earlier_bos], "EURUSD", "5M")
        assert setups[0].updated_at == TS2


class TestDuplicateSetups:
    def test_two_distinct_sweeps_produce_two_setups(self) -> None:
        sweep_a = _sweep(ts=TS)
        sweep_b = _sweep(ts=TS3)
        setups = SetupDetector().detect_liquidity_sweep_setups([sweep_a, sweep_b], [], "EURUSD", "5M")
        assert len(setups) == 2
        assert setups[0].id != setups[1].id

    def test_identical_sweep_observations_produce_identical_ids(self) -> None:
        setups = SetupDetector().detect_liquidity_sweep_setups([_sweep(), _sweep()], [], "EURUSD", "5M")
        assert setups[0].id == setups[1].id  # same trigger timestamp -> same deterministic id


class TestIncompleteSetup:
    def test_missing_swept_level_metadata_still_produces_a_setup(self) -> None:
        sweep = Observation(
            observation_type=ObservationType.LIQUIDITY_SWEEP, instrument_symbol="EURUSD", timeframe="5M",
            timestamp=TS, source=ObservationSource.MARKET_DATA, direction="long", price=97.0,
        )
        setups = SetupDetector().detect_liquidity_sweep_setups([sweep], [], "EURUSD", "5M")
        assert setups[0].levels[0].price is None


class TestMinimumConfidenceFilter:
    def test_low_confidence_trigger_is_skipped(self) -> None:
        config = SetupDetectionConfig(minimum_trigger_confidence=0.8)
        setups = SetupDetector(config).detect_liquidity_sweep_setups([_sweep(confidence=0.5)], [_bos()], "EURUSD", "5M")
        assert setups == []

    def test_rejects_invalid_confidence_threshold(self) -> None:
        with pytest.raises(ValueError):
            SetupDetectionConfig(minimum_trigger_confidence=1.5)


class TestInvalidationAndExpiration:
    def test_detected_setup_can_be_invalidated(self) -> None:
        setups = SetupDetector().detect_liquidity_sweep_setups([_sweep()], [], "EURUSD", "5M")
        invalidated = transition(setups[0], SetupStatus.INVALIDATED, at=TS3)
        assert invalidated.status == SetupStatus.INVALIDATED

    def test_detected_setup_can_expire(self) -> None:
        setups = SetupDetector().detect_liquidity_sweep_setups([_sweep()], [], "EURUSD", "5M")
        expired = transition(setups[0], SetupStatus.EXPIRED, at=TS3)
        assert expired.status == SetupStatus.EXPIRED
