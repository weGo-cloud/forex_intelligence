from __future__ import annotations

import json
from datetime import datetime, timezone

from app.observations.models import Observation, ObservationSource, ObservationType
from app.setups.models import InvalidationCondition, Level, Setup, SetupStatus, SetupType

TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _setup(**overrides) -> Setup:
    kwargs = dict(
        id="test:1", instrument_symbol="EURUSD", timeframe="5M", direction="long",
        setup_type=SetupType.LIQUIDITY_SWEEP_REVERSAL, created_at=TS, updated_at=TS,
        observations=(), levels=(Level(label="swept_level", price=99.0),),
        confirmations=("BOS confirmed",),
        invalidation_conditions=(InvalidationCondition(description="closes below 99", price=99.0, direction="long"),),
        status=SetupStatus.VALID,
    )
    kwargs.update(overrides)
    return Setup(**kwargs)


class TestSerialization:
    def test_to_dict_is_json_safe(self) -> None:
        setup = _setup()
        json.dumps(setup.to_dict())

    def test_includes_observations(self) -> None:
        obs = Observation(
            observation_type=ObservationType.LIQUIDITY_SWEEP, instrument_symbol="EURUSD", timeframe="5M",
            timestamp=TS, source=ObservationSource.MARKET_DATA, direction="long", price=99.0,
        )
        setup = _setup(observations=(obs,))
        payload = setup.to_dict()
        assert len(payload["observations"]) == 1
        assert payload["observations"][0]["observation_type"] == "liquidity_sweep"

    def test_levels_and_invalidation_serialize(self) -> None:
        payload = _setup().to_dict()
        assert payload["levels"][0]["label"] == "swept_level"
        assert payload["invalidation_conditions"][0]["price"] == 99.0

    def test_evidence_defaults_to_none(self) -> None:
        assert _setup().to_dict()["evidence"] is None


class TestImmutability:
    def test_setup_is_frozen(self) -> None:
        setup = _setup()
        try:
            setup.status = SetupStatus.INVALIDATED  # type: ignore[misc]
            assert False, "should have raised"
        except Exception:
            pass
