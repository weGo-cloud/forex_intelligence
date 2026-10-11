from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.observations.models import Observation, ObservationSource, ObservationType


def _obs(**overrides) -> Observation:
    kwargs = dict(
        observation_type=ObservationType.SWING_HIGH,
        instrument_symbol="EURUSD",
        timeframe="5M",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source=ObservationSource.MARKET_DATA,
        price=100.0,
    )
    kwargs.update(overrides)
    return Observation(**kwargs)


class TestValidObservation:
    def test_minimal_valid_observation(self) -> None:
        obs = _obs()
        assert obs.observation_type == ObservationType.SWING_HIGH
        assert obs.confidence == 1.0

    def test_zone_observation(self) -> None:
        obs = _obs(observation_type=ObservationType.FVG, price=None, zone_low=100.0, zone_high=102.0)
        assert obs.zone_low == 100.0
        assert obs.zone_high == 102.0


class TestInvalidObservation:
    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            _obs(confidence=1.5)

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            _obs(confidence=-0.1)

    def test_zone_low_above_zone_high_rejected(self) -> None:
        with pytest.raises(ValueError):
            _obs(zone_low=105.0, zone_high=100.0)


class TestConfidenceBoundaries:
    def test_confidence_zero_is_valid(self) -> None:
        _obs(confidence=0.0)

    def test_confidence_one_is_valid(self) -> None:
        _obs(confidence=1.0)


class TestSourceHandling:
    def test_every_source_constructs(self) -> None:
        for source in ObservationSource:
            _obs(source=source)

    def test_screenshot_source_is_representable(self) -> None:
        """The hard architectural requirement (mission section 6/14):
        a screenshot-derived observation must be constructible with
        the exact same model a market-data observation uses."""
        obs = _obs(source=ObservationSource.SCREENSHOT)
        assert obs.source == ObservationSource.SCREENSHOT


class TestSerialization:
    def test_to_dict_is_json_safe(self) -> None:
        obs = _obs()
        json.dumps(obs.to_dict())

    def test_enum_values_serialize_as_strings(self) -> None:
        obs = _obs()
        payload = obs.to_dict()
        assert payload["observation_type"] == "swing_high"
        assert payload["source"] == "market_data"
