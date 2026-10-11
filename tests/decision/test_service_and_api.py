from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.decision.models import DecisionState
from app.decision.service import detect_setups, evaluate_decisions
from app.main import app
from app.observations.models import Observation, ObservationSource, ObservationType
from app.setups.models import SetupStatus

T0 = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
T1 = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
T2 = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)

EVIDENCE = {
    "total_observations": 40, "resolved_observations": 40, "evidence_tier": "adequate",
    "p_win": 0.6, "historical_expectancy": 0.3, "warnings": [],
}


def _obs(kind, ts, source=ObservationSource.MARKET_DATA, **kw) -> Observation:
    return Observation(
        observation_type=kind, instrument_symbol="EURUSD", timeframe="5M", timestamp=ts,
        source=source, direction="long", price=97.0, **kw,
    )


class TestDomainService:
    def test_detect_routes_by_type(self) -> None:
        obs = [_obs(ObservationType.LIQUIDITY_SWEEP, T0, metadata={"swept_level": 99.0}), _obs(ObservationType.BOS, T1)]
        setups = detect_setups("EURUSD", "5M", obs)
        assert len(setups) == 1
        assert setups[0].status == SetupStatus.VALID

    def test_detect_as_of_excludes_later_confirmation(self) -> None:
        obs = [_obs(ObservationType.LIQUIDITY_SWEEP, T0, metadata={"swept_level": 99.0}), _obs(ObservationType.BOS, T2)]
        assert detect_setups("EURUSD", "5M", obs, as_of=T1)[0].status == SetupStatus.DETECTED
        assert detect_setups("EURUSD", "5M", obs, as_of=T2)[0].status == SetupStatus.VALID

    def test_other_instruments_ignored(self) -> None:
        obs = [_obs(ObservationType.LIQUIDITY_SWEEP, T0, metadata={"swept_level": 99.0})]
        assert detect_setups("GBPUSD", "5M", obs) == []

    def test_evaluate_without_evidence_is_insufficient(self) -> None:
        obs = [_obs(ObservationType.LIQUIDITY_SWEEP, T0, metadata={"swept_level": 99.0}), _obs(ObservationType.BOS, T1)]
        results = evaluate_decisions("EURUSD", "5M", obs, at=T2, evidence=None)
        assert results[0].decision.state == DecisionState.INSUFFICIENT_EVIDENCE


class TestScreenshotCompatibility:
    """Mission sections 6/14: screenshot-derived observations must
    enter the SAME pipeline and produce the SAME result."""

    def test_source_does_not_change_setup_or_decision(self) -> None:
        def run(source):
            obs = [
                _obs(ObservationType.LIQUIDITY_SWEEP, T0, source=source, metadata={"swept_level": 99.0}),
                _obs(ObservationType.BOS, T1, source=source),
            ]
            results = evaluate_decisions("EURUSD", "5M", obs, at=T2, evidence=None)
            return results[0].decision.state, results[0].setup.status

        assert run(ObservationSource.MARKET_DATA) == run(ObservationSource.SCREENSHOT) == run(ObservationSource.MANUAL)


class TestObservationDeserialization:
    def test_round_trip(self) -> None:
        obs = _obs(ObservationType.FVG, T0, zone_low=1.0, zone_high=2.0, metadata={"kind": "bullish"})
        assert Observation.from_dict(obs.to_dict()) == obs

    def test_unknown_type_rejected(self) -> None:
        payload = _obs(ObservationType.BOS, T0).to_dict()
        payload["observation_type"] = "made_up"
        with pytest.raises(ValueError):
            Observation.from_dict(payload)

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError):
            Observation(
                observation_type=ObservationType.BOS, instrument_symbol="EURUSD", timeframe="5M",
                timestamp=datetime(2024, 1, 1), source=ObservationSource.MARKET_DATA,
            )


client = TestClient(app)


def _payload(kind, ts, source="market_data", **kw):
    base = {
        "observation_type": kind, "instrument_symbol": "EURUSD", "timeframe": "5M",
        "timestamp": ts.isoformat(), "source": source, "direction": "long", "price": 97.0,
    }
    base.update(kw)
    return base


class TestApi:
    def test_detect_endpoint(self) -> None:
        body = {
            "instrument_symbol": "EURUSD", "timeframe": "5M",
            "observations": [
                _payload("liquidity_sweep", T0, metadata={"swept_level": 99.0}),
                _payload("bos", T1),
            ],
        }
        response = client.post("/setups/detect", json=body)
        assert response.status_code == 200
        setups = response.json()["setups"]
        assert len(setups) == 1
        assert setups[0]["status"] == "valid"

    def test_evaluate_endpoint_with_evidence(self) -> None:
        body = {
            "instrument_symbol": "EURUSD", "timeframe": "5M", "at": T2.isoformat(), "evidence": EVIDENCE,
            "observations": [_payload("liquidity_sweep", T0, metadata={"swept_level": 99.0}), _payload("bos", T1)],
        }
        response = client.post("/setups/evaluate", json=body)
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["decision"]["state"] == "valid_setup"
        assert result["decision"]["invalidation_conditions"]

    def test_evaluate_endpoint_without_evidence_is_insufficient(self) -> None:
        body = {
            "instrument_symbol": "EURUSD", "timeframe": "5M", "at": T2.isoformat(),
            "observations": [_payload("liquidity_sweep", T0, metadata={"swept_level": 99.0})],
        }
        result = client.post("/setups/evaluate", json=body).json()["results"][0]
        assert result["decision"]["state"] == "insufficient_evidence"

    def test_screenshot_source_accepted_over_api(self) -> None:
        body = {
            "instrument_symbol": "EURUSD", "timeframe": "5M", "at": T2.isoformat(), "evidence": EVIDENCE,
            "observations": [
                _payload("liquidity_sweep", T0, source="screenshot", metadata={"swept_level": 99.0}),
                _payload("bos", T1, source="screenshot"),
            ],
        }
        response = client.post("/setups/evaluate", json=body)
        assert response.status_code == 200
        assert response.json()["results"][0]["decision"]["state"] == "valid_setup"

    def test_invalid_observation_type_returns_422(self) -> None:
        body = {"instrument_symbol": "EURUSD", "timeframe": "5M", "observations": [_payload("nonsense", T0)]}
        assert client.post("/setups/detect", json=body).status_code == 422

    def test_invalid_evidence_returns_422(self) -> None:
        body = {
            "instrument_symbol": "EURUSD", "timeframe": "5M", "at": T2.isoformat(),
            "evidence": {"evidence_tier": "adequate"}, "observations": [],
        }
        assert client.post("/setups/evaluate", json=body).status_code == 422

    def test_naive_timestamp_rejected_by_api(self) -> None:
        body = {
            "instrument_symbol": "EURUSD", "timeframe": "5M",
            "observations": [_payload("bos", T0) | {"timestamp": "2024-01-01T00:00:00"}],
        }
        assert client.post("/setups/detect", json=body).status_code == 422
