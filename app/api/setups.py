"""Thin HTTP wrapper over `app.decision.service` - request parsing and
error translation only; every rule lives in the domain layer. Stateless
(no database): observations arrive in the request body, so a future
screenshot analyzer can POST `source="screenshot"` observations here and
receive the same setups/decisions market-data observations produce.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.decision.api_schemas import DetectRequest, EvaluateRequest
from app.decision.models import M6Evidence
from app.decision.service import detect_setups, evaluate_decisions
from app.observations.models import Observation

router = APIRouter(prefix="/setups", tags=["setups"])


def _parse_observations(payloads) -> list[Observation]:
    try:
        return [Observation.from_dict({**p.model_dump(mode="json")}) for p in payloads]
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid observation: {exc}") from exc


@router.post("/detect")
def detect(request: DetectRequest) -> dict:
    observations = _parse_observations(request.observations)
    setups = detect_setups(request.instrument_symbol, request.timeframe, observations, as_of=request.as_of)
    return {"setups": [s.to_dict() for s in setups]}


@router.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    observations = _parse_observations(request.observations)
    try:
        evidence = M6Evidence.from_dict(request.evidence) if request.evidence is not None else None
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid evidence: {exc}") from exc

    results = evaluate_decisions(
        request.instrument_symbol, request.timeframe, observations, at=request.at, evidence=evidence
    )
    return {"results": [{"setup": r.setup.to_dict(), "decision": r.decision.to_dict()} for r in results]}
