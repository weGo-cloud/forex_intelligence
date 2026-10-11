"""Domain-level service tying the M7 pipeline together:

    observations -> SetupDetector -> Setups -> DecisionEngine (+ M6Evidence) -> Decisions

Kept as plain functions over plain domain objects so it is usable
directly from Python (research scripts, tests, a future screenshot
analyzer) - `app.api.setups` is a thin HTTP wrapper over exactly these
functions and adds no logic of its own (mission section 15: "prefer
domain services over bloated route handlers").

Observations are routed to detectors purely by `observation_type` -
NEVER by `source`. A screenshot-derived `LIQUIDITY_SWEEP` and a
market-data-derived one take the identical path (mission section 14).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.decision.engine import DecisionEngine
from app.decision.models import Decision, M6Evidence
from app.observations.models import Observation, ObservationType, observations_as_of
from app.setups.detection import SetupDetector
from app.setups.models import Setup

_SWEEP_TYPES = (ObservationType.LIQUIDITY_SWEEP,)
_LEVEL_TYPES = (ObservationType.SUPPORT, ObservationType.RESISTANCE)
_STRUCTURE_TYPES = (ObservationType.BOS, ObservationType.CHOCH)


@dataclass(frozen=True)
class SetupEvaluation:
    setup: Setup
    decision: Decision


def detect_setups(
    instrument_symbol: str,
    timeframe: str,
    observations: list[Observation],
    as_of: Optional[datetime] = None,
    detector: Optional[SetupDetector] = None,
) -> list[Setup]:
    """`as_of`, if given, restricts detection to observations knowable
    at that time (`observations_as_of`) - the walk-forward-safe entry
    point for historical evaluation."""
    if as_of is not None:
        observations = observations_as_of(observations, as_of)

    relevant = [o for o in observations if o.instrument_symbol == instrument_symbol and o.timeframe == timeframe]
    return (detector or SetupDetector()).detect(
        instrument_symbol,
        timeframe,
        sweep_observations=[o for o in relevant if o.observation_type in _SWEEP_TYPES],
        level_observations=[o for o in relevant if o.observation_type in _LEVEL_TYPES],
        structure_observations=[o for o in relevant if o.observation_type in _STRUCTURE_TYPES],
    )


def evaluate_decisions(
    instrument_symbol: str,
    timeframe: str,
    observations: list[Observation],
    at: datetime,
    evidence: Optional[M6Evidence] = None,
    engine: Optional[DecisionEngine] = None,
    detector: Optional[SetupDetector] = None,
) -> list[SetupEvaluation]:
    """Detect setups as of `at` and produce one `Decision` per setup.
    `evidence` is supplied by the caller (from M6, via
    `app.decision.evidence.evidence_from_evaluation`) - this function
    never computes historical evidence itself."""
    setups = detect_setups(instrument_symbol, timeframe, observations, as_of=at, detector=detector)
    decision_engine = engine or DecisionEngine()
    return [SetupEvaluation(setup=s, decision=decision_engine.decide(s, evidence, at=at)) for s in setups]
