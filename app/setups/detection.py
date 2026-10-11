"""Setup detection - assembles `Observation`s into `Setup` candidates.

Two concrete, deterministic patterns are implemented, per the mission's
explicit "do not attempt every possible setup... implement a clean
extensible foundation and at least a small number of deterministic
setup patterns supported by the existing M1-M6 capabilities":

1. `detect_liquidity_sweep_setups` - a `LIQUIDITY_SWEEP` observation
   followed by a matching-direction `BOS` observation.
2. `detect_support_resistance_setups` - a `SUPPORT`/`RESISTANCE`
   observation followed by a matching-direction `BOS`/`CHOCH`
   observation.

Both patterns look for the EARLIEST confirming structure observation
whose `timestamp` is strictly after the trigger observation's own
`timestamp` - never at or before it (that would be using the
confirmation to detect the very thing it's supposed to confirm).

SCOPE NOTE: this foundation's detectors produce only `DETECTED` (no
confirmation found yet) or `VALID` (a confirming observation found) -
they do not themselves produce `WATCH`/`CONFIRMING`/
`ENTRY_CONDITION_MET`. Those states exist and are fully supported by
`app.setups.state_machine` for a future live/incremental detector (or
manual re-evaluation) to use as a setup's evidence accumulates over
time; a single deterministic pass over already-complete historical
observations has no natural notion of "still forming" - either the
confirming observation is there in the data or it isn't yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.observations.models import Observation, ObservationType
from app.setups.models import InvalidationCondition, Level, Setup, SetupStatus, SetupType


@dataclass(frozen=True)
class SetupDetectionConfig:
    # A minimum confidence a trigger observation (S/R level touch
    # count, etc.) must meet before it's considered at all - filters
    # out barely-evidenced levels without a separate hard-coded
    # threshold buried in the detector logic.
    minimum_trigger_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.minimum_trigger_confidence <= 1.0):
            raise ValueError("minimum_trigger_confidence must be between 0.0 and 1.0")


def _earliest_confirmation(
    trigger: Observation, structure_observations: list[Observation], allowed_types: tuple[ObservationType, ...]
) -> Observation | None:
    candidates = [
        o
        for o in structure_observations
        if o.direction == trigger.direction and o.observation_type in allowed_types and o.timestamp > trigger.timestamp
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda o: o.timestamp)


class SetupDetector:
    def __init__(self, config: SetupDetectionConfig | None = None) -> None:
        self.config = config or SetupDetectionConfig()

    def detect_liquidity_sweep_setups(
        self,
        sweep_observations: list[Observation],
        structure_observations: list[Observation],
        instrument_symbol: str,
        timeframe: str,
    ) -> list[Setup]:
        setups: list[Setup] = []
        for sweep in sweep_observations:
            if sweep.confidence < self.config.minimum_trigger_confidence:
                continue

            confirmation = _earliest_confirmation(sweep, structure_observations, (ObservationType.BOS,))
            swept_level = sweep.metadata.get("swept_level")

            status = SetupStatus.VALID if confirmation is not None else SetupStatus.DETECTED
            confirmations = (
                (f"Structure confirmation (BOS) at {confirmation.timestamp.isoformat()}",) if confirmation else ()
            )
            observations = (sweep, confirmation) if confirmation else (sweep,)
            updated_at = confirmation.timestamp if confirmation else sweep.timestamp

            invalidation = InvalidationCondition(
                description=(
                    f"Price closes beyond the swept level ({swept_level}) in the direction "
                    "opposite the setup, without the expected rejection holding"
                ),
                price=swept_level,
                direction=sweep.direction,
            )

            setups.append(
                Setup(
                    id=f"liquidity_sweep_reversal:{instrument_symbol}:{timeframe}:{sweep.timestamp.isoformat()}",
                    instrument_symbol=instrument_symbol,
                    timeframe=timeframe,
                    direction=sweep.direction,
                    setup_type=SetupType.LIQUIDITY_SWEEP_REVERSAL,
                    created_at=sweep.timestamp,
                    updated_at=updated_at,
                    observations=observations,
                    levels=(Level(label="swept_level", price=swept_level),),
                    confirmations=confirmations,
                    invalidation_conditions=(invalidation,),
                    status=status,
                )
            )
        return setups

    def detect_support_resistance_setups(
        self,
        level_observations: list[Observation],
        structure_observations: list[Observation],
        instrument_symbol: str,
        timeframe: str,
    ) -> list[Setup]:
        setups: list[Setup] = []
        for level_obs in level_observations:
            if level_obs.confidence < self.config.minimum_trigger_confidence:
                continue
            if level_obs.observation_type not in (ObservationType.SUPPORT, ObservationType.RESISTANCE):
                continue

            direction = "long" if level_obs.observation_type == ObservationType.SUPPORT else "short"
            confirmation = _earliest_confirmation(
                Observation(
                    observation_type=level_obs.observation_type,
                    instrument_symbol=level_obs.instrument_symbol,
                    timeframe=level_obs.timeframe,
                    timestamp=level_obs.timestamp,
                    source=level_obs.source,
                    direction=direction,
                ),
                structure_observations,
                (ObservationType.BOS, ObservationType.CHOCH),
            )

            status = SetupStatus.VALID if confirmation is not None else SetupStatus.DETECTED
            confirmations = (
                (f"Structure confirmation ({confirmation.observation_type.value}) at {confirmation.timestamp.isoformat()}",)
                if confirmation
                else ()
            )
            observations = (level_obs, confirmation) if confirmation else (level_obs,)
            updated_at = confirmation.timestamp if confirmation else level_obs.timestamp

            invalidation = InvalidationCondition(
                description=f"Price closes through the level ({level_obs.price}) against the setup direction",
                price=level_obs.price,
                direction=direction,
            )

            setups.append(
                Setup(
                    id=f"support_resistance_reaction:{instrument_symbol}:{timeframe}:{level_obs.timestamp.isoformat()}",
                    instrument_symbol=instrument_symbol,
                    timeframe=timeframe,
                    direction=direction,
                    setup_type=SetupType.SUPPORT_RESISTANCE_REACTION,
                    created_at=level_obs.timestamp,
                    updated_at=updated_at,
                    observations=observations,
                    levels=(Level(label=level_obs.observation_type.value, price=level_obs.price),),
                    confirmations=confirmations,
                    invalidation_conditions=(invalidation,),
                    status=status,
                )
            )
        return setups

    def detect(
        self,
        instrument_symbol: str,
        timeframe: str,
        sweep_observations: list[Observation] | None = None,
        level_observations: list[Observation] | None = None,
        structure_observations: list[Observation] | None = None,
    ) -> list[Setup]:
        """Runs every implemented pattern and returns the combined
        candidate list. New patterns join this method as new
        `detect_*` methods are added - no caller code changes when a
        third pattern is introduced later."""
        structure_observations = structure_observations or []
        setups: list[Setup] = []
        setups += self.detect_liquidity_sweep_setups(
            sweep_observations or [], structure_observations, instrument_symbol, timeframe
        )
        setups += self.detect_support_resistance_setups(
            level_observations or [], structure_observations, instrument_symbol, timeframe
        )
        return setups
