"""Canonical setup representation.

`SetupType` is intentionally a plain enum, not a plugin registry -
"the smallest set of new abstractions required" (mission section 3.5)
for M7's two concrete detectors. Adding a new setup type later is one
new enum member plus one new detector function in
`app.setups.detection` that produces `Setup`s the same way the
existing two do - nothing about `Setup`, the state machine, or
`app.decision` is hard-coded to any one type.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.observations.models import Observation
from app.strategies.base import Direction


class SetupType(str, enum.Enum):
    LIQUIDITY_SWEEP_REVERSAL = "liquidity_sweep_reversal"
    SUPPORT_RESISTANCE_REACTION = "support_resistance_reaction"


class SetupStatus(str, enum.Enum):
    """Mission section 8's seven states, verbatim. Transitions between
    them are governed entirely by `app.setups.state_machine` - nothing
    outside that module decides whether a transition is legal, and
    nothing (LLM or otherwise) sets `status` directly without going
    through it (see `state_machine.transition`)."""

    DETECTED = "detected"
    WATCH = "watch"
    CONFIRMING = "confirming"
    VALID = "valid"
    ENTRY_CONDITION_MET = "entry_condition_met"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


@dataclass(frozen=True)
class Level:
    """A named price or zone attached to a setup (e.g. the swept level,
    an entry zone, a target) - deliberately NOT the same type as
    `Observation` (a `Level` is a setup's own interpreted reference
    point, which may or may not trace back to a single observation)."""

    label: str
    price: Optional[float] = None
    zone_low: Optional[float] = None
    zone_high: Optional[float] = None


@dataclass(frozen=True)
class InvalidationCondition:
    """A structured, checkable invalidation rule - never just a prose
    string. `price`/`direction` are set when the condition is a
    concrete "price closes beyond this level in this direction" check;
    `description` is always human-readable regardless."""

    description: str
    price: Optional[float] = None
    direction: Optional[Direction] = None


@dataclass(frozen=True)
class Setup:
    """A candidate trading structure assembled from `Observation`s.
    Carries no historical evidence of its own and makes no decision -
    `app.decision` is what combines a `Setup` with M6 evidence to
    produce a `Decision`. `evidence` exists as an OPTIONAL attachment
    point (mirroring the mission's own field list on `Setup`) that
    `app.decision` populates via `app.decision.evidence.evidence_from_evaluation`
    - `Setup` itself never computes it.
    """

    id: str
    instrument_symbol: str
    timeframe: str
    direction: Direction
    setup_type: SetupType

    created_at: datetime
    updated_at: datetime

    observations: tuple[Observation, ...]
    levels: tuple[Level, ...]
    confirmations: tuple[str, ...]
    invalidation_conditions: tuple[InvalidationCondition, ...]

    status: SetupStatus

    expires_at: Optional[datetime] = None
    evidence: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instrument_symbol": self.instrument_symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "setup_type": self.setup_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "observations": [o.to_dict() for o in self.observations],
            "levels": [
                {"label": lv.label, "price": lv.price, "zone_low": lv.zone_low, "zone_high": lv.zone_high}
                for lv in self.levels
            ],
            "confirmations": list(self.confirmations),
            "invalidation_conditions": [
                {"description": ic.description, "price": ic.price, "direction": ic.direction}
                for ic in self.invalidation_conditions
            ],
            "status": self.status.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "evidence": self.evidence,
            "metadata": dict(self.metadata),
        }
