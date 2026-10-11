"""Decision domain model.

`M6Evidence` is a BRIDGE, not a recomputation: every field is read
directly off an existing `app.evaluation.models.StrategyContextEvaluation`
(M6) - see `app.decision.evidence.evidence_from_evaluation`, the one
function that constructs it. Nothing in `app.decision` calls
`ProbabilityEstimator` or `StrategyEvaluator` itself.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.evaluation.models import EvidenceTier
from app.strategies.base import Direction


class DecisionState(str, enum.Enum):
    """Mission section 9's six states, verbatim. This is an ACTION
    STATE, never a prediction or a probability - "WATCH" and
    "VALID_SETUP" describe what the evidence and setup structure
    currently justify, not what will happen."""

    WAIT = "wait"
    WATCH = "watch"
    VALID_SETUP = "valid_setup"
    ENTRY_CONDITION_MET = "entry_condition_met"
    INVALIDATED = "invalidated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class M6Evidence:
    """A read-only snapshot of an M6 `StrategyContextEvaluation` -
    every field here already existed on that object or one it embeds
    (`ProbabilityEstimate`, `StabilityMetrics`); this dataclass exists
    only to give `app.decision` a stable, narrow, decision-relevant
    view rather than passing the whole M6 object (and its M5-shaped
    internals) around.

    `regime_filter` is the CONDITIONING FILTER the evaluation was run
    with (trend/volatility/structure/session), not a live
    `RegimeContext` snapshot - `StrategyContextEvaluation` records what
    was filtered on, not a captured regime object, so that is the
    honest thing to expose here.
    """

    strategy_name: Optional[str]
    instrument_symbol: Optional[str]
    timeframe: Optional[str]
    direction: Optional[Direction]

    total_observations: int
    resolved_observations: int
    evidence_tier: EvidenceTier

    p_win: Optional[float]
    win_interval: Optional[dict[str, Any]]
    historical_expectancy: Optional[float]

    stability: Optional[dict[str, Any]]
    walk_forward_window_count: Optional[int]

    regime_filter: dict[str, Any]
    warnings: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "M6Evidence":
        """Inverse of `to_dict()` - lets a caller that already holds a
        serialized M6 evidence snapshot (e.g. over the API) hand it to
        the decision engine without re-running M6."""
        return cls(
            strategy_name=data.get("strategy_name"),
            instrument_symbol=data.get("instrument_symbol"),
            timeframe=data.get("timeframe"),
            direction=data.get("direction"),
            total_observations=data["total_observations"],
            resolved_observations=data["resolved_observations"],
            evidence_tier=EvidenceTier(data["evidence_tier"]),
            p_win=data.get("p_win"),
            win_interval=data.get("win_interval"),
            historical_expectancy=data.get("historical_expectancy"),
            stability=data.get("stability"),
            walk_forward_window_count=data.get("walk_forward_window_count"),
            regime_filter=dict(data.get("regime_filter") or {}),
            warnings=tuple(data.get("warnings") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "instrument_symbol": self.instrument_symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "total_observations": self.total_observations,
            "resolved_observations": self.resolved_observations,
            "evidence_tier": self.evidence_tier.value,
            "p_win": self.p_win,
            "win_interval": self.win_interval,
            "historical_expectancy": self.historical_expectancy,
            "stability": self.stability,
            "walk_forward_window_count": self.walk_forward_window_count,
            "regime_filter": self.regime_filter,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class Decision:
    """The explanation contract (mission section 12). Every field is
    structured data - no prose-generation happens in this layer (see
    `app.decision.engine`'s module docstring); a future UI/LLM layer
    turns this into natural language, not the other way around."""

    state: DecisionState
    setup_id: Optional[str]
    summary: str
    reasons: tuple[str, ...]
    supporting_observations: tuple[dict[str, Any], ...]
    supporting_evidence: Optional[dict[str, Any]]
    missing_conditions: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]
    warnings: tuple[str, ...]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "setup_id": self.setup_id,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "supporting_observations": list(self.supporting_observations),
            "supporting_evidence": self.supporting_evidence,
            "missing_conditions": list(self.missing_conditions),
            "invalidation_conditions": list(self.invalidation_conditions),
            "warnings": list(self.warnings),
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }
