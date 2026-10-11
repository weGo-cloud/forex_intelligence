"""Part B — FIAS (Forex Intelligence Aggregate Score).

NOT one opaque number produced by a hidden formula. `FIASScore` always
carries its `components` (`FIASComponents`) alongside the composite -
every sub-score that went into it stays visible, same discipline as
M6's `EvidenceTier` never hiding the raw `SampleCounts` behind itself.

Four components, each already bounded to `[0, 1]` by construction (no
arbitrary normalization constant needed for any of them):

| Component | Formula | Source |
|---|---|---|
| `sample_adequacy` | `{INSUFFICIENT: 0.0, LIMITED: 0.5, ADEQUATE: 1.0}[evidence_tier]` | M6 `EvidenceTier` |
| `win_rate_confidence` | `1 - (win_interval.upper_bound - win_interval.lower_bound)` | M5 Wilson interval (via M6) - a narrower interval scores higher |
| `stability` | `windows_with_positive_expected_r / number_of_windows` | M6 `StabilityMetrics`, only if walk-forward was run |
| `robustness` | `{"stable": 1.0, "sensitive": 0.5, "isolated": 0.0, "insufficient_data": 0.0}[stability_label]` | M3 `RobustnessResult`, only if attached |

The composite is a weighted mean over ONLY the components that could
actually be computed - a missing component (no walk-forward run, no
robustness analysis attached) is excluded from the average, never
silently treated as `0` or backfilled with a guess (the same "never
invent a missing value" rule M4/M5/M6 all follow). If NOTHING could be
computed, the score is withheld (`None`) with a reason, not defaulted
to `0`.

FIAS is also withheld outright below `FIASConfig.minimum_tier_for_scoring`
(default: `EvidenceTier.LIMITED`) - a thin or nonexistent sample gets no
composite score at all, regardless of how favorable its raw statistics
look, matching the mission's explicit "do not fabricate precision" rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.evaluation.models import EvidenceTier, StrategyContextEvaluation

_TIER_RANK: dict[EvidenceTier, int] = {
    EvidenceTier.INSUFFICIENT: 0,
    EvidenceTier.LIMITED: 1,
    EvidenceTier.ADEQUATE: 2,
}
_SAMPLE_ADEQUACY_BY_TIER: dict[EvidenceTier, float] = {
    EvidenceTier.INSUFFICIENT: 0.0,
    EvidenceTier.LIMITED: 0.5,
    EvidenceTier.ADEQUATE: 1.0,
}
_ROBUSTNESS_BY_LABEL: dict[str, float] = {
    "stable": 1.0,
    "sensitive": 0.5,
    "isolated": 0.0,
    "insufficient_data": 0.0,
}


@dataclass(frozen=True)
class FIASConfig:
    weight_sample_adequacy: float = 1.0
    weight_win_rate_confidence: float = 1.0
    weight_stability: float = 1.0
    weight_robustness: float = 1.0
    minimum_tier_for_scoring: EvidenceTier = EvidenceTier.LIMITED

    def __post_init__(self) -> None:
        weights = (self.weight_sample_adequacy, self.weight_win_rate_confidence, self.weight_stability, self.weight_robustness)
        if any(w < 0 for w in weights):
            raise ValueError("all component weights must be >= 0")
        if all(w == 0 for w in weights):
            raise ValueError("at least one component weight must be > 0")


@dataclass(frozen=True)
class FIASComponents:
    sample_adequacy: Optional[float]
    win_rate_confidence: Optional[float]
    stability: Optional[float]
    robustness: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_adequacy": self.sample_adequacy,
            "win_rate_confidence": self.win_rate_confidence,
            "stability": self.stability,
            "robustness": self.robustness,
        }


@dataclass(frozen=True)
class FIASScore:
    score: Optional[float]
    components: FIASComponents
    withheld_reason: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "components": self.components.to_dict(), "withheld_reason": self.withheld_reason}


def compute_fias(
    evaluation: StrategyContextEvaluation,
    config: Optional[FIASConfig] = None,
    robustness_summary: Optional[dict] = None,
) -> FIASScore:
    config = config or FIASConfig()

    if _TIER_RANK[evaluation.evidence_tier] < _TIER_RANK[config.minimum_tier_for_scoring]:
        return FIASScore(
            score=None,
            components=FIASComponents(None, None, None, None),
            withheld_reason=(
                f"Evidence tier ({evaluation.evidence_tier.value}) is below the minimum required "
                f"for FIAS scoring ({config.minimum_tier_for_scoring.value})."
            ),
        )

    sample_adequacy = _SAMPLE_ADEQUACY_BY_TIER[evaluation.evidence_tier]

    win_rate_confidence = None
    interval = evaluation.probability.win_interval
    if interval is not None:
        width = interval.upper_bound - interval.lower_bound
        win_rate_confidence = max(0.0, 1.0 - width)

    stability = None
    if evaluation.walk_forward is not None and evaluation.walk_forward.stability.number_of_windows > 0:
        s = evaluation.walk_forward.stability
        stability = s.windows_with_positive_expected_r / s.number_of_windows

    robustness = None
    if robustness_summary is not None:
        robustness = _ROBUSTNESS_BY_LABEL.get(robustness_summary.get("stability_label"))

    components = FIASComponents(sample_adequacy, win_rate_confidence, stability, robustness)

    weighted_terms: list[tuple[float, float]] = []
    if sample_adequacy is not None:
        weighted_terms.append((sample_adequacy, config.weight_sample_adequacy))
    if win_rate_confidence is not None:
        weighted_terms.append((win_rate_confidence, config.weight_win_rate_confidence))
    if stability is not None:
        weighted_terms.append((stability, config.weight_stability))
    if robustness is not None:
        weighted_terms.append((robustness, config.weight_robustness))

    weighted_terms = [(v, w) for v, w in weighted_terms if w > 0]
    if not weighted_terms:
        return FIASScore(
            score=None, components=components,
            withheld_reason="No FIAS component with positive weight could be computed for this evaluation.",
        )

    total_weight = sum(w for _, w in weighted_terms)
    score = sum(v * w for v, w in weighted_terms) / total_weight

    return FIASScore(score=score, components=components, withheld_reason=None)
