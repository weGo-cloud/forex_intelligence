"""Descriptive comparison of two or more `StrategyContextEvaluation`s -
raw evidence side by side, NEVER a ranking or a winner (mission section
16: "Do not return Winner = Strategy A. Do not rank strategies. Do not
assign scores. M8 owns ranking."). Rows are returned in exactly the
order the caller supplied the evaluations - never re-sorted by a
metric, mirroring `app.research.comparison.compare_experiments`'s (M3)
identical design decision for the identical reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.evaluation.models import EvidenceTier, StrategyContextEvaluation

MULTIPLE_TESTING_CAVEAT = (
    "This report evaluates multiple strategy/context combinations. Comparing many combinations and "
    "highlighting whichever performed best introduces selection bias - a combination that looks strong "
    "here has not been independently validated merely for having been the best among those tested. "
    "See docs/M6_STRATEGY_REGIME_EVALUATION.md for the full caveat."
)


@dataclass(frozen=True)
class ComparisonRow:
    label: str
    strategy_name: Optional[str]
    instrument_symbol: Optional[str]
    timeframe: Optional[str]
    direction: Optional[str]
    resolved_observations: int
    total_observations: int
    p_win: Optional[float]
    win_interval_lower: Optional[float]
    win_interval_upper: Optional[float]
    expected_r: Optional[float]
    evidence_tier: EvidenceTier

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "strategy_name": self.strategy_name,
            "instrument_symbol": self.instrument_symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "resolved_observations": self.resolved_observations,
            "total_observations": self.total_observations,
            "p_win": self.p_win,
            "win_interval_lower": self.win_interval_lower,
            "win_interval_upper": self.win_interval_upper,
            "expected_r": self.expected_r,
            "evidence_tier": self.evidence_tier.value,
        }


def default_label(evaluation: StrategyContextEvaluation, index: int) -> str:
    parts = [
        evaluation.strategy_name,
        evaluation.instrument_symbol,
        evaluation.timeframe,
        evaluation.direction,
    ]
    label = " / ".join(p for p in parts if p is not None)
    return label or f"evaluation_{index}"


_default_label = default_label  # backward-compatible alias; default_label is the public name


def compare_evaluations(
    evaluations: list[StrategyContextEvaluation], labels: Optional[list[str]] = None
) -> list[ComparisonRow]:
    """`labels`, if given, must be the same length as `evaluations` -
    otherwise a label is derived from each evaluation's own
    strategy/instrument/timeframe/direction."""
    if labels is not None and len(labels) != len(evaluations):
        raise ValueError("labels must be the same length as evaluations")

    rows: list[ComparisonRow] = []
    for i, evaluation in enumerate(evaluations):
        label = labels[i] if labels is not None else default_label(evaluation, i)
        interval = evaluation.probability.win_interval
        rows.append(
            ComparisonRow(
                label=label,
                strategy_name=evaluation.strategy_name,
                instrument_symbol=evaluation.instrument_symbol,
                timeframe=evaluation.timeframe,
                direction=evaluation.direction,
                resolved_observations=evaluation.probability.counts.resolved_observations,
                total_observations=evaluation.probability.counts.total_observations,
                p_win=evaluation.probability.p_win,
                win_interval_lower=interval.lower_bound if interval else None,
                win_interval_upper=interval.upper_bound if interval else None,
                expected_r=evaluation.probability.expected_r,
                evidence_tier=evaluation.evidence_tier,
            )
        )
    return rows
