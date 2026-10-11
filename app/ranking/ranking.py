"""Part C — ranking.

Ranks WITHIN a single, caller-supplied batch (mirrors M6's
`evaluate_combinations` — never an automatic search, never a claim
about "the best strategy in the system"). Unlike M6's
`compare_evaluations` (which deliberately preserves caller order
because a side-by-side comparison should not silently editorialize),
`rank_evaluations` DOES reorder its output — establishing an order is
the entire point of a ranking function — but every entry still carries
its full `FIASScore` (with components), corrected significance
(`p_value`/`q_value`), and `eligible`/`exclusion_reasons`, so nothing
about how the order was reached is hidden.

An evaluation is EXCLUDED from ranking (gets `rank=None`, placed after
every ranked entry) when:
- `evidence_tier == INSUFFICIENT`, or
- the robustness gate disqualifies it (Part A), or
- optimization-leakage is detected (Part A), or
- `compute_fias` withheld a score for it (Part B).

Excluded entries are NEVER deleted from the output — they still carry
their full evaluation, FIAS components (if any), and
`exclusion_reasons`, per this project's consistent "never hide a
result, only label it" rule (identical in spirit to M6's `EvidenceTier`
and M5's "probability = unavailable rather than a fallback").
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from app.evaluation.comparison import MULTIPLE_TESTING_CAVEAT, default_label
from app.evaluation.models import EvidenceTier, StrategyContextEvaluation
from app.ranking.fias import FIASConfig, FIASScore, compute_fias
from app.ranking.validation import (
    OptimizationProvenance,
    benjamini_hochberg_adjusted,
    check_optimization_leakage,
    one_sided_binomial_p_value,
    robustness_gate,
)

RANKING_CAVEAT = (
    MULTIPLE_TESTING_CAVEAT
    + " Unlike a plain side-by-side comparison, this ranking applies a Benjamini-Hochberg false-discovery-rate "
    "correction across the batch (see p_value/q_value/significant on each entry) - but correction reduces false "
    "positives, it does not eliminate the risk, and it says nothing about whether the ranked order will hold in "
    "the future."
)


@dataclass(frozen=True)
class RankedEvaluation:
    rank: Optional[int]
    label: str
    evaluation: StrategyContextEvaluation
    fias: FIASScore
    p_value: Optional[float]
    q_value: Optional[float]
    significant: Optional[bool]
    eligible: bool
    exclusion_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "label": self.label,
            "evaluation": self.evaluation.to_dict(),
            "fias": self.fias.to_dict(),
            "p_value": self.p_value,
            "q_value": self.q_value,
            "significant": self.significant,
            "eligible": self.eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
        }


def rank_evaluations(
    evaluations: list[StrategyContextEvaluation],
    labels: Optional[list[str]] = None,
    robustness_summaries: Optional[list[Optional[dict]]] = None,
    provenances: Optional[list[Optional[OptimizationProvenance]]] = None,
    fias_config: Optional[FIASConfig] = None,
    null_p: float = 0.5,
    alpha: float = 0.05,
) -> list[RankedEvaluation]:
    """Rank a caller-supplied batch of `StrategyContextEvaluation`s.

    `robustness_summaries`/`provenances`, if given, must be the same
    length as `evaluations` (use `None` entries for evaluations with no
    attached robustness analysis / no known parameter provenance -
    both gates degrade to "not checkable" per their own docstrings in
    `app.ranking.validation`, never to "assumed clean").
    """
    n = len(evaluations)
    if labels is not None and len(labels) != n:
        raise ValueError("labels must be the same length as evaluations")
    if robustness_summaries is not None and len(robustness_summaries) != n:
        raise ValueError("robustness_summaries must be the same length as evaluations")
    if provenances is not None and len(provenances) != n:
        raise ValueError("provenances must be the same length as evaluations")

    labels = labels or [default_label(e, i) for i, e in enumerate(evaluations)]
    robustness_summaries = robustness_summaries or [None] * n
    provenances = provenances or [None] * n
    fias_config = fias_config or FIASConfig()

    p_values: list[Optional[float]] = []
    for evaluation in evaluations:
        counts = evaluation.probability.counts
        if counts.resolved_observations == 0:
            p_values.append(None)
        else:
            p_values.append(one_sided_binomial_p_value(counts.wins, counts.resolved_observations, null_p))

    indices_with_p = [i for i, p in enumerate(p_values) if p is not None]
    q_values: list[Optional[float]] = [None] * n
    if indices_with_p:
        adjusted = benjamini_hochberg_adjusted([p_values[i] for i in indices_with_p])
        for idx, q in zip(indices_with_p, adjusted):
            q_values[idx] = q

    entries: list[RankedEvaluation] = []
    for i, evaluation in enumerate(evaluations):
        exclusion_reasons: list[str] = []
        eligible = True

        if evaluation.evidence_tier == EvidenceTier.INSUFFICIENT:
            eligible = False
            exclusion_reasons.append("Evidence tier is INSUFFICIENT.")

        robust_ok, robust_reason = robustness_gate(robustness_summaries[i])
        if not robust_ok:
            eligible = False
            exclusion_reasons.append(robust_reason)

        if provenances[i] is not None:
            leaked, leak_reason = check_optimization_leakage(
                provenances[i], evaluation.evaluation_period_start, evaluation.evaluation_period_end
            )
            if leaked:
                eligible = False
                exclusion_reasons.append(leak_reason)

        fias = compute_fias(evaluation, fias_config, robustness_summaries[i])
        if fias.score is None:
            eligible = False
            exclusion_reasons.append(fias.withheld_reason or "FIAS score unavailable.")

        significant = q_values[i] <= alpha if q_values[i] is not None else None

        entries.append(
            RankedEvaluation(
                rank=None,
                label=labels[i],
                evaluation=evaluation,
                fias=fias,
                p_value=p_values[i],
                q_value=q_values[i],
                significant=significant,
                eligible=eligible,
                exclusion_reasons=tuple(exclusion_reasons),
            )
        )

    rankable = [e for e in entries if e.eligible and e.fias.score is not None]
    unrankable = [e for e in entries if not (e.eligible and e.fias.score is not None)]
    rankable.sort(key=lambda e: (-e.fias.score, e.label))

    result = [dataclasses.replace(entry, rank=rank) for rank, entry in enumerate(rankable, start=1)]
    result.extend(unrankable)
    return result
