"""Part A — validation hardening.

M6 only *documented* the multiple-testing risk (`MULTIPLE_TESTING_CAVEAT`)
and never actually corrected for it. This module applies a real,
standard correction (Benjamini-Hochberg FDR control) plus two gates
that were previously just caveats in M6's own mission text:

1. **Multiple-testing correction** (`one_sided_binomial_p_value` +
   `benjamini_hochberg_adjusted`): every evaluation's `P(win)` is
   tested against a null win rate (default 0.5 — "no edge") via an
   exact one-sided binomial test, then Benjamini-Hochberg controls the
   false-discovery rate across the whole batch being ranked together.
   No `scipy` dependency — computed via `math.comb`, same stdlib-only
   discipline M5's Wilson interval already established.
2. **Optimization-leakage guard** (`check_optimization_leakage`): if a
   strategy's parameters came from an M3 sweep, the evaluation period
   must not overlap the sweep's own selection period — otherwise the
   "out-of-sample" evaluation was computed on data the parameters were
   already picked to fit.
3. **Robustness gate** (`robustness_gate`): an M3 `RobustnessResult`
   labeled `"isolated"` or `"insufficient_data"` disqualifies that
   combination from ranking — a result that only looks good at one
   isolated parameter point is exactly what M3's own robustness
   analysis exists to catch (`docs/M3_RESEARCH_EXPERIMENTS.md`).

None of these recompute M3/M5/M6 statistics — they consume
`StrategyContextEvaluation.probability.counts` (M5/M6) and an M3
`RobustnessResult`-shaped summary dict (the same shape
`app.evaluation.evaluator.summarize_robustness_result` already
produces), unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

DISQUALIFYING_STABILITY_LABELS = frozenset({"isolated", "insufficient_data"})


def one_sided_binomial_p_value(successes: int, n: int, null_p: float = 0.5) -> float:
    """Exact one-sided binomial test p-value: `P(X >= successes)` under
    `X ~ Binomial(n, null_p)`. "One-sided" because the question being
    asked is specifically "does this beat the null win rate", not "does
    it differ from it in either direction" - a strategy performing
    suspiciously WORSE than the null is not the failure mode multiple-
    testing correction exists to guard against here.

    Computed via `math.comb` (stdlib, exact) rather than a normal
    approximation - correct at the small sample sizes this project's
    evidence tiers already flag as LIMITED/INSUFFICIENT, where a normal
    approximation would be least trustworthy.
    """
    if not (0 < null_p < 1):
        raise ValueError("null_p must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0 <= successes <= n):
        raise ValueError("successes must be between 0 and n")

    return sum(math.comb(n, k) * (null_p**k) * ((1 - null_p) ** (n - k)) for k in range(successes, n + 1))


def benjamini_hochberg_adjusted(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values ("q-values"), in the
    same order as `p_values`. Standard procedure: sort ascending,
    `q_(i) = min(1, min_{k=i}^{n} (n/k * p_(k)))`, computed via a
    running minimum from the largest rank down. A q-value `<= alpha`
    means the false-discovery rate is controlled at `alpha` across the
    WHOLE batch - not just for that one comparison in isolation, which
    is the entire point of correcting for multiple testing.
    """
    n = len(p_values)
    if n == 0:
        return []
    for p in p_values:
        if not (0 <= p <= 1):
            raise ValueError("every p-value must be between 0 and 1")

    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * n
    running_min = 1.0
    for rank in range(n - 1, -1, -1):
        k = rank + 1
        idx = order[rank]
        candidate = p_values[idx] * n / k
        running_min = min(running_min, candidate)
        adjusted_sorted[rank] = min(1.0, running_min)

    adjusted = [0.0] * n
    for rank in range(n):
        adjusted[order[rank]] = adjusted_sorted[rank]
    return adjusted


@dataclass(frozen=True)
class OptimizationProvenance:
    """Where a strategy's parameters came from - attached by the
    caller, NEVER inferred (M8 has no way to know, on its own, whether
    a given `StrategyContextEvaluation`'s strategy was hand-picked or
    swept). Leave every field `None` when parameters were not selected
    via any M3 sweep (e.g. a strategy's documented defaults) - absence
    of provenance means "not checkable", not "clean" (see
    `check_optimization_leakage`'s own limitation note).
    """

    source_experiment_id: Optional[str] = None
    selection_period_start: Optional[datetime] = None
    selection_period_end: Optional[datetime] = None


def check_optimization_leakage(
    provenance: OptimizationProvenance,
    evaluation_period_start: Optional[datetime],
    evaluation_period_end: Optional[datetime],
) -> tuple[bool, Optional[str]]:
    """Returns `(leaked, reason)`. `leaked=True` means the evaluation
    period overlaps the period the parameters were SELECTED from - that
    evaluation is not genuinely out-of-sample and must not be presented
    as validating those parameters.

    LIMITATION, stated plainly: when `provenance` carries no selection-
    period dates (the default `OptimizationProvenance()`), this returns
    `(False, None)` - "not flagged" - because there is nothing to check
    against, NOT because leakage has been ruled out. A caller ranking
    strategies whose parameter provenance isn't tracked gets no leakage
    protection from this function; that gap is closed only by actually
    supplying provenance, not by this function guessing.
    """
    if provenance.selection_period_start is None or provenance.selection_period_end is None:
        return False, None
    if evaluation_period_start is None or evaluation_period_end is None:
        return False, None

    overlaps = evaluation_period_start < provenance.selection_period_end and provenance.selection_period_start < evaluation_period_end
    if not overlaps:
        return False, None

    return True, (
        f"Evaluation period [{evaluation_period_start.isoformat()}, {evaluation_period_end.isoformat()}) overlaps "
        f"the parameter-selection period [{provenance.selection_period_start.isoformat()}, "
        f"{provenance.selection_period_end.isoformat()}) - this evaluation is not genuinely out-of-sample."
    )


def robustness_gate(robustness_summary: Optional[dict]) -> tuple[bool, Optional[str]]:
    """Returns `(eligible, reason_if_not)`. `robustness_summary` is the
    same dict shape `app.evaluation.evaluator.summarize_robustness_result`
    produces from an M3 `RobustnessResult` - this function does not
    recompute or re-run any robustness analysis, it only reads
    `stability_label`.

    `robustness_summary=None` (no M3 robustness analysis was attached
    at all) returns `(True, None)` - eligible - NOT because the
    combination is known robust, but because there is nothing to gate
    on. This is the same "absence is not evidence of cleanliness"
    caveat as `check_optimization_leakage` above; a ranking that omits
    robustness summaries entirely gets no protection from this gate.
    """
    if robustness_summary is None:
        return True, None
    label = robustness_summary.get("stability_label")
    if label in DISQUALIFYING_STABILITY_LABELS:
        return False, f"M3 robustness analysis labeled this combination's parameter sensitivity as {label!r}."
    return True, None
