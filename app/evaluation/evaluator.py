"""The M6 evaluator: composes M5's `ProbabilityEstimator` + M3's
walk-forward window arithmetic + M4's `RegimeContext` (already embedded
in every `TradeOutcome` by M5) into a `StrategyContextEvaluation`.

TEMPORAL HONESTY (mission section 13, the mandatory requirement): a
walk-forward window's evaluation uses ONLY outcomes whose
`signal_timestamp` falls inside `[test_start, test_end)` AND whose
`exit_timestamp < test_end` - both conditions together, not just one.
`signal_timestamp` scoping keeps a window's population to signals that
actually fired during that window; `exit_timestamp` scoping (via
`OutcomeDataset.resolved_before`, M5's own causal mechanism, reused
unchanged) additionally guarantees nothing whose fate was still unknown
at `test_end` sneaks in even if it happened to fire inside the window
(e.g. a signal on the window's last bar with a long evaluation horizon
that resolves well after `test_end`). Neither condition alone is
sufficient; both are mandatory and both are tested directly (see
`tests/evaluation/test_causality.py`).

No second walk-forward implementation exists here - window boundaries
come from `app.backtesting.validation.walk_forward_windows`, the exact
function `app.research.runner.run_walk_forward_experiment` (M3) already
uses (mission section 10: "M3 already provides walk-forward
infrastructure. Reuse it. Do not create a second walk-forward
implementation.").
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from app.backtesting.validation import walk_forward_windows
from app.evaluation.models import (
    EvaluationConfig,
    EvidenceTier,
    StrategyContextEvaluation,
    WalkForwardEvaluation,
    WalkForwardWindowEvaluation,
)
from app.evaluation.stability import compute_stability_metrics
from app.probability.estimators import ProbabilityEstimator, SampleCounts
from app.probability.filters import OutcomeFilter
from app.probability.outcomes import OutcomeDataset


def summarize_robustness_result(robustness_result: Any) -> dict[str, Any]:
    """A thin, read-only summary of an M3 `RobustnessResult`
    (`app.research.robustness`) for attachment to a
    `StrategyContextEvaluation` - none of M3's parameter-neighborhood
    math is recomputed; this just lifts the fields a research report
    would want to show alongside M6 evidence. Takes the object
    structurally (duck-typed, no import of `app.research` at module
    level) so `app.evaluation` doesn't acquire a hard import-time
    dependency on `app.research` for callers who never use this
    function - the dependency rule (mission section 25) permits M6 to
    use `app.research`, but nothing requires every M6 caller to pull it
    in.
    """
    return {
        "parent_experiment_id": robustness_result.parent_experiment_id,
        "parameter_name": robustness_result.parameter_name,
        "values_tested": list(robustness_result.values_tested),
        "stability_label": robustness_result.stability_label,
        "notes": robustness_result.notes,
    }


class StrategyEvaluator:
    def __init__(self, config: Optional[EvaluationConfig] = None, estimator: Optional[ProbabilityEstimator] = None) -> None:
        self.config = config or EvaluationConfig()
        self.estimator = estimator or ProbabilityEstimator(confidence_level=self.config.confidence_level)

    def _classify(self, counts: SampleCounts) -> tuple[EvidenceTier, list[str]]:
        warnings: list[str] = []
        n = counts.resolved_observations

        if counts.total_observations == 0:
            warnings.append("No observations matched this filter at all.")
        if n < self.config.limited_resolved_observations:
            tier = EvidenceTier.INSUFFICIENT
            warnings.append(
                f"Only {n} resolved observation(s) - below the LIMITED threshold "
                f"({self.config.limited_resolved_observations}). Treat this as exploratory, not evidence."
            )
        elif n < self.config.minimum_resolved_observations:
            tier = EvidenceTier.LIMITED
            warnings.append(
                f"{n} resolved observations - below the ADEQUATE threshold "
                f"({self.config.minimum_resolved_observations}). Interpret with caution; the interval is likely wide."
            )
        else:
            tier = EvidenceTier.ADEQUATE

        if counts.unresolved > 0:
            warnings.append(
                f"{counts.unresolved} outcome(s) were UNRESOLVED and excluded from win/loss/breakeven probabilities."
            )

        return tier, warnings

    def evaluate(
        self,
        dataset: OutcomeDataset,
        filter: Optional[OutcomeFilter] = None,
        as_of: Optional[datetime] = None,
        robustness_summary: Optional[dict[str, Any]] = None,
    ) -> StrategyContextEvaluation:
        """The baseline (non-windowed) evaluation: M5's
        `ProbabilityEstimator` applied to `dataset`, optionally
        restricted by `filter` and/or a causal `as_of` cutoff -
        identical semantics to `ProbabilityEstimator.estimate`, wrapped
        with an explicit sample-size verdict.
        """
        filter = filter or OutcomeFilter()
        estimate = self.estimator.estimate(dataset, filter=filter, as_of=as_of)
        tier, warnings = self._classify(estimate.counts)

        eligible = dataset.resolved_before(as_of) if as_of is not None else dataset
        eligible = filter.apply(eligible)
        signal_times = [o.signal_timestamp for o in eligible]
        period_start = min(signal_times) if signal_times else None
        period_end = max(signal_times) if signal_times else None

        return StrategyContextEvaluation(
            context_filter=filter,
            probability=estimate,
            evaluation_period_start=period_start,
            evaluation_period_end=period_end,
            as_of=as_of,
            evidence_tier=tier,
            warnings=tuple(warnings),
            robustness_summary=robustness_summary,
        )

    def evaluate_combinations(
        self,
        dataset: OutcomeDataset,
        filters: list[OutcomeFilter],
        as_of: Optional[datetime] = None,
    ) -> list[StrategyContextEvaluation]:
        """Evaluate exactly the filters the caller supplied - never an
        automatic search over every possible combination (mission
        section 15: "Do not automatically search every possible
        combination unless explicitly requested. This avoids
        combinatorial explosion."). The caller is responsible for
        noting `len(filters)` in any report as the number of tested
        configurations (mission section 18) -
        `app.evaluation.comparison.compare_evaluations` does this
        automatically when building a comparison report.
        """
        return [self.evaluate(dataset, f, as_of) for f in filters]

    def evaluate_walk_forward(
        self,
        dataset: OutcomeDataset,
        period_start: datetime,
        period_end: datetime,
        train_days: int,
        test_days: int,
        step_days: Optional[int] = None,
        filter: Optional[OutcomeFilter] = None,
    ) -> StrategyContextEvaluation:
        """A baseline evaluation over `[period_start, period_end)` (see
        `evaluate`'s semantics - `as_of` is `period_end` here, so the
        baseline number is exactly what the walk-forward windows
        collectively cover), plus a per-window temporal breakdown and
        stability metrics.

        If no windows fit `train_days`/`test_days`/`step_days` within
        the period, this does NOT raise (unlike
        `app.research.runner.run_walk_forward_experiment`, which does -
        a different design choice for a different context: M3's
        version drives actual backtest EXECUTION, where "nothing to run"
        is a caller error; M6's version drives a research REPORT, where
        the mission explicitly lists "no valid walk-forward windows" as
        an edge case to represent, not reject - see mission section 22).
        Instead, `walk_forward` is an empty `WalkForwardEvaluation` and
        a warning is added to the baseline evaluation's `warnings`.
        """
        filter = filter or OutcomeFilter()
        windows = walk_forward_windows(period_start, period_end, train_days, test_days, step_days)

        window_evals: list[WalkForwardWindowEvaluation] = []
        for window in windows:
            window_dataset = dataset.filter_by(
                lambda o, w=window: w.test_start <= o.signal_timestamp < w.test_end
            )
            window_dataset = window_dataset.resolved_before(window.test_end)
            window_dataset = filter.apply(window_dataset)
            window_estimate = self.estimator.estimate(window_dataset)
            window_evals.append(
                WalkForwardWindowEvaluation(
                    window_index=window.window_index,
                    train_start=window.train_start,
                    train_end=window.train_end,
                    test_start=window.test_start,
                    test_end=window.test_end,
                    probability=window_estimate,
                )
            )

        stability = compute_stability_metrics(window_evals)
        walk_forward = WalkForwardEvaluation(windows=tuple(window_evals), stability=stability)

        baseline = self.evaluate(dataset, filter=filter, as_of=period_end)
        extra_warnings = list(baseline.warnings)
        if not windows:
            extra_warnings.append(
                f"No walk-forward windows fit train_days={train_days}, test_days={test_days}, "
                f"step_days={step_days} within [{period_start.isoformat()}, {period_end.isoformat()})."
            )

        return StrategyContextEvaluation(
            context_filter=baseline.context_filter,
            probability=baseline.probability,
            evaluation_period_start=period_start,
            evaluation_period_end=period_end,
            as_of=period_end,
            evidence_tier=baseline.evidence_tier,
            warnings=tuple(extra_warnings),
            walk_forward=walk_forward,
            robustness_summary=baseline.robustness_summary,
        )
