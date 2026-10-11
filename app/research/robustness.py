"""Parameter-neighborhood robustness analysis.

Answers one narrow question: does a strategy's performance exist only
at one isolated parameter point, or does it hold up across nearby
values? (e.g. risk_reward=2.0 looking great but 1.75 and 2.25 both
losing money is a strong signal the 2.0 result is curve-fit noise
rather than a real edge.)

Built directly on `app.research.sweep` - a robustness analysis IS a
parameter sweep over a single parameter's neighborhood, with the
values supplied by the caller (never hard-coded here, per section 10:
"the caller supplies the neighborhood/grid").

Deliberately does NOT compute a single composite "robustness score".
Section 10 is explicit that an invented score is out of scope unless
mathematically justified and documented, and no such justification
exists yet - scoring/ranking strategies is explicitly the future
ranking engine's job (section 11: "the future ranking engine will own
strategy ranking. M3 owns research comparison."). What this module
does instead is expose the per-neighbor metrics plus a small set of
plainly-defined, non-scoring descriptive labels (see `classify_stability`)
that a human - or later, the ranking engine - can read without having
to trust a hidden formula.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.research.experiment import execute_experiment
from app.research.models import ExperimentDefinition, ExperimentType
from app.research.registry import ExperimentRegistry
from app.research.sweep import SweepChildResult

MIN_TRADES_FOR_STABILITY_JUDGEMENT = 10


@dataclass(frozen=True)
class RobustnessResult:
    parent_experiment_id: str
    parameter_name: str
    values_tested: list[Any]
    children: list[SweepChildResult]
    stability_label: str
    notes: str


def classify_stability(children: list[SweepChildResult]) -> tuple[str, str]:
    """A plain, documented classification - not a score.

    Rules, applied in order:
    1. "insufficient_data" - any neighbor has fewer than
       MIN_TRADES_FOR_STABILITY_JUDGEMENT trades (including failed
       runs), because a return computed from a handful of trades is not
       a basis for judging sensitivity either way.
    2. "isolated" - the return at every neighbor is not of the same
       sign as the majority of neighbors (i.e. profitability does not
       hold across the neighborhood at all, only at scattered points).
    3. "sensitive" - returns are same-signed but the standard deviation
       of returns across neighbors exceeds the mean absolute return
       (performance swings by more than its own average magnitude
       across small parameter changes).
    4. "stable" - otherwise: same-signed returns with variability no
       larger than their own average magnitude.
    """
    returns: list[float] = []
    for child in children:
        result = child.outcome.result
        if result is None or result.metrics.trade_count < MIN_TRADES_FOR_STABILITY_JUDGEMENT:
            return (
                "insufficient_data",
                f"At least one neighbor has fewer than {MIN_TRADES_FOR_STABILITY_JUDGEMENT} trades "
                "(or failed to run) - too few observations to judge sensitivity.",
            )
        returns.append(result.metrics.total_return_pct)

    if not returns:
        return "insufficient_data", "No neighbors were evaluated."

    positive = sum(1 for r in returns if r > 0)
    negative = sum(1 for r in returns if r < 0)
    majority_positive = positive >= negative
    same_sign_count = positive if majority_positive else negative
    if same_sign_count < len(returns):
        return (
            "isolated",
            f"Only {same_sign_count}/{len(returns)} neighbors share the majority sign of return - "
            "performance does not hold across the neighborhood.",
        )

    mean_abs_return = statistics.mean(abs(r) for r in returns)
    stdev_return = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    if mean_abs_return > 0 and stdev_return > mean_abs_return:
        return (
            "sensitive",
            f"Return standard deviation ({stdev_return:.2f}) exceeds mean absolute return "
            f"({mean_abs_return:.2f}) across the neighborhood.",
        )

    return (
        "stable",
        f"All {len(returns)} neighbors share the same return sign with standard deviation "
        f"({stdev_return:.2f}) no larger than the mean absolute return ({mean_abs_return:.2f}).",
    )


def run_robustness_analysis(
    session: Session,
    registry: ExperimentRegistry,
    base_definition: ExperimentDefinition,
    parameter_name: str,
    values: list[Any],
) -> RobustnessResult:
    if len(values) < 2:
        raise ValueError("Robustness analysis needs at least 2 neighborhood values to be meaningful")

    # classify_stability needs the executed children, but the sweep's own
    # mark_completed call happens inside run_parameter_sweep - so the
    # label/notes are computed by running the sweep's combination+execution
    # logic directly here and folding the classification into the single
    # completion write, rather than attempting a second write afterwards
    # (which the registry's immutability rule would reject).
    from app.research.sweep import _combinations  # internal helper reuse: same cartesian-product/dedup logic

    combinations = _combinations({parameter_name: values})
    parent_definition = base_definition.with_overrides(experiment_type=ExperimentType.PARAMETER_SWEEP)
    parent_record = registry.create(parent_definition)
    registry.mark_running(parent_record.experiment_id)

    children: list[SweepChildResult] = []
    for combo in combinations:
        child_strategy_params = {**base_definition.strategy_params, **combo}
        child_definition = base_definition.with_overrides(
            strategy_params=child_strategy_params,
            experiment_type=ExperimentType.SINGLE_BACKTEST,
            parent_experiment_id=parent_record.experiment_id,
            tags={**base_definition.tags, "robustness_parent": parent_record.experiment_id, "parameter_name": parameter_name},
        )
        outcome = execute_experiment(session, registry, child_definition)
        children.append(SweepChildResult(parameter_combination=combo, outcome=outcome))

    label, notes = classify_stability(children)
    completed_count = sum(1 for c in children if c.outcome.result is not None)
    registry.mark_completed(
        parent_record.experiment_id,
        {
            "parameter_grid": {parameter_name: list(values)},
            "combination_count": len(combinations),
            "completed_children": completed_count,
            "failed_children": len(combinations) - completed_count,
            "child_experiment_ids": [c.outcome.record.experiment_id for c in children],
            "parameter_name": parameter_name,
            "stability_label": label,
            "stability_notes": notes,
        },
    )

    return RobustnessResult(
        parent_experiment_id=parent_record.experiment_id,
        parameter_name=parameter_name,
        values_tested=values,
        children=children,
        stability_label=label,
        notes=notes,
    )
