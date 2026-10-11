"""Walk-forward as a persisted, provenance-tracked experiment.

`app.backtesting.validation.walk_forward_windows` already does the pure
date-arithmetic of carving a range into (train, test) windows, and
`run_walk_forward` already runs+logs each window to the M2 JSONL log.
Neither is duplicated here. What this module adds is the M3-specific
part: representing each window as its own persisted `ExperimentRecord`
(SINGLE_BACKTEST, whose test period *is* the window's test period) that
is a child of one WALK_FORWARD parent experiment - so a walk-forward
run is queryable, comparable, and immutable-once-completed exactly like
any other experiment, with `GET /experiments/{parent}/children`
returning the per-window results in order.

Train boundaries are preserved on the parent's result summary (section
9: "for each window preserve... train period"), even though this
minimal version does not run any parameter optimization on the train
window - see module docstring in validation.py for why that's
deliberate. No future information leaks: each child's test period is
strictly the window's test range, computed once up front from the
window boundaries, not derived from anything computed during another
window's run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.backtesting.validation import WalkForwardWindow, walk_forward_windows
from app.research.experiment import ExperimentOutcome, execute_experiment
from app.research.models import ExperimentDefinition, ExperimentType
from app.research.registry import ExperimentRegistry


@dataclass(frozen=True)
class WalkForwardChildResult:
    window: WalkForwardWindow
    outcome: ExperimentOutcome


@dataclass(frozen=True)
class WalkForwardResult:
    parent_experiment_id: str
    children: list[WalkForwardChildResult]


def run_walk_forward_experiment(
    session: Session,
    registry: ExperimentRegistry,
    base_definition: ExperimentDefinition,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
) -> WalkForwardResult:
    """`base_definition.test_period_start/end` define the OVERALL range
    to carve into windows; each child's own test period is then set to
    that window's test_start/test_end (never the overall range) - the
    one invariant that must hold for this to mean anything as an
    out-of-sample check.
    """
    windows = walk_forward_windows(
        base_definition.test_period_start, base_definition.test_period_end, train_days, test_days, step_days
    )
    if not windows:
        raise ValueError(
            "No walk-forward windows fit in the given test period with these "
            f"train_days={train_days}, test_days={test_days}, step_days={step_days}"
        )

    parent_definition = base_definition.with_overrides(experiment_type=ExperimentType.WALK_FORWARD)
    parent_record = registry.create(parent_definition)
    registry.mark_running(parent_record.experiment_id)

    children: list[WalkForwardChildResult] = []
    for window in windows:
        child_definition = base_definition.with_overrides(
            experiment_type=ExperimentType.SINGLE_BACKTEST,
            test_period_start=window.test_start,
            test_period_end=window.test_end,
            parent_experiment_id=parent_record.experiment_id,
            tags={
                **base_definition.tags,
                "walk_forward_parent": parent_record.experiment_id,
                "window_index": str(window.window_index),
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
            },
        )
        outcome = execute_experiment(session, registry, child_definition)
        children.append(WalkForwardChildResult(window=window, outcome=outcome))

    completed_count = sum(1 for c in children if c.outcome.result is not None)
    registry.mark_completed(
        parent_record.experiment_id,
        {
            "window_count": len(windows),
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days if step_days is not None else test_days,
            "completed_children": completed_count,
            "failed_children": len(windows) - completed_count,
            "windows": [
                {
                    "window_index": c.window.window_index,
                    "train_start": c.window.train_start.isoformat(),
                    "train_end": c.window.train_end.isoformat(),
                    "test_start": c.window.test_start.isoformat(),
                    "test_end": c.window.test_end.isoformat(),
                    "child_experiment_id": c.outcome.record.experiment_id,
                }
                for c in children
            ],
        },
    )

    return WalkForwardResult(parent_experiment_id=parent_record.experiment_id, children=children)
