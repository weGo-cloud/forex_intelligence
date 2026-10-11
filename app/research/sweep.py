"""Generic, strategy-agnostic parameter-sweep engine.

Takes a base experiment definition and a parameter grid (a dict of
`strategy_param_name -> list of values to try`), and runs one child
SINGLE_BACKTEST experiment per combination, linked to a PARAMETER_SWEEP
parent experiment via `parent_experiment_id`.

No strategy parameter name (e.g. "risk_reward", "opening_range_minutes")
is ever referenced here - the grid is opaque key/value pairs merged into
`base_definition.strategy_params`. This is what keeps the sweep engine
usable for any current or future strategy without modification.

Synchronous by design (M3 spec section 8: "do not prematurely add
distributed execution... design the interface so parallel execution can
be added later without changing experiment semantics"). Each child run
is fully independent (its own ExperimentDefinition, its own hash, its
own registry row), which is exactly the property that would let a
future executor run them concurrently without any change to this
module's public contract.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.research.experiment import ExperimentOutcome, execute_experiment
from app.research.models import ExperimentDefinition, ExperimentType
from app.research.registry import ExperimentRegistry


@dataclass(frozen=True)
class SweepChildResult:
    parameter_combination: dict[str, Any]
    outcome: ExperimentOutcome


@dataclass(frozen=True)
class SweepResult:
    parent_experiment_id: str
    children: list[SweepChildResult]


def _combinations(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid, in deterministic order (grid keys
    sorted alphabetically; each key's own value list keeps the caller's
    order), with exact-duplicate combinations removed. A combination is
    a duplicate only if every parameter value is identical - this can
    legitimately happen if the caller passes a duplicate value within a
    single parameter's list (e.g. [2.0, 2.0, 2.5])."""
    if not parameter_grid:
        return [{}]

    keys = sorted(parameter_grid.keys())
    value_lists = [parameter_grid[k] for k in keys]

    seen: set[tuple[Any, ...]] = set()
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        key_tuple = tuple(values)
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        combos.append(dict(zip(keys, values)))
    return combos


def run_parameter_sweep(
    session: Session,
    registry: ExperimentRegistry,
    base_definition: ExperimentDefinition,
    parameter_grid: dict[str, Any],
    extra_summary: dict[str, Any] | None = None,
) -> SweepResult:
    """Run one child experiment per combination in `parameter_grid`.

    `base_definition` is never mutated (definitions are frozen
    dataclasses; children are produced via `with_overrides`). The
    parent experiment record stores the base configuration and the
    grid itself in its result summary once all children complete, so
    the sweep as a whole remains traceable to a single experiment_id.

    `extra_summary`, if given, is merged into the parent's result
    summary before it is marked COMPLETED (rather than requiring a
    second write afterwards, which the registry's immutable-once-
    completed rule would reject) - this is how
    `app.research.robustness` attaches its stability classification to
    the same completion event as the sweep itself, instead of the
    sweep and the robustness label being two separate persisted facts.
    """
    parent_definition = base_definition.with_overrides(experiment_type=ExperimentType.PARAMETER_SWEEP)
    parent_record = registry.create(parent_definition)
    registry.mark_running(parent_record.experiment_id)

    combinations = _combinations(parameter_grid)
    children: list[SweepChildResult] = []

    for combo in combinations:
        child_strategy_params = {**base_definition.strategy_params, **combo}
        child_definition = base_definition.with_overrides(
            strategy_params=child_strategy_params,
            experiment_type=ExperimentType.SINGLE_BACKTEST,
            parent_experiment_id=parent_record.experiment_id,
            tags={**base_definition.tags, "sweep_parent": parent_record.experiment_id},
        )
        outcome = execute_experiment(session, registry, child_definition)
        children.append(SweepChildResult(parameter_combination=combo, outcome=outcome))

    completed_count = sum(1 for c in children if c.outcome.result is not None)
    registry.mark_completed(
        parent_record.experiment_id,
        {
            "parameter_grid": {k: list(v) for k, v in parameter_grid.items()},
            "combination_count": len(combinations),
            "completed_children": completed_count,
            "failed_children": len(combinations) - completed_count,
            "child_experiment_ids": [c.outcome.record.experiment_id for c in children],
            **(extra_summary or {}),
        },
    )

    return SweepResult(parent_experiment_id=parent_record.experiment_id, children=children)
