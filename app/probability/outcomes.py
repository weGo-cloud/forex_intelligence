"""A reusable, composable collection of `TradeOutcome`s.

Prefers plain Python (a tuple + list comprehensions) over pandas for
storage and filtering - a `TradeOutcome` embeds a `RegimeContext`
(itself not pandas-friendly, being a frozen dataclass with nested enums
and a dict), so flattening every outcome into DataFrame columns would
either lose information or require re-deriving `RegimeContext` fields
as ad-hoc column names. `to_dataframe()` exists for the cases pandas
genuinely helps (bulk aggregation, CSV export) without making it the
primary representation - matching the M5 mission's "prefer pandas only
where it fits existing project conventions."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import pandas as pd

from app.probability.models import TradeOutcome


@dataclass(frozen=True)
class OutcomeDataset:
    outcomes: tuple[TradeOutcome, ...]

    def __iter__(self):
        return iter(self.outcomes)

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(self, index: int) -> TradeOutcome:
        return self.outcomes[index]

    @classmethod
    def from_list(cls, outcomes: list[TradeOutcome]) -> "OutcomeDataset":
        return cls(outcomes=tuple(outcomes))

    def filter_by(self, predicate: Callable[[TradeOutcome], bool]) -> "OutcomeDataset":
        """Generic escape hatch for a one-off predicate. For the
        standard conditioning dimensions (strategy/instrument/timeframe/
        direction/regime states/session), prefer
        `app.probability.filters.OutcomeFilter` + `.apply(self)` instead -
        it's composable without writing a new function per combination
        (M5 mission section 17)."""
        return OutcomeDataset(outcomes=tuple(o for o in self.outcomes if predicate(o)))

    def resolved_before(self, as_of: datetime) -> "OutcomeDataset":
        """Every outcome whose `exit_timestamp <= as_of` - i.e. every
        outcome that was ALREADY KNOWABLE (including "knowably
        unresolved") as of `as_of`. This is the one mechanism that
        makes a probability estimate walk-forward-safe (M5 mission
        section 21): pass the evaluation timestamp `T` here before
        handing the result to `ProbabilityEstimator`, and nothing that
        resolved after `T` can possibly be included. See
        `TradeOutcome.exit_timestamp`'s own docstring for why this
        single comparison is sufficient (uniform across every
        `ExitReason`, including `UNRESOLVED` ones).
        """
        return self.filter_by(lambda o: o.exit_timestamp <= as_of)

    def group_by(self, key: Callable[[TradeOutcome], Any]) -> dict[Any, "OutcomeDataset"]:
        groups: dict[Any, list[TradeOutcome]] = {}
        for outcome in self.outcomes:
            groups.setdefault(key(outcome), []).append(outcome)
        return {k: OutcomeDataset.from_list(v) for k, v in groups.items()}

    def to_list_of_dicts(self) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.outcomes]

    def to_dataframe(self) -> pd.DataFrame:
        """Flattened, JSON-safe tabular view - convenient for bulk
        aggregation/CSV export, not used internally by
        `ProbabilityEstimator` (which operates on the `TradeOutcome`
        objects directly)."""
        return pd.DataFrame(self.to_list_of_dicts())
