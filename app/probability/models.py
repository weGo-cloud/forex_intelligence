"""M5 domain models.

`TradeOutcome` is a NEW, deliberately smaller model - not a reuse of
`app.backtesting.results.Trade` despite the conceptual overlap (both
describe "a trade that happened"). `Trade` carries fields that make
sense only inside a full, position-sized, equity-tracked backtest run
(`position_size`, `gross_pnl`, `net_pnl`, transaction-cost application)
- M5's labeler evaluates signals in isolation, with no equity curve or
position sizing at all (that is explicitly M2/M3's job, not M5's - see
the M5 mission's hard scope boundary). Reusing `Trade` would mean
either leaving those fields meaninglessly unset or inventing fake
values for them, neither of which is better than a purpose-built,
honestly-smaller model. What IS reused, because the underlying concept
is genuinely identical: `Direction` (`app.backtesting.results`) and
`RegimeContext` (`app.regimes.context`, embedded as-is, never
reimplemented - per the mission's explicit instruction).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.backtesting.results import Direction
from app.regimes.context import RegimeContext


class ExitReason(str, enum.Enum):
    """The MECHANICAL cause a labeled outcome concluded - finer-grained
    than `OutcomeLabel` (see that enum's own docstring for how the two
    relate)."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_EXPIRY = "time_expiry"  # neither barrier hit within the evaluation horizon
    SAME_BAR_UNRESOLVED = "same_bar_unresolved"  # both barriers touched on one bar; collision policy = "unresolved"
    INSUFFICIENT_DATA = "insufficient_data"  # ran out of real candle data before a horizon's worth could be evaluated


class OutcomeLabel(str, enum.Enum):
    """The STATISTICAL category every `TradeOutcome` is bucketed into -
    the four the M5 mission asks for, no more. `UNRESOLVED` covers BOTH
    `ExitReason.SAME_BAR_UNRESOLVED` and `ExitReason.INSUFFICIENT_DATA`:
    both mean "we cannot honestly assign a win/loss/breakeven verdict",
    which is the only distinction that matters for probability
    estimation (see `app.probability.estimators`) - the *reason* is
    still recorded on `TradeOutcome.exit_reason` for anyone who wants
    to distinguish them.
    """

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class TradeOutcome:
    """One labeled historical outcome: a signal, evaluated forward
    through a triple-barrier process, with enough context to condition
    probability queries on (see `app.probability.filters.OutcomeFilter`).

    `exit_timestamp` is ALWAYS set once labeling concludes - even for
    `UNRESOLVED` outcomes (an ambiguous same-bar collision, or running
    out of real data) - because "we now know this outcome is
    unresolvable" is itself information that became available at a
    specific point in time. This is what makes uniform causal filtering
    possible: `outcome.exit_timestamp <= T` is the ONE rule for "was
    this outcome knowable as of evaluation time T", regardless of which
    `ExitReason` produced it (see `app.probability.outcomes.OutcomeDataset.resolved_before`
    and M5 mission section 21).

    `entry_timestamp`/`entry_price` are the only fields that can be
    `None` - the one case where there was no bar to enter on at all
    (the signal fired on the very last bar of available data).
    `exit_price`/`r_multiple` are `None` whenever the outcome is
    `UNRESOLVED` (we know THAT it's unresolved, never WHAT price it
    would have resolved at - inventing one would be exactly the kind
    of fabricated precision M5 exists to avoid).
    """

    strategy_name: str
    instrument_symbol: str
    timeframe: str
    direction: Direction

    signal_timestamp: datetime
    entry_timestamp: Optional[datetime]
    exit_timestamp: datetime

    entry_price: Optional[float]
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float

    r_multiple: Optional[float]
    outcome_label: OutcomeLabel
    exit_reason: ExitReason

    evaluation_horizon_bars: int
    bars_to_exit: Optional[int]

    regime_context: Optional[RegimeContext] = None
    signal_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation - enums to `.value`, timestamps to
        ISO-8601, `regime_context` via its own `to_dict()` (never
        hand-rolled here, so the two stay in sync automatically)."""
        return {
            "strategy_name": self.strategy_name,
            "instrument_symbol": self.instrument_symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "signal_timestamp": self.signal_timestamp.isoformat(),
            "entry_timestamp": self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            "exit_timestamp": self.exit_timestamp.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "r_multiple": self.r_multiple,
            "outcome_label": self.outcome_label.value,
            "exit_reason": self.exit_reason.value,
            "evaluation_horizon_bars": self.evaluation_horizon_bars,
            "bars_to_exit": self.bars_to_exit,
            "regime_context": self.regime_context.to_dict() if self.regime_context is not None else None,
            "signal_meta": dict(self.signal_meta),
        }
