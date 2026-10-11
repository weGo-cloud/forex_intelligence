"""The strategy abstraction every strategy implements and the only
interface the backtesting engine depends on. The engine never imports
ORBStrategy or RSIDivergenceStrategy directly — it only knows about
`Strategy` and `Signal` — so adding a third strategy later means adding
one new module, not touching `app/backtesting/engine.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

import pandas as pd

Direction = Literal["long", "short"]


@dataclass(frozen=True)
class MarketContext:
    """Everything a strategy needs to generate signals for one run.

    `execution_df` is the timeframe the strategy actually enters/exits
    on. `context` holds any additional timeframes (keyed by their
    Timeframe.value, e.g. "1H", "15M") a strategy wants for
    higher-timeframe confirmation — already causally aligned onto
    `execution_df`'s index via `data_access.align_higher_timeframe`, so
    strategies never do their own timeframe joins (and can't
    accidentally introduce leakage by doing it wrong).
    """

    instrument_symbol: str
    execution_df: pd.DataFrame
    context: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True)
class Signal:
    """A single trade signal produced by a strategy.

    `signal_time` is the timestamp of the bar on which the triggering
    condition became true (e.g. the breakout bar's close, or a
    divergence pivot's confirmation bar). It is NOT the fill time — the
    engine fills at the next bar's open by default, exactly so that no
    signal is ever executed at a price that required knowing that bar's
    own close in advance. See `app/backtesting/engine.py` fill policy.

    `stop_loss` is an absolute price. `take_profit` is OPTIONAL and,
    when left as `None` (the default — ORB/RSI divergence never set
    it), is derived by the engine from the actual fill price and the
    configured `risk_reward` ratio, exactly as before this field
    existed. A strategy that targets a specific level rather than a
    fixed R-multiple (e.g. "previous day's high/low") sets this
    explicitly instead — the engine then uses it as-is rather than
    computing one from `risk_reward`. Either way it reflects the real
    entry only insofar as the strategy computed it from information
    available at `signal_time`; the engine itself never recomputes it
    from the fill price when explicitly set.
    """

    signal_time: pd.Timestamp
    direction: Direction
    stop_loss: float
    strategy_name: str
    take_profit: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)


class Strategy(Protocol):
    """Protocol every strategy implements.

    Deliberately a `Protocol`, not an ABC with required inheritance —
    a strategy only needs to provide `name`, `params`, and
    `generate_signals`; it does not need to inherit from a shared base
    class, keeping strategies free of coupling to engine internals.
    """

    name: str

    def params(self) -> dict[str, Any]:
        """Return the exact configuration used, for recording alongside
        every trade (reproducibility requirement)."""
        ...

    def generate_signals(self, market: MarketContext) -> list[Signal]:
        """Return signals in ascending `signal_time` order. Must not
        reference any row of `market.execution_df` or `market.context`
        timestamped after the bar being evaluated — see each strategy's
        module docstring for how it enforces this."""
        ...
