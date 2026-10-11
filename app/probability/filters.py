"""A single, composable filter specification for `OutcomeDataset` -
every field is `Optional`, `None` meaning "don't filter on this
dimension". This is what lets a caller query:

    OutcomeFilter(strategy_name="orb")
    OutcomeFilter(strategy_name="orb", instrument_symbol="EURUSD")
    OutcomeFilter(strategy_name="orb", instrument_symbol="EURUSD", session="london")
    OutcomeFilter(strategy_name="orb", instrument_symbol="EURUSD", session="london",
                  trend_state=TrendState.UPTREND, volatility_state=VolatilityState.HIGH)

without a hard-coded function per combination (M5 mission section 17) -
`matches()` just checks whichever fields are set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.backtesting.results import Direction
from app.probability.models import TradeOutcome
from app.probability.outcomes import OutcomeDataset
from app.regimes.models import StructureState, TrendState, VolatilityState


@dataclass(frozen=True)
class OutcomeFilter:
    strategy_name: Optional[str] = None
    instrument_symbol: Optional[str] = None
    timeframe: Optional[str] = None
    direction: Optional[Direction] = None

    trend_state: Optional[TrendState] = None
    volatility_state: Optional[VolatilityState] = None
    structure_state: Optional[StructureState] = None
    # Membership test, not equality - RegimeContext.sessions is a tuple
    # (sessions can overlap), so "session='london'" means "london is one
    # of the active sessions", not "the only one".
    session: Optional[str] = None

    def matches(self, outcome: TradeOutcome) -> bool:
        if self.strategy_name is not None and outcome.strategy_name != self.strategy_name:
            return False
        if self.instrument_symbol is not None and outcome.instrument_symbol != self.instrument_symbol:
            return False
        if self.timeframe is not None and outcome.timeframe != self.timeframe:
            return False
        if self.direction is not None and outcome.direction != self.direction:
            return False

        needs_regime = self.trend_state is not None or self.volatility_state is not None or self.structure_state is not None or self.session is not None
        if needs_regime:
            # A filter that asks for a regime dimension cannot match an
            # outcome with no regime context at all (M5 mission section 23:
            # "missing regime context" is an edge case to handle explicitly,
            # not silently ignore).
            if outcome.regime_context is None:
                return False
            ctx = outcome.regime_context
            if self.trend_state is not None and ctx.trend_state != self.trend_state:
                return False
            if self.volatility_state is not None and ctx.volatility_state != self.volatility_state:
                return False
            if self.structure_state is not None and ctx.structure_state != self.structure_state:
                return False
            if self.session is not None and self.session not in ctx.sessions:
                return False

        return True

    def apply(self, dataset: OutcomeDataset) -> OutcomeDataset:
        return OutcomeDataset(outcomes=tuple(o for o in dataset if self.matches(o)))
