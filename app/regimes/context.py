"""`RegimeContext`: the M4 engine's single output value object.

Named `RegimeContext` rather than `MarketContext` (the M4 mission's own
pseudocode calls it "MarketContext") on purpose: `app.strategies.base`
already defines a class called `MarketContext` with a completely
different shape and purpose (a strategy's INPUT - execution_df + HTF
context frames), not the regime engine's OUTPUT (a description of
market conditions). Two same-named, differently-shaped classes in one
codebase is a real cost (ambiguous imports, confusing greps, "which
MarketContext?" in code review) for a name that isn't load-bearing -
"regime context" describes exactly what this is. This is a deliberate
deviation from the mission's literal pseudocode naming; see
docs/M4_REGIME_CONTEXT.md for the same note in context.

This is purely DESCRIPTIVE. Per the M4 mission (section 11): "This is
descriptive context, NOT a trading recommendation." There is
deliberately no direction/confidence/action field here, and there
never should be - that belongs to a future decision/probability layer
that CONSUMES a `RegimeContext`, not this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.regimes.models import StructureState, TrendState, VolatilityState


@dataclass(frozen=True)
class RegimeContext:
    timestamp: pd.Timestamp
    instrument_symbol: str
    timeframe: str

    trend_state: TrendState
    volatility_state: VolatilityState
    structure_state: StructureState

    # Named sessions active at `timestamp` (e.g. ("london", "new_york")
    # during the overlap window) - empty tuple, not None, when no
    # configured session covers this timestamp (an explicit, JSON-safe
    # "unavailable" rather than a missing key). See
    # `app.regimes.classifier` for the session definitions used.
    sessions: tuple[str, ...] = field(default_factory=tuple)

    # Every raw feature value used to produce the three states above,
    # already JSON-safe (NaN -> None) - see
    # `RegimeFeatureSet.feature_values_at`. Kept so a consumer (or a
    # human debugging a classification) never has to take the states on
    # faith.
    feature_values: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A fully JSON-safe representation - enums to their `.value`
        string, the timestamp to ISO-8601, sessions to a plain list."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "instrument_symbol": self.instrument_symbol,
            "timeframe": self.timeframe,
            "trend_state": self.trend_state.value,
            "volatility_state": self.volatility_state.value,
            "structure_state": self.structure_state.value,
            "sessions": list(self.sessions),
            "feature_values": dict(self.feature_values),
        }
