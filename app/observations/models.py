"""The canonical observation shape every future source must produce.

`ObservationSource` exists specifically so a screenshot-derived
observation and a market-data-derived observation are THE SAME TYPE at
the setup-detection boundary (mission section 6/14 - "hard
architectural requirement": screenshot-derived observations must not be
architecturally impossible, even though no screenshot analyzer is
built in M7). `app.setups.detection.SetupDetector` consumes
`list[Observation]` and never asks which source produced them - a
future vision-analysis module (M9) only has to produce
`Observation`s with `source=ObservationSource.SCREENSHOT` and can feed
directly into the exact same detection/decision pipeline market-data
observations already use, with zero changes to `app.setups` or
`app.decision`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.strategies.base import Direction


class ObservationType(str, enum.Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    LIQUIDITY_POOL = "liquidity_pool"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    BOS = "bos"
    CHOCH = "choch"
    FVG = "fvg"
    TREND_ALIGNMENT = "trend_alignment"
    VOLATILITY_STATE = "volatility_state"
    PRICE_LOCATION = "price_location"


class ObservationSource(str, enum.Enum):
    """Where an observation came from. Only `MARKET_DATA` and
    `STRATEGY_ENGINE` are actually produced in M7 - `SCREENSHOT` and
    `MANUAL` exist now purely so the type system and every downstream
    consumer already account for them; adding a real screenshot
    analyzer later is additive (a new detector module that emits
    `Observation`s), not a schema change.
    """

    MARKET_DATA = "market_data"
    STRATEGY_ENGINE = "strategy_engine"
    SCREENSHOT = "screenshot"
    MANUAL = "manual"


@dataclass(frozen=True)
class Observation:
    """One structured, timestamped fact about the market.

    `direction`, `price`, and `zone` are all `Optional` because not
    every `ObservationType` uses all of them (e.g. `VOLATILITY_STATE`
    has no price/direction at all; `SUPPORT`/`RESISTANCE` have a price
    but often no inherent direction). `confidence` is a plain `[0, 1]`
    float describing how well-evidenced the observation itself is
    (e.g. a support level touched 4 times vs. 2) - it is NOT a
    trading-confidence score and is never propagated into the decision
    layer as one (see `app.decision`'s module docstring).
    """

    observation_type: ObservationType
    instrument_symbol: str
    timeframe: str
    timestamp: datetime  # when this observation became causally knowable (confirmed_at / index timestamp)
    source: ObservationSource

    direction: Optional[Direction] = None
    price: Optional[float] = None
    zone_low: Optional[float] = None
    zone_high: Optional[float] = None
    confidence: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (naive timestamps are rejected project-wide)")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.zone_low is not None and self.zone_high is not None and self.zone_low > self.zone_high:
            raise ValueError("zone_low must be <= zone_high")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        """Inverse of `to_dict()`. Any source (including a future
        screenshot analyzer) can submit observations as plain dicts and
        get the same validation market-data observations get - unknown
        `observation_type`/`source` values raise `ValueError`."""
        return cls(
            observation_type=ObservationType(data["observation_type"]),
            instrument_symbol=data["instrument_symbol"],
            timeframe=data["timeframe"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=ObservationSource(data["source"]),
            direction=data.get("direction"),
            price=data.get("price"),
            zone_low=data.get("zone_low"),
            zone_high=data.get("zone_high"),
            confidence=data.get("confidence", 1.0),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_type": self.observation_type.value,
            "instrument_symbol": self.instrument_symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "direction": self.direction,
            "price": self.price,
            "zone_low": self.zone_low,
            "zone_high": self.zone_high,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


def observations_as_of(observations: list[Observation], as_of: datetime) -> list[Observation]:
    """Only the observations that were causally knowable at `as_of`
    (`timestamp <= as_of`). This is the one call that makes a
    historical setup/decision evaluation walk-forward-safe (mission
    section 18): pass the evaluation time `T` here BEFORE handing
    observations to `app.setups.detection.SetupDetector`, and nothing
    that only became knowable after `T` can influence the result.
    Same rule, same shape as M5's `OutcomeDataset.resolved_before`.
    """
    return [o for o in observations if o.timestamp <= as_of]
