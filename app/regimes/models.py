"""Domain enums for the M4 Market Regime & Context Engine.

Each dimension (trend/volatility/structure) is deliberately kept
SEPARATE rather than combined into one giant enum of every
combination (per the M4 mission's explicit instruction) - a consumer
that cares about "uptrend AND ranging" reads two fields, not one
magic combined value.

Every enum includes an explicit `UNKNOWN` member alongside its normal
states. This is not one of the mission's minimum-listed members, but
it exists to satisfy a specific, explicit requirement (M4 mission
section 16, Test F): "Missing/invalid feature values result in safe
explicit states rather than invented values." Without `UNKNOWN`, the
only options when a feature is NaN (insufficient warm-up, bad data,
etc.) would be to either raise (which would make the whole context
computation for a whole series fail on incomplete leading bars - the
normal case, not an edge case) or silently pick one of the "real"
states (which IS inventing a value - the exact thing Test F forbids).
`TrendState.NEUTRAL` is kept distinct from `UNKNOWN`: NEUTRAL is a
*measured* "no clear trend" (weak/conflicting evidence, computed from
real data), UNKNOWN is "we don't have enough/valid data to measure
anything at all". Conflating the two would make it impossible to tell
"the market is genuinely ranging" apart from "we can't tell yet".
"""

from __future__ import annotations

import enum


class TrendState(str, enum.Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class VolatilityState(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class StructureState(str, enum.Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    TRANSITION = "transition"
    UNKNOWN = "unknown"
