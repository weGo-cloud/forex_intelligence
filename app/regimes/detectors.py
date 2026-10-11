"""Pure classification functions: scalar feature values in, a single
domain state out. No hidden state (every function takes its config
explicitly), deterministic (same inputs -> same output, always), and
safe on missing data (NaN anywhere required -> the `UNKNOWN` member of
whichever enum - never a guessed real state).

Kept as free functions operating on plain floats (not on
`RegimeFeatureSet`/DataFrame positions) specifically so they are cheap
to unit test directly with hand-picked values, independent of the
vectorized feature-computation plumbing in `app.regimes.features`.
"""

from __future__ import annotations

import math

from app.regimes.config import RegimeClassifierConfig
from app.regimes.models import StructureState, TrendState, VolatilityState


def _is_missing(*values: float) -> bool:
    return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values)


def classify_trend(
    close: float,
    slow_ma: float,
    slow_ma_slope: float,
    plus_di: float,
    minus_di: float,
    adx: float,
    config: RegimeClassifierConfig,
) -> TrendState:
    """Combine THREE independent directional votes (per the M4 mission's
    "use multiple measurable signals" guidance) - price vs the slow MA,
    the slow MA's own slope sign, and +DI vs -DI (directional movement)
    - with ADX used separately as a STRENGTH GATE, not a vote (ADX has
    no direction of its own). Requires `config.trend_min_agreeing_votes`
    (default 2 of 3) to agree AND `adx >= config.trend_adx_threshold`
    for a directional call; anything else is NEUTRAL - a measured "no
    clear trend", not a data problem.
    """
    if _is_missing(close, slow_ma, slow_ma_slope, plus_di, minus_di, adx):
        return TrendState.UNKNOWN

    votes = 0
    votes += 1 if close > slow_ma else (-1 if close < slow_ma else 0)
    votes_slope = 1 if slow_ma_slope > 0 else (-1 if slow_ma_slope < 0 else 0)
    votes_di = 1 if plus_di > minus_di else (-1 if plus_di < minus_di else 0)

    bullish_votes = sum(1 for v in (votes, votes_slope, votes_di) if v > 0)
    bearish_votes = sum(1 for v in (votes, votes_slope, votes_di) if v < 0)

    if adx < config.trend_adx_threshold:
        return TrendState.NEUTRAL

    if bullish_votes >= config.trend_min_agreeing_votes and bullish_votes > bearish_votes:
        return TrendState.UPTREND
    if bearish_votes >= config.trend_min_agreeing_votes and bearish_votes > bullish_votes:
        return TrendState.DOWNTREND
    return TrendState.NEUTRAL


def classify_volatility(volatility_percentile: float, config: RegimeClassifierConfig) -> VolatilityState:
    """`volatility_percentile` (see
    `app.features.volatility.rolling_percentile_rank`) is where the
    current normalized ATR ranks (0-100) within its own recent
    (`config.volatility_percentile_window`-bar) history - a RELATIVE,
    self-referential measure per-instrument, never a fixed absolute ATR
    value (see config.py's docstring / the M4 mission's explicit
    requirement on this point).
    """
    if _is_missing(volatility_percentile):
        return VolatilityState.UNKNOWN
    if volatility_percentile <= config.volatility_low_percentile:
        return VolatilityState.LOW
    if volatility_percentile >= config.volatility_high_percentile:
        return VolatilityState.HIGH
    return VolatilityState.NORMAL


def classify_structure(
    structural_trend_strength: float, range_compression_ratio: float, config: RegimeClassifierConfig
) -> StructureState:
    """Requires BOTH a consistent swing sequence (|structural_trend_strength|
    at/above threshold) AND range expansion for TRENDING; BOTH a weak/
    absent swing sequence AND range compression for RANGING. Anything
    else (the two signals disagree - e.g. consistent higher-highs but a
    compressed range, or a weak swing sequence with an expanding range)
    is TRANSITION: genuinely mixed evidence, not an arbitrary tie-break.
    This keeps structure independent from trend DIRECTION by design
    (per the M4 mission section 10) - `structural_trend_strength`'s
    magnitude (not sign) drives this, so a strong downtrend structure
    and a strong uptrend structure are both "TRENDING" here; direction
    itself is `TrendState`'s job, not `StructureState`'s.
    """
    if _is_missing(structural_trend_strength, range_compression_ratio):
        return StructureState.UNKNOWN

    strong_structure = abs(structural_trend_strength) >= config.structure_trend_strength_threshold
    expanding = range_compression_ratio >= config.structure_expansion_threshold
    compressed = range_compression_ratio <= config.structure_compression_threshold

    if strong_structure and expanding:
        return StructureState.TRENDING
    if not strong_structure and compressed:
        return StructureState.RANGING
    return StructureState.TRANSITION
