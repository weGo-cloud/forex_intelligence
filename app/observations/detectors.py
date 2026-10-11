"""Deterministic observation detectors.

Every function here is a thin wrapper that packages an ALREADY-CAUSAL
M1-M6 output into the uniform `Observation` shape - none of them
recompute an indicator, reclassify a regime, or invent a detection
technique that doesn't already exist somewhere in M2-M4:

- `swing_observations` / `fvg_observations` / `structure_observations` /
  `support_resistance_observations` are built on
  `app.features.indicators.detect_swing_points`/`detect_fvg` and
  `app.features.structure.label_swings` (M2/M4).
- `liquidity_sweep_observations` generalizes the exact sweep-detection
  technique `app.strategies.liquidity_sweep_fvg` already uses inline
  (1D bias vs. a 1H candle sweeping and rejecting the previous day's
  extreme) into a standalone, reusable, source-agnostic detector - not
  a duplicate implementation, an extraction of a technique that was
  previously private to one strategy's control flow.
- `regime_observations` packages an already-computed `RegimeContext`
  (M4) - it does not reclassify anything.

CAUSALITY: every observation's `timestamp` is the position at which it
became CAUSALLY KNOWABLE (a swing's `confirmed_at`, an FVG's
`confirmed_at`, the bar at which a sweep candle's OHLC closed) - never
the position of the event it describes if that event's true extent
wasn't knowable until later. This is inherited directly from the
M2/M4 primitives' own confirmation-lag discipline, not reimplemented
here.
"""

from __future__ import annotations

import statistics
from typing import Optional

import pandas as pd

from app.features.indicators import detect_fvg, detect_swing_points
from app.features.structure import label_swings
from app.observations.models import Observation, ObservationSource, ObservationType
from app.regimes.context import RegimeContext
from app.regimes.models import TrendState


def swing_observations(
    high: pd.Series,
    low: pd.Series,
    instrument_symbol: str,
    timeframe: str,
    left_bars: int = 3,
    right_bars: int = 3,
) -> list[Observation]:
    swings = detect_swing_points(high, low, left_bars=left_bars, right_bars=right_bars)
    return [
        Observation(
            observation_type=ObservationType.SWING_HIGH if s.kind == "high" else ObservationType.SWING_LOW,
            instrument_symbol=instrument_symbol,
            timeframe=timeframe,
            timestamp=s.confirmed_at,
            source=ObservationSource.MARKET_DATA,
            price=s.price,
            metadata={"index_pos": s.index_pos, "confirmed_pos": s.confirmed_pos},
        )
        for s in swings
    ]


def fvg_observations(
    high: pd.Series,
    low: pd.Series,
    instrument_symbol: str,
    timeframe: str,
    min_gap_size: float = 0.0,
) -> list[Observation]:
    zones = detect_fvg(high, low, min_gap_size=min_gap_size)
    return [
        Observation(
            observation_type=ObservationType.FVG,
            instrument_symbol=instrument_symbol,
            timeframe=timeframe,
            timestamp=z.confirmed_at,
            source=ObservationSource.MARKET_DATA,
            direction="long" if z.kind == "bullish" else "short",
            zone_low=z.zone_low,
            zone_high=z.zone_high,
            metadata={"kind": z.kind},
        )
        for z in zones
    ]


def structure_observations(
    high: pd.Series,
    low: pd.Series,
    instrument_symbol: str,
    timeframe: str,
    left_bars: int = 3,
    right_bars: int = 3,
) -> list[Observation]:
    """BOS (break of structure) / CHoCH (change of character), derived
    from `label_swings` (M4): a labeled swing that CONTINUES the
    currently-tracked bias (e.g. a higher_high while bias is already
    bullish) is BOS; one that CONTRADICTS it (e.g. a lower_low while
    bias is bullish) is CHoCH, and flips the tracked bias going forward.
    The very first labeled swing only establishes a starting bias -
    there's nothing before it to call a break or a change of.
    """
    swings = detect_swing_points(high, low, left_bars=left_bars, right_bars=right_bars)
    labeled = [ls for ls in label_swings(swings) if ls.label is not None]
    labeled.sort(key=lambda ls: ls.available_at)

    observations: list[Observation] = []
    bias: Optional[str] = None
    for ls in labeled:
        label_bias = "bullish" if ls.label in ("higher_high", "higher_low") else "bearish"
        if bias is None:
            bias = label_bias
            continue

        if label_bias == bias:
            obs_type = ObservationType.BOS
        else:
            obs_type = ObservationType.CHOCH
            bias = label_bias

        observations.append(
            Observation(
                observation_type=obs_type,
                instrument_symbol=instrument_symbol,
                timeframe=timeframe,
                timestamp=ls.swing.confirmed_at,
                source=ObservationSource.MARKET_DATA,
                direction="long" if label_bias == "bullish" else "short",
                price=ls.swing.price,
                metadata={"label": ls.label},
            )
        )
    return observations


def support_resistance_observations(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    instrument_symbol: str,
    timeframe: str,
    left_bars: int = 3,
    right_bars: int = 3,
    tolerance_pct: float = 0.001,
    min_touches: int = 2,
) -> list[Observation]:
    """Clusters confirmed swing prices (both highs and lows - a level
    can be touched by either) within `tolerance_pct` of each other; a
    cluster with at least `min_touches` swings becomes a level, reported
    ONCE, at the moment it first reaches `min_touches`.

    PREFIX-STABLE BY CONSTRUCTION: swings are processed one at a time in
    confirmation order, and re-clustered using only the swings confirmed
    so far - so an observation emitted at time t depends solely on swings
    confirmed <= t. (An earlier version clustered all swings globally,
    which let later data shift an existing level's timestamp and price -
    caught by `tests/decision/test_causality.py`.) Consequence,
    documented as a known limitation: a level is reported at formation
    only; later touches that strengthen it are not re-emitted.

    SUPPORT vs RESISTANCE is decided by comparing the level to `close`
    at the confirming swing's own `confirmed_pos` - never to a later
    price.
    """
    if tolerance_pct <= 0:
        raise ValueError("tolerance_pct must be > 0")
    if min_touches < 2:
        raise ValueError("min_touches must be >= 2 (a single swing is not a level)")

    swings = detect_swing_points(high, low, left_bars=left_bars, right_bars=right_bars)
    if not swings:
        return []

    observations: list[Observation] = []
    seen: list = []
    emitted_members: set[int] = set()  # index_pos of swings already part of an emitted level

    for new_swing in swings:  # detect_swing_points returns swings sorted by (confirmed_pos, index_pos)
        seen.append(new_swing)

        ordered = sorted(seen, key=lambda sp: sp.price)
        clusters: list[list] = []
        current: list = []
        for sp in ordered:
            if current and abs(sp.price - current[-1].price) > tolerance_pct * current[-1].price:
                clusters.append(current)
                current = []
            current.append(sp)
        if current:
            clusters.append(current)

        cluster = next((c for c in clusters if any(sp.index_pos == new_swing.index_pos for sp in c)), None)
        if cluster is None or len(cluster) < min_touches:
            continue
        if any(sp.index_pos in emitted_members for sp in cluster):
            continue  # this level was already reported when it first formed

        reference_close = close.iloc[new_swing.confirmed_pos]
        if pd.isna(reference_close):
            continue

        level_price = statistics.mean(sp.price for sp in cluster)
        obs_type = ObservationType.RESISTANCE if level_price > reference_close else ObservationType.SUPPORT
        observations.append(
            Observation(
                observation_type=obs_type,
                instrument_symbol=instrument_symbol,
                timeframe=timeframe,
                timestamp=new_swing.confirmed_at,
                source=ObservationSource.MARKET_DATA,
                price=level_price,
                confidence=min(1.0, len(cluster) / (min_touches * 2)),
                metadata={"touch_count": len(cluster)},
            )
        )
        emitted_members.update(sp.index_pos for sp in cluster)

    return observations


def liquidity_sweep_observations(
    execution_df: pd.DataFrame,
    daily_context: pd.DataFrame,
    session_context: pd.DataFrame,
    instrument_symbol: str,
    timeframe: str,
) -> list[Observation]:
    """The same daily-bias + 1H-sweep-and-reject technique
    `app.strategies.liquidity_sweep_fvg` uses inline, extracted as a
    reusable, standalone detector (see module docstring). `daily_context`/
    `session_context` must already be causally aligned via
    `app.backtesting.data_access.align_higher_timeframe` (M2/M3) - this
    function does no alignment of its own, exactly like the strategy it
    generalizes.
    """
    observations: list[Observation] = []
    last_seen_1h: Optional[tuple] = None

    for i, ts in enumerate(execution_df.index):
        daily_row = daily_context.iloc[i]
        open_p, close_p = daily_row.get("open_htf"), daily_row.get("close_htf")
        high_p, low_p = daily_row.get("high_htf"), daily_row.get("low_htf")
        if pd.isna(open_p) or pd.isna(close_p) or pd.isna(high_p) or pd.isna(low_p):
            continue
        if close_p == open_p:
            continue
        bias = "long" if close_p > open_p else "short"

        session_row = session_context.iloc[i]
        o1, h1, l1, c1 = (
            session_row.get("open_htf"),
            session_row.get("high_htf"),
            session_row.get("low_htf"),
            session_row.get("close_htf"),
        )
        if pd.isna(c1):
            continue
        current_1h = (o1, h1, l1, c1)
        if current_1h == last_seen_1h:
            continue
        last_seen_1h = current_1h

        if bias == "long" and l1 < low_p and c1 > low_p:
            observations.append(
                Observation(
                    observation_type=ObservationType.LIQUIDITY_SWEEP,
                    instrument_symbol=instrument_symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    source=ObservationSource.MARKET_DATA,
                    direction="long",
                    price=float(l1),
                    metadata={"swept_level": float(low_p), "bias": bias},
                )
            )
        elif bias == "short" and h1 > high_p and c1 < high_p:
            observations.append(
                Observation(
                    observation_type=ObservationType.LIQUIDITY_SWEEP,
                    instrument_symbol=instrument_symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    source=ObservationSource.MARKET_DATA,
                    direction="short",
                    price=float(h1),
                    metadata={"swept_level": float(high_p), "bias": bias},
                )
            )

    return observations


def regime_observations(regime_context: RegimeContext, current_price: Optional[float] = None) -> list[Observation]:
    """Packages an already-classified `RegimeContext` (M4) into
    observations - no reclassification happens here."""
    observations: list[Observation] = []

    if regime_context.trend_state != TrendState.UNKNOWN:
        direction = None
        if regime_context.trend_state == TrendState.UPTREND:
            direction = "long"
        elif regime_context.trend_state == TrendState.DOWNTREND:
            direction = "short"
        observations.append(
            Observation(
                observation_type=ObservationType.TREND_ALIGNMENT,
                instrument_symbol=regime_context.instrument_symbol,
                timeframe=regime_context.timeframe,
                timestamp=regime_context.timestamp,
                source=ObservationSource.MARKET_DATA,
                direction=direction,
                metadata={"trend_state": regime_context.trend_state.value},
            )
        )

    observations.append(
        Observation(
            observation_type=ObservationType.VOLATILITY_STATE,
            instrument_symbol=regime_context.instrument_symbol,
            timeframe=regime_context.timeframe,
            timestamp=regime_context.timestamp,
            source=ObservationSource.MARKET_DATA,
            metadata={"volatility_state": regime_context.volatility_state.value},
        )
    )

    observations.append(
        Observation(
            observation_type=ObservationType.PRICE_LOCATION,
            instrument_symbol=regime_context.instrument_symbol,
            timeframe=regime_context.timeframe,
            timestamp=regime_context.timestamp,
            source=ObservationSource.MARKET_DATA,
            price=current_price,
            metadata={"structure_state": regime_context.structure_state.value},
        )
    )

    return observations
