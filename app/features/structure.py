"""Reusable market-structure MEASUREMENTS (not interpretations).

Built entirely on `app.features.indicators.detect_swing_points`
(reused, not duplicated) - this module only adds what raw swing points
don't give on their own: labeling each swing relative to the prior
swing of the same kind (higher high / lower high / higher low / lower
low), and a compact numeric "structural trend strength" derived from
recent labels, plus a separate price-range compression/expansion
measurement.

CAUSALITY: `detect_swing_points` already enforces confirmation lag (a
pivot at `index_pos` isn't knowable until `confirmed_pos`) - every
function here preserves that lag exactly, never treating a swing (or a
label built from two swings) as known before both swings' own
`confirmed_pos`/`confirmed_at`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from app.features.indicators import SwingPoint

StructureLabel = Literal["higher_high", "lower_high", "higher_low", "lower_low"]


@dataclass(frozen=True)
class LabeledSwing:
    """A swing point plus its label relative to the PRIOR swing of the
    same kind. `label` is None for the first high (or first low) in the
    series, since there is nothing to compare it to yet. `available_at`
    is the position from which this label is causally knowable - the
    LATER of the two swings' own `confirmed_pos` (in practice always
    this swing's own `confirmed_pos`, since the prior same-kind swing
    necessarily confirmed earlier).
    """

    swing: SwingPoint
    label: Optional[StructureLabel]
    available_at: int


def label_swings(swings: list[SwingPoint]) -> list[LabeledSwing]:
    """Label each swing high against the previous swing high, and each
    swing low against the previous swing low, in chronological
    (`index_pos`) order - which of the two events happened doesn't
    matter for this comparison, only "the last swing of the same kind
    before this one".
    """
    ordered = sorted(swings, key=lambda s: s.index_pos)
    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None
    labeled: list[LabeledSwing] = []

    for swing in ordered:
        if swing.kind == "high":
            if last_high is None:
                label = None
            else:
                label = "higher_high" if swing.price > last_high.price else "lower_high"
            available_at = swing.confirmed_pos if last_high is None else max(swing.confirmed_pos, last_high.confirmed_pos)
            last_high = swing
        else:
            if last_low is None:
                label = None
            else:
                label = "higher_low" if swing.price > last_low.price else "lower_low"
            available_at = swing.confirmed_pos if last_low is None else max(swing.confirmed_pos, last_low.confirmed_pos)
            last_low = swing

        labeled.append(LabeledSwing(swing=swing, label=label, available_at=available_at))

    return labeled


def structural_trend_strength_series(
    index: pd.DatetimeIndex, swings: list[SwingPoint], lookback_count: int
) -> pd.Series:
    """A per-bar series in `[-1.0, 1.0]` (NaN until the first labeled
    swing is available): among the most recent `lookback_count` LABELED
    swings known as of bar i, the fraction that are higher_high/
    higher_low minus the fraction that are lower_high/lower_low.
    +1.0 = every recent swing confirms an uptrend structure, -1.0 = every
    recent swing confirms a downtrend structure, values near 0 = mixed/
    weak structural evidence.

    Causal by construction: a labeled swing's value is only attached at
    its own `available_at` position, and `Series.reindex(...,
    method="ffill")` only ever propagates a value FORWARD in time onto
    later bars - never backward onto bars before it was known.
    """
    if lookback_count < 1:
        raise ValueError("lookback_count must be >= 1")

    labeled = [ls for ls in label_swings(swings) if ls.label is not None]
    labeled.sort(key=lambda ls: ls.available_at)

    if not labeled:
        return pd.Series(float("nan"), index=index)

    scores: list[float] = []
    score_positions: list[int] = []
    window: list[int] = []  # +1 / -1 per label, most-recent-last
    for ls in labeled:
        window.append(1 if ls.label in ("higher_high", "higher_low") else -1)
        if len(window) > lookback_count:
            window.pop(0)
        scores.append(sum(window) / len(window))
        score_positions.append(ls.available_at)

    # Attach each score to its bar's timestamp, then forward-fill onto
    # every later bar up to (but not past) the next score. Two labeled
    # swings can legitimately share the same `available_at` position
    # (e.g. a high and a low confirming on the same bar) - keep only the
    # LAST score for a duplicated position, since `labeled` is already
    # sorted by `available_at` and the last one incorporates every swing
    # known as of that bar (ties broken by list order, which is stable).
    score_series = pd.Series(scores, index=[index[p] for p in score_positions])
    score_series = score_series[~score_series.index.duplicated(keep="last")]
    return score_series.reindex(index, method="ffill")


def range_compression_ratio(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int, baseline_period: int
) -> pd.Series:
    """Current `period`-bar high-low range (normalized by price) versus
    its own rolling mean over `baseline_period` bars: < 1.0 means the
    recent range is compressed relative to its own history (consistent
    with a ranging market), > 1.0 means expanded (consistent with a
    trending/breakout market). Deliberately self-referential (same
    reasoning as `app.features.volatility.volatility_ratio`) rather than
    any fixed absolute range size, so it needs no per-instrument
    calibration. Causal. NaN through both rolling windows' warm-up.
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    if baseline_period < 2:
        raise ValueError("baseline_period must be >= 2")

    current_range = (high.rolling(window=period).max() - low.rolling(window=period).min()) / close
    baseline = current_range.rolling(window=baseline_period).mean()
    baseline_guarded = baseline.where(baseline != 0.0)
    return current_range / baseline_guarded
