"""Shared, reusable technical indicators for M2 strategies.

Deliberately NOT using TA-Lib or the `ta` PyPI package here — two reasons,
not just convenience:

1. This environment has no network access to install/verify a new native
   or pure-Python dependency, and silently trusting an unverified
   implementation of an indicator that feeds directly into trading
   decisions is a worse risk than a small, fully-inspectable
   implementation of a well-defined formula (RSI is ~10 lines; Wilder's
   original definition is unambiguous).
2. Swing/pivot detection (the harder half of this module) has no
   standard library implementation that handles *causal confirmation*
   correctly (see `detect_swing_points` below) — this had to be written
   from scratch regardless of what indicator library was chosen for RSI.

Both functions here are causal: at row i, the result depends only on
values at row <= i, EXCEPT `detect_swing_points`, which explicitly
returns a separate `confirmed_at` timestamp/index for exactly this
reason — a swing point at bar i is only *knowable* once `right` future
bars exist, and callers (strategies) MUST gate on `confirmed_at`, never
on the pivot's own timestamp. This is the load-bearing anti-lookahead
mechanism for the whole divergence strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SwingKind = Literal["high", "low"]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, causal (each value depends only on past closes).

    Implements Wilder's exact original recurrence, not the common
    "plain EWMA from bar zero" approximation: the first average
    gain/loss is a simple mean of the first `period` changes, and every
    value after that is the Wilder-smoothed recurrence
    `avg[t] = (avg[t-1] * (period - 1) + value[t]) / period`. A plain
    `ewm(alpha=1/period, adjust=False)` from the first observation gives
    numerically different (converging, but not identical) early values,
    which matters for reproducibility against reference values. Written
    as an explicit loop rather than a vectorized approximation because
    this recurrence is inherently sequential and auditability matters
    more than the small speed cost here (correctness > speed).

    Returns NaN for the first `period` bars, where the average gain/loss
    is not yet defined. When average loss is exactly zero (a strict
    uptrend over the lookback), RSI is defined as 100.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    delta = close.diff()
    gain = delta.clip(lower=0.0).to_numpy(dtype=float)
    loss = (-delta.clip(upper=0.0)).to_numpy(dtype=float)
    n = len(close)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    if n > period:
        # gain[0]/loss[0] are NaN (diff has no predecessor for the first
        # bar); the first `period` valid changes are gain[1:period+1].
        avg_gain[period] = np.nanmean(gain[1 : period + 1])
        avg_loss[period] = np.nanmean(loss[1 : period + 1])
        for t in range(period + 1, n):
            avg_gain[t] = (avg_gain[t - 1] * (period - 1) + gain[t]) / period
            avg_loss[t] = (avg_loss[t - 1] * (period - 1) + loss[t]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        result = 100.0 - (100.0 / (1.0 + rs))
    result = np.where(avg_loss == 0.0, 100.0, result)
    result = np.where(np.isnan(avg_gain) | np.isnan(avg_loss), np.nan, result)
    return pd.Series(result, index=close.index, name=f"rsi_{period}")


@dataclass(frozen=True)
class SwingPoint:
    """A single confirmed swing high/low.

    `index_pos` is the integer position (in the source series) of the
    pivot bar itself. `confirmed_pos` is the integer position of the
    earliest bar at which this pivot is knowable without hindsight —
    strategies must never act on a SwingPoint using data timestamped
    before `confirmed_pos`.
    """

    index_pos: int
    confirmed_pos: int
    timestamp: pd.Timestamp
    confirmed_at: pd.Timestamp
    price: float
    kind: SwingKind


def detect_swing_points(
    high: pd.Series,
    low: pd.Series,
    left_bars: int = 3,
    right_bars: int = 3,
) -> list[SwingPoint]:
    """Detect fractal-style swing highs/lows with explicit confirmation lag.

    A bar at position i is a swing HIGH if its `high` is strictly greater
    than every other high in the window [i-left_bars, i+right_bars]. A
    swing LOW is the symmetric condition on `low`. Strict inequality
    against every neighbor avoids ambiguous ties.

    This is an objective, mechanical definition (no visual/discretionary
    judgment), per the "no vague visual logic" requirement — but the
    critical property for backtest validity is `confirmed_pos`: bar i
    cannot be confirmed as a pivot until bar i+right_bars has occurred,
    because confirming it requires knowing that no higher high (or lower
    low) occurred in the following `right_bars` bars. A caller that acts
    on this pivot before `confirmed_at` has has introduced look-ahead
    bias. `detect_bias_test.py` in tests/backtesting specifically checks
    this property.

    high/low must share the same index, sorted ascending, with no gaps
    assumed (the function operates on integer position, not time).
    """
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must both be >= 1")
    if len(high) != len(low):
        raise ValueError("high and low must be the same length")

    n = len(high)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    index = high.index

    points: list[SwingPoint] = []
    for i in range(left_bars, n - right_bars):
        window_slice = slice(i - left_bars, i + right_bars + 1)

        window_high = highs[window_slice]
        pivot_high = highs[i]
        others_high = np.delete(window_high, left_bars)
        if others_high.size and pivot_high > others_high.max():
            confirmed_pos = i + right_bars
            points.append(
                SwingPoint(
                    index_pos=i,
                    confirmed_pos=confirmed_pos,
                    timestamp=index[i],
                    confirmed_at=index[confirmed_pos],
                    price=float(pivot_high),
                    kind="high",
                )
            )

        window_low = lows[window_slice]
        pivot_low = lows[i]
        others_low = np.delete(window_low, left_bars)
        if others_low.size and pivot_low < others_low.min():
            confirmed_pos = i + right_bars
            points.append(
                SwingPoint(
                    index_pos=i,
                    confirmed_pos=confirmed_pos,
                    timestamp=index[i],
                    confirmed_at=index[confirmed_pos],
                    price=float(pivot_low),
                    kind="low",
                )
            )

    points.sort(key=lambda p: (p.confirmed_pos, p.index_pos))
    return points


FVGKind = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class FVGZone:
    """A single Fair Value Gap (FVG): a 3-candle imbalance where candle 1
    and candle 3 don't overlap, leaving a "gap" in candle 2's range that
    price is considered likely to revisit.

    `index_pos` is the middle candle's position (conventionally "the FVG
    candle"). `confirmed_pos` is the third candle's position - the gap's
    exact boundaries aren't knowable until that candle closes (its own
    high/low might have extended the gap further while the bar was still
    forming), matching the same confirmation-lag discipline as
    `SwingPoint` above. Strategies must not act on a zone using data
    timestamped before `confirmed_at`.
    """

    index_pos: int
    confirmed_pos: int
    timestamp: pd.Timestamp
    confirmed_at: pd.Timestamp
    zone_low: float
    zone_high: float
    kind: FVGKind


def detect_fvg(high: pd.Series, low: pd.Series, min_gap_size: float = 0.0) -> list[FVGZone]:
    """Detect Fair Value Gaps over consecutive candle triples (i, i+1, i+2).

    - Bullish FVG: `low[i+2] > high[i]` - candle 3's low sits above
      candle 1's high. Zone = `[high[i], low[i+2]]`.
    - Bearish FVG: `high[i+2] < low[i]` - candle 3's high sits below
      candle 1's low. Zone = `[high[i+2], low[i]]`.

    `min_gap_size` filters out gaps narrower than this (in price units) -
    default 0.0 keeps every gap regardless of size; callers on
    instruments with a meaningfully large minimum tick/pip size should
    set this explicitly rather than relying on the default, per this
    codebase's "no unstated hard-coded thresholds" convention.

    Purely a 3-candle structural pattern (no lookback/lookahead window
    like `detect_swing_points`), so this needs only `len(high) >= 3` and
    involves no left/right symmetry parameter.
    """
    if min_gap_size < 0:
        raise ValueError("min_gap_size must be >= 0")
    if len(high) != len(low):
        raise ValueError("high and low must be the same length")

    n = len(high)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    index = high.index

    zones: list[FVGZone] = []
    for i in range(n - 2):
        third = i + 2
        if lows[third] > highs[i] and (lows[third] - highs[i]) >= min_gap_size:
            zones.append(
                FVGZone(
                    index_pos=i + 1,
                    confirmed_pos=third,
                    timestamp=index[i + 1],
                    confirmed_at=index[third],
                    zone_low=float(highs[i]),
                    zone_high=float(lows[third]),
                    kind="bullish",
                )
            )
        elif highs[third] < lows[i] and (lows[i] - highs[third]) >= min_gap_size:
            zones.append(
                FVGZone(
                    index_pos=i + 1,
                    confirmed_pos=third,
                    timestamp=index[i + 1],
                    confirmed_at=index[third],
                    zone_low=float(highs[third]),
                    zone_high=float(lows[i]),
                    kind="bearish",
                )
            )

    return zones


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Classic Bollinger Bands: a simple moving average middle band, and
    upper/lower bands `num_std` sample standard deviations away.

    Causal: `middle`/`upper`/`lower` at row i depend only on
    `close[i-period+1:i+1]` (pandas' `rolling` is backward-looking by
    construction). NaN for the first `period - 1` bars, where the
    rolling window is not yet full - callers must not treat NaN as
    "band is at price" (a common bug when NaN silently compares as
    False in `close >= NaN`).

    Uses the population/sample standard deviation via
    `Series.rolling(...).std()` (pandas default: ddof=1, sample std) -
    the standard convention for Bollinger Bands.
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"bb_middle": middle, "bb_upper": upper, "bb_lower": lower})


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    smooth_k: int = 3,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D), the standard "slow stochastic"
    formulation: a raw %K from the `k_period`-bar high/low range,
    itself smoothed by `smooth_k` (this smoothed value is what most
    charting platforms display as "%K"), then %D is `d_period`'s
    rolling mean of that smoothed %K.

    Causal for the same reason as `bollinger_bands`: every rolling
    window looks backward from row i only. Returns NaN until enough
    bars exist for the full `k_period + smooth_k + d_period - 2`
    warmup.

    When the `k_period`-bar high/low range is exactly zero (a fully flat
    market — every OHLC identical), raw %K is undefined (0/0); this is
    left as NaN rather than defaulting to 50 or 100, matching this
    codebase's "undefined statistics remain undefined" convention.
    """
    if k_period < 1 or smooth_k < 1 or d_period < 1:
        raise ValueError("k_period, smooth_k, and d_period must all be >= 1")

    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    rng = highest_high - lowest_low

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_k = 100.0 * (close - lowest_low) / rng
    raw_k = raw_k.where(rng != 0.0)  # 0/0 -> NaN, not 0 or 100

    k = raw_k.rolling(window=smooth_k).mean()
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range: max(high-low, |high-prev_close|, |low-prev_close|).
    The first bar has no previous close, so TR there is just high-low."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing recurrence, shared by ATR/ADX: the first
    smoothed value is a simple mean of the first `period` raw values;
    every value after that is `smoothed[t] = smoothed[t-1] - smoothed[t-1]/period + raw[t]`
    (equivalent to, but computed without repeated division, the classic
    `(prev*(period-1) + raw[t])/period` recurrence used in `rsi()`
    above). Returns NaN before the window is full, matching `rsi()`."""
    n = len(values)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    out[period - 1] = np.nanmean(values[:period])
    for t in range(period, n):
        out[t] = (out[t - 1] * (period - 1) + values[t]) / period
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed (see `_wilder_smooth`) - used
    here as a volatility-scaled stop-loss buffer, not a signal
    generator in its own right. Causal and NaN for the first `period`
    bars, same warmup convention as `rsi()`."""
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(high, low, close).to_numpy(dtype=float)
    smoothed = _wilder_smooth(tr, period)
    return pd.Series(smoothed, index=high.index, name=f"atr_{period}")


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Wilder's Average Directional Index: +DI/-DI (directional
    movement, Wilder-smoothed) and ADX (Wilder-smoothed DX, the
    normalized difference between +DI and -DI). Used here purely as a
    trend-strength filter - low ADX indicates a non-trending ("ranging")
    market, which is the objective, mechanical condition
    `bb_stoch_mean_reversion` requires before considering mean-reversion
    entries (per its module docstring: mean reversion is unsound logic
    in a trending market, where price can walk one band for an extended
    period without ever reverting).

    Causal (each value depends on bars <= i only) and NaN for roughly
    the first `2 * period` bars, since ADX itself is a smoothed
    function of DX, which is itself a smoothed function of raw
    directional movement - a longer warmup than `rsi()`/`atr()`, and
    expected for this indicator.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    h = high.to_numpy(dtype=float)
    low_arr = low.to_numpy(dtype=float)
    n = len(h)

    up_move = np.diff(h, prepend=np.nan)
    down_move = -np.diff(low_arr, prepend=np.nan)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan

    tr = true_range(high, low, close).to_numpy(dtype=float)

    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
        minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0.0, 0.0, dx)  # both DI zero -> no directional movement at all -> DX 0, not undefined

    adx_values = _wilder_smooth(dx, period)

    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "adx": adx_values},
        index=high.index,
    )
