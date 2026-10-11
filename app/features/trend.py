"""Reusable trend MEASUREMENTS (not interpretations - see app/regimes/
for the layer that turns these into a TrendState).

Trend strength/direction evidence itself (ADX, +DI/-DI) already exists
in `app.features.indicators.adx` and is reused as-is, not duplicated
here - this module only adds the measurements ADX doesn't provide:
moving averages, MA slope, and simple directional return.

Every function here follows the same causality contract as
`app.features.indicators`: row i depends only on rows <= i (pandas
`rolling`/`shift`/`diff` are backward-looking by construction), and
values are NaN until each function's own warm-up window is full -
never a fabricated 0 or midpoint value.
"""

from __future__ import annotations

import pandas as pd


def moving_average(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average. Causal (`rolling` looks backward only).
    NaN for the first `period - 1` bars (warm-up).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    return close.rolling(window=period).mean()


def ma_alignment(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """+1.0 where `fast_ma > slow_ma` (bullish alignment), -1.0 where
    `fast_ma < slow_ma` (bearish), 0.0 where exactly equal. NaN
    wherever either input is NaN (both series' own warm-up periods
    propagate through). Purely a same-bar comparison of two
    already-causal series, so this itself introduces no new lookahead
    and no new warm-up beyond its inputs'.
    """
    diff = fast_ma - slow_ma
    return diff.apply(lambda x: float("nan") if pd.isna(x) else (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)))


def ma_slope(ma: pd.Series, lookback: int) -> pd.Series:
    """Fractional change in a moving-average series over `lookback`
    bars: `(ma[i] - ma[i-lookback]) / ma[i-lookback]`. Causal
    (`Series.diff`/indexing only look backward). NaN for the first
    `lookback` bars beyond `ma`'s own warm-up, and wherever `ma[i-lookback]`
    is exactly 0 (division is undefined, not silently treated as a huge
    or zero slope).
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    prior = ma.shift(lookback)
    slope = (ma - prior) / prior
    return slope.replace([float("inf"), float("-inf")], float("nan"))


def directional_return(close: pd.Series, period: int) -> pd.Series:
    """Simple percentage return over `period` bars:
    `(close[i] - close[i-period]) / close[i-period]`. Causal. NaN for
    the first `period` bars.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    return close.pct_change(periods=period)
