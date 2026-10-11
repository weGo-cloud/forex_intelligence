"""Reusable volatility MEASUREMENTS (not interpretations).

`app.features.indicators.atr` already provides raw ATR and is reused
as-is - this module only adds what raw ATR doesn't give: a
price-normalized version (comparable across instruments, since raw ATR
in price units means something different for EURUSD vs USDJPY vs a
crypto pair) and a simple relative "is this stretched vs its own recent
history" ratio.

Same causality contract as `app.features.trend`: backward-looking only,
NaN through warm-up, never a fabricated value.
"""

from __future__ import annotations

import pandas as pd


def normalized_atr(atr: pd.Series, close: pd.Series) -> pd.Series:
    """ATR expressed as a fraction of price (`atr / close`), the
    standard way to make ATR comparable across instruments/price
    levels - per this module's docstring, raw ATR alone should never be
    used to call a market "high" or "low" volatility. Causal (same-bar
    ratio of two already-causal series). NaN wherever `atr` is NaN
    (its own warm-up) or `close` is exactly 0 (a data-quality problem
    this module does not attempt to paper over - forex closes are never
    legitimately 0, so this should never fire on real M1 data).
    """
    with_zero_guard = close.where(close != 0.0)
    return atr / with_zero_guard


def rolling_return_volatility(close: pd.Series, period: int) -> pd.Series:
    """Rolling standard deviation of simple bar-to-bar returns over
    `period` bars - a second, independent volatility measurement
    alongside ATR (ATR captures intrabar range; this captures close-to-
    close dispersion, which can diverge from ATR e.g. in gappy markets).
    Causal (`rolling` + `pct_change` both backward-looking). NaN for the
    first `period` bars (plus one, for `pct_change`'s own warm-up).
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    returns = close.pct_change()
    return returns.rolling(window=period).std()


def volatility_ratio(normalized_atr_series: pd.Series, baseline_period: int) -> pd.Series:
    """Current normalized ATR relative to its own rolling mean over
    `baseline_period` bars: > 1.0 means volatility is expanding
    relative to its recent history, < 1.0 means contracting. This is
    the *relative* signal the regime classifier actually uses for
    LOW/NORMAL/HIGH (see app/regimes/detectors.py) - deliberately
    self-referential (an instrument's volatility compared only to its
    own past) rather than any fixed absolute threshold, so it adapts
    per instrument without configuration. Causal. NaN for the first
    `baseline_period` bars beyond `normalized_atr_series`'s own warm-up,
    and wherever the rolling mean is exactly 0.
    """
    if baseline_period < 2:
        raise ValueError("baseline_period must be >= 2")
    baseline = normalized_atr_series.rolling(window=baseline_period).mean()
    baseline_guarded = baseline.where(baseline != 0.0)
    return normalized_atr_series / baseline_guarded


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """The percentile rank (0-100) of `series[i]` within the trailing
    `window` bars ending at and including bar i - the mechanism behind
    the regime classifier's LOW/NORMAL/HIGH volatility thresholds
    (configurable percentile cutoffs, see app/regimes/config.py),
    rather than a hard-coded absolute value. Causal: the window is
    `[i - window + 1, i]`, never anything at or after `i + 1`. NaN for
    the first `window - 1` bars (requires a full window - a percentile
    rank computed from a partial, growing window would mean different
    things at different bars, which defeats the point of a rank).
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    def _rank_of_last(values):
        last = values[-1]
        if pd.isna(last):
            return float("nan")
        valid = values[~pd.isna(values)]
        if len(valid) == 0:
            return float("nan")
        return float((valid <= last).sum()) / len(valid) * 100.0

    return series.rolling(window=window, min_periods=window).apply(_rank_of_last, raw=True)
