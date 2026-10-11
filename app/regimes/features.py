"""Computes every raw feature the regime classifier needs, once, over a
whole OHLC frame - vectorized (pandas rolling/shift operations), not
recomputed per-bar. `app.regimes.detectors` then reads individual
values out of this at a given bar position; `app.regimes.classifier`
owns calling this once per `compute_feature_set` call.

This module contains no interpretation/thresholds at all - it is pure
measurement, same spirit as `app.features.trend`/`volatility`/
`structure` themselves (which it is entirely built from).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.features.indicators import SwingPoint, adx, atr, detect_swing_points
from app.features.structure import range_compression_ratio, structural_trend_strength_series
from app.features.trend import directional_return, ma_alignment, ma_slope, moving_average
from app.features.volatility import normalized_atr, rolling_percentile_rank, rolling_return_volatility
from app.regimes.config import RegimeClassifierConfig


@dataclass(frozen=True)
class RegimeFeatureSet:
    close: pd.Series
    high: pd.Series
    low: pd.Series

    fast_ma: pd.Series
    slow_ma: pd.Series
    ma_alignment: pd.Series
    slow_ma_slope: pd.Series
    directional_return: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series
    trend_adx: pd.Series

    atr: pd.Series
    normalized_atr: pd.Series
    return_volatility: pd.Series
    volatility_percentile: pd.Series

    swings: list[SwingPoint]
    structural_trend_strength: pd.Series
    range_compression_ratio: pd.Series

    def feature_values_at(self, position: int) -> dict[str, float]:
        """A JSON-safe (NaN -> None) snapshot of every scalar feature at
        a given bar position - used to populate `RegimeContext.feature_values`."""

        def _get(series: pd.Series) -> float | None:
            value = series.iloc[position]
            return None if pd.isna(value) else float(value)

        return {
            "fast_ma": _get(self.fast_ma),
            "slow_ma": _get(self.slow_ma),
            "ma_alignment": _get(self.ma_alignment),
            "slow_ma_slope": _get(self.slow_ma_slope),
            "directional_return": _get(self.directional_return),
            "plus_di": _get(self.plus_di),
            "minus_di": _get(self.minus_di),
            "trend_adx": _get(self.trend_adx),
            "atr": _get(self.atr),
            "normalized_atr": _get(self.normalized_atr),
            "return_volatility": _get(self.return_volatility),
            "volatility_percentile": _get(self.volatility_percentile),
            "structural_trend_strength": _get(self.structural_trend_strength),
            "range_compression_ratio": _get(self.range_compression_ratio),
        }


def compute_feature_set(df: pd.DataFrame, config: RegimeClassifierConfig) -> RegimeFeatureSet:
    """Compute every raw feature over the whole frame `df` (must have
    `open`/`high`/`low`/`close` columns - `open` is not itself used but
    required for consistency with the rest of the codebase's OHLC
    frame convention). Raises on an empty frame or missing columns
    rather than silently producing an empty/garbage feature set - the
    regime engine does not repair invalid input, per the M4 mission
    (section 17): "The data layer owns data validation."
    """
    required_columns = {"high", "low", "close"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")
    if df.empty:
        raise ValueError("df must not be empty")

    close, high, low = df["close"], df["high"], df["low"]

    fast_ma = moving_average(close, config.trend_fast_ma_period)
    slow_ma = moving_average(close, config.trend_slow_ma_period)
    alignment = ma_alignment(fast_ma, slow_ma)
    slope = ma_slope(slow_ma, config.trend_slope_lookback)
    d_return = directional_return(close, config.trend_directional_return_period)

    adx_df = adx(high, low, close, period=config.trend_adx_period)

    atr_series = atr(high, low, close, period=config.volatility_atr_period)
    norm_atr = normalized_atr(atr_series, close)
    return_vol = rolling_return_volatility(close, period=config.trend_slope_lookback)
    vol_percentile = rolling_percentile_rank(norm_atr, window=config.volatility_percentile_window)

    swings = detect_swing_points(
        high, low, left_bars=config.structure_swing_left_bars, right_bars=config.structure_swing_right_bars
    )
    structural_strength = structural_trend_strength_series(df.index, swings, config.structure_swing_lookback_count)
    range_ratio = range_compression_ratio(
        high, low, close, period=config.structure_range_period, baseline_period=config.structure_range_baseline_period
    )

    return RegimeFeatureSet(
        close=close,
        high=high,
        low=low,
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        ma_alignment=alignment,
        slow_ma_slope=slope,
        directional_return=d_return,
        plus_di=adx_df["plus_di"],
        minus_di=adx_df["minus_di"],
        trend_adx=adx_df["adx"],
        atr=atr_series,
        normalized_atr=norm_atr,
        return_volatility=return_vol,
        volatility_percentile=vol_percentile,
        swings=swings,
        structural_trend_strength=structural_strength,
        range_compression_ratio=range_ratio,
    )
