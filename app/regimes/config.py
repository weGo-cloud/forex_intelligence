"""Configuration for `app.regimes.classifier.RegimeClassifier`.

Every threshold the classifier uses lives here, not scattered as magic
numbers through `detectors.py`/`classifier.py`. All defaults are
DOCUMENTED GUESSES, not values tuned against historical performance -
per the M4 mission (section 8/9): "Do not optimize these thresholds
against historical performance in M4... Future research milestones may
evaluate and optimize parameters using proper out-of-sample
methodology." Treat every default below as a reasonable, literature-
typical starting point that a future M3 research workflow (parameter
sweep / robustness analysis - both already exist) should evaluate
before being relied on for anything consequential.

Multiple configs can coexist side by side
(`RegimeClassifier(config=A)` vs `RegimeClassifier(config=B)`) since
this is a plain immutable dataclass, not global state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeClassifierConfig:
    # --- Trend ---
    trend_fast_ma_period: int = 20
    trend_slow_ma_period: int = 50
    trend_slope_lookback: int = 10
    trend_directional_return_period: int = 20
    trend_adx_period: int = 14
    # ADX below this = "not enough directional strength to call a trend
    # either way", regardless of what the MA/slope/DI votes say -
    # standard-ish ADX convention (values below ~20-25 are commonly
    # read as "non-trending"), NOT derived from backtesting this
    # project's instruments.
    trend_adx_threshold: float = 20.0
    # Of the 3 directional votes (price vs slow MA, MA slope sign,
    # +DI vs -DI), how many must agree for a direction to be called.
    trend_min_agreeing_votes: int = 2

    # --- Volatility ---
    volatility_atr_period: int = 14
    # Rolling window the percentile rank (see
    # app.features.volatility.rolling_percentile_rank) is computed
    # over - i.e. "high/low relative to the last N bars", not to all
    # history. Larger = more stable but slower to reflect regime
    # changes; smaller = more reactive but noisier.
    volatility_percentile_window: int = 100
    volatility_low_percentile: float = 25.0
    volatility_high_percentile: float = 75.0

    # --- Structure ---
    structure_swing_left_bars: int = 3
    structure_swing_right_bars: int = 3
    # How many of the most recent labeled swings feed
    # structural_trend_strength - see
    # app.features.structure.structural_trend_strength_series.
    structure_swing_lookback_count: int = 4
    structure_range_period: int = 20
    structure_range_baseline_period: int = 100
    # abs(structural_trend_strength) at/above this = "swings are
    # consistently making the same kind of higher/lower pattern".
    structure_trend_strength_threshold: float = 0.5
    # range_compression_ratio at/below this = "current range is
    # genuinely smaller than its own recent history", not just below 1.0
    # by noise.
    structure_compression_threshold: float = 0.85
    # range_compression_ratio at/above this = "current range is
    # genuinely larger than its own recent history".
    structure_expansion_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.trend_fast_ma_period < 1 or self.trend_slow_ma_period < 1:
            raise ValueError("MA periods must be >= 1")
        if self.trend_fast_ma_period >= self.trend_slow_ma_period:
            raise ValueError("trend_fast_ma_period must be < trend_slow_ma_period")
        if self.trend_slope_lookback < 1:
            raise ValueError("trend_slope_lookback must be >= 1")
        if self.trend_directional_return_period < 1:
            raise ValueError("trend_directional_return_period must be >= 1")
        if self.trend_adx_period < 1:
            raise ValueError("trend_adx_period must be >= 1")
        if self.trend_adx_threshold < 0:
            raise ValueError("trend_adx_threshold must be >= 0")
        if not (1 <= self.trend_min_agreeing_votes <= 3):
            raise ValueError("trend_min_agreeing_votes must be between 1 and 3")

        if self.volatility_atr_period < 1:
            raise ValueError("volatility_atr_period must be >= 1")
        if self.volatility_percentile_window < 2:
            raise ValueError("volatility_percentile_window must be >= 2")
        if not (0 <= self.volatility_low_percentile < self.volatility_high_percentile <= 100):
            raise ValueError("require 0 <= volatility_low_percentile < volatility_high_percentile <= 100")

        if self.structure_swing_left_bars < 1 or self.structure_swing_right_bars < 1:
            raise ValueError("structure_swing_left_bars/right_bars must both be >= 1")
        if self.structure_swing_lookback_count < 1:
            raise ValueError("structure_swing_lookback_count must be >= 1")
        if self.structure_range_period < 2 or self.structure_range_baseline_period < 2:
            raise ValueError("structure_range_period and structure_range_baseline_period must both be >= 2")
        if not (0 <= self.structure_trend_strength_threshold <= 1):
            raise ValueError("structure_trend_strength_threshold must be between 0 and 1")
        if self.structure_compression_threshold <= 0:
            raise ValueError("structure_compression_threshold must be > 0")
        if self.structure_expansion_threshold <= 0:
            raise ValueError("structure_expansion_threshold must be > 0")
