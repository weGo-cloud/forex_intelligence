"""Bollinger-Stochastic Mean Reversion.

RULE (single execution timeframe, no higher-timeframe context needed):

1. Ranging-market filter: ADX(adx_period) on the execution timeframe
   must be BELOW `adx_ranging_threshold` at bar i. Mean reversion
   assumes price oscillates around a stable mean; in a trending market
   (high ADX) price can walk along one Bollinger Band for an extended
   period without ever reverting, which is exactly the case this filter
   exists to exclude. This mirrors the user's own "not used in trending
   markets but ranging markets" requirement with a standard, objective,
   mechanical trend-strength measure rather than a vague "does this
   look choppy" judgement call.

2. LONG setup: at bar i,
   - close was at or below the lower Bollinger Band at some point in the
     last `band_lookback_bars` bars (inclusive of bar i) - the
     "oversold, volatility-extended" location condition
   - stoch_k was at or below `stoch_oversold` at some point in the
     `band_lookback_bars` bars before bar i - the oscillator was in the
     oversold zone recently (see `band_lookback_bars` field docstring
     for why "recently" rather than requiring the exact same bar - a
     band touch and the momentum turn rarely land on one bar in real
     price action)
   - stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i]  (a %K/%D
     bullish crossover confirmed exactly at bar i - "if %K crosses above
     %D we enter trade", per the request)
   - ranging[i] is True

   SHORT setup is the mirror image: a recent upper-band touch, a recent
   overbought reading, a bearish %K/%D crossover
   (stoch_k[i-1] >= stoch_d[i-1] and stoch_k[i] < stoch_d[i]) at bar i,
   and ranging[i].

3. Stop-loss: placed just beyond the two-bar swing extreme in the
   trade's risk direction, buffered by `stop_atr_mult * ATR` so the stop
   isn't sitting exactly on a level price action just touched (a
   volatility-scaled buffer rather than an arbitrary fixed number of
   points, since the same buffer in points means something different on
   a quiet day vs. a volatile one).
   - long:  stop = min(low[i], low[i-1]) - stop_atr_mult * atr[i]
   - short: stop = max(high[i], high[i-1]) + stop_atr_mult * atr[i]

4. Take-profit: the middle Bollinger Band (the SMA) at bar i - "back to
   the mean" is the literal definition of this trade's thesis, so the
   target is that specific level rather than a fixed R-multiple (using
   the `Signal.take_profit` override - see app/strategies/base.py).

Every signal is generated purely from `execution_df`; `market.context`
is unused (this strategy runs on one timeframe only).

CAUSALITY: `bollinger_bands`, `stochastic_oscillator`, and `adx` are all
backward-looking rolling/recursive computations (see
app/features/indicators.py docstrings) - bar i's value never depends on
bar i+1 or later. The crossover check itself only compares bars i-1 and
i. `signal_time` is bar i's own timestamp; the engine fills at the next
bar's open (see app/strategies/base.py::Signal docstring), so nothing
here is ever executed at a price that required knowing bar i's own
close in advance beyond what's already true of every strategy in this
codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.features.indicators import adx, atr, bollinger_bands, stochastic_oscillator
from app.strategies.base import MarketContext, Signal
from app.strategies.registry import register_strategy


@dataclass(frozen=True)
class BBStochMeanReversionConfig:
    instrument_symbol: str

    bb_period: int = 20
    bb_num_std: float = 2.0

    stoch_k_period: int = 14
    stoch_smooth_k: int = 3
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # A bottom (or top) and the %K/%D crossover confirming it rarely land
    # on the exact same bar in real price action - price often touches
    # the band a bar or two before momentum visibly turns. band_lookback_bars
    # is how many bars back (inclusive of the crossover bar itself) to look
    # for "price was at/beyond the band" / "%K was in the oversold/
    # overbought zone", rather than requiring same-bar coincidence. Still
    # fully mechanical and causal (a backward-looking rolling window, no
    # different in kind from the indicators' own rolling windows) - not a
    # "does this look like a bottom" visual judgement.
    band_lookback_bars: int = 3

    adx_period: int = 14
    adx_ranging_threshold: float = 20.0

    atr_period: int = 14
    stop_atr_mult: float = 1.0

    def __post_init__(self) -> None:
        if self.bb_period < 2:
            raise ValueError("bb_period must be >= 2")
        if self.bb_num_std <= 0:
            raise ValueError("bb_num_std must be > 0")
        if self.stoch_k_period < 1 or self.stoch_smooth_k < 1 or self.stoch_d_period < 1:
            raise ValueError("stoch_k_period, stoch_smooth_k, and stoch_d_period must all be >= 1")
        if self.band_lookback_bars < 1:
            raise ValueError("band_lookback_bars must be >= 1")
        if not (0 <= self.stoch_oversold < self.stoch_overbought <= 100):
            raise ValueError("require 0 <= stoch_oversold < stoch_overbought <= 100")
        if self.adx_period < 1:
            raise ValueError("adx_period must be >= 1")
        if self.adx_ranging_threshold <= 0:
            raise ValueError("adx_ranging_threshold must be > 0")
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")


class BBStochMeanReversionStrategy:
    def __init__(self, config: BBStochMeanReversionConfig) -> None:
        self.config = config
        self.name = "bb_stoch_mean_reversion"

    def params(self) -> dict[str, Any]:
        return {
            "instrument_symbol": self.config.instrument_symbol,
            "bb_period": self.config.bb_period,
            "bb_num_std": self.config.bb_num_std,
            "stoch_k_period": self.config.stoch_k_period,
            "stoch_smooth_k": self.config.stoch_smooth_k,
            "stoch_d_period": self.config.stoch_d_period,
            "stoch_oversold": self.config.stoch_oversold,
            "stoch_overbought": self.config.stoch_overbought,
            "band_lookback_bars": self.config.band_lookback_bars,
            "adx_period": self.config.adx_period,
            "adx_ranging_threshold": self.config.adx_ranging_threshold,
            "atr_period": self.config.atr_period,
            "stop_atr_mult": self.config.stop_atr_mult,
        }

    def generate_signals(self, market: MarketContext) -> list[Signal]:
        cfg = self.config
        df = market.execution_df
        if df.empty:
            return []

        bands = bollinger_bands(df["close"], period=cfg.bb_period, num_std=cfg.bb_num_std)
        stoch = stochastic_oscillator(
            df["high"], df["low"], df["close"],
            k_period=cfg.stoch_k_period, smooth_k=cfg.stoch_smooth_k, d_period=cfg.stoch_d_period,
        )
        adx_df = adx(df["high"], df["low"], df["close"], period=cfg.adx_period)
        atr_series = atr(df["high"], df["low"], df["close"], period=cfg.atr_period)

        k, d = stoch["stoch_k"], stoch["stoch_d"]
        k_prev, d_prev = k.shift(1), d.shift(1)
        bullish_cross = (k_prev <= d_prev) & (k > d)
        bearish_cross = (k_prev >= d_prev) & (k < d)

        ranging = adx_df["adx"] < cfg.adx_ranging_threshold

        # "Recently at the band" / "recently in the oversold/overbought zone" -
        # see band_lookback_bars docstring above for why this isn't a
        # same-bar-only check.
        lookback = cfg.band_lookback_bars
        recent_lower_touch = (df["close"] <= bands["bb_lower"]).rolling(window=lookback, min_periods=1).max().astype(bool)
        recent_upper_touch = (df["close"] >= bands["bb_upper"]).rolling(window=lookback, min_periods=1).max().astype(bool)
        recent_oversold = (k <= cfg.stoch_oversold).shift(1).rolling(window=lookback, min_periods=1).max().fillna(0).astype(bool)
        recent_overbought = (k >= cfg.stoch_overbought).shift(1).rolling(window=lookback, min_periods=1).max().fillna(0).astype(bool)

        long_setup = ranging & recent_lower_touch & recent_oversold & bullish_cross
        short_setup = ranging & recent_upper_touch & recent_overbought & bearish_cross

        signals: list[Signal] = []
        low_prev, high_prev = df["low"].shift(1), df["high"].shift(1)

        for i, ts in enumerate(df.index):
            if i == 0:
                continue  # every condition needs a "prev" bar
            if bool(long_setup.iloc[i]):
                stop = min(df["low"].iloc[i], low_prev.iloc[i]) - cfg.stop_atr_mult * atr_series.iloc[i]
                target = bands["bb_middle"].iloc[i]
                if pd.isna(stop) or pd.isna(target):
                    continue
                signals.append(
                    Signal(
                        signal_time=ts,
                        direction="long",
                        stop_loss=float(stop),
                        take_profit=float(target),
                        strategy_name=self.name,
                        meta={"adx": float(adx_df["adx"].iloc[i]), "stoch_k": float(k.iloc[i])},
                    )
                )
            elif bool(short_setup.iloc[i]):
                stop = max(df["high"].iloc[i], high_prev.iloc[i]) + cfg.stop_atr_mult * atr_series.iloc[i]
                target = bands["bb_middle"].iloc[i]
                if pd.isna(stop) or pd.isna(target):
                    continue
                signals.append(
                    Signal(
                        signal_time=ts,
                        direction="short",
                        stop_loss=float(stop),
                        take_profit=float(target),
                        strategy_name=self.name,
                        meta={"adx": float(adx_df["adx"].iloc[i]), "stoch_k": float(k.iloc[i])},
                    )
                )

        return signals


@register_strategy(
    "bb_stoch_mean_reversion",
    BBStochMeanReversionConfig,
    "Bollinger Bands + Stochastic %K/%D crossover mean reversion, filtered to ranging (low-ADX) markets only",
)
def _build_bb_stoch_mean_reversion(**kwargs) -> BBStochMeanReversionStrategy:
    return BBStochMeanReversionStrategy(BBStochMeanReversionConfig(**kwargs))
