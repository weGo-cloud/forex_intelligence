"""RSI Divergence strategy.

Objective definitions (per the "no vague visual logic" requirement):

- Swing highs/lows are detected on PRICE using
  `app.features.indicators.detect_swing_points` (a mechanical fractal
  rule: strictly the highest/lowest value in a `left_bars`/`right_bars`
  window). RSI's value AT those same price-pivot bars is what's compared
  — this is the standard operational definition of price/RSI divergence
  (compare the oscillator's reading at price's own turning points),
  rather than independently detecting RSI's own local extrema, which
  would require reconciling two different sets of pivot timestamps for
  no added rigor.
- Bearish divergence: two confirmed swing HIGHS at bar positions i1 < i2
  (i2 - i1 >= min_pivot_separation) where price[i2] > price[i1] (higher
  high) and rsi[i2] < rsi[i1] (lower high) -> short signal.
- Bullish divergence: symmetric on swing LOWS -> long signal.

Causality: a divergence signal can only be emitted once BOTH pivots are
confirmed, i.e. at `max(confirmed_pos_i1, confirmed_pos_i2)` — which for
two ascending pivots is always `confirmed_pos_i2` (see
`detect_swing_points` docstring for what "confirmed" means and why).
The engine then fills at the next bar's open, same policy as ORB.

Optional HTF trend filter: if `htf_timeframe` is set, a divergence is
only taken when it agrees with a simple HTF trend read (HTF close vs.
HTF SMA). This is DISABLED by default (`htf_filter_enabled=False`)
because the roadmap does not specify what "HTF context" should actually
do here, and inventing an unstated filter rule and shipping it as the
default would be exactly the kind of hidden rule the spec warns against.
The mechanism exists so a researcher can opt in and test it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandas as pd

from app.features.indicators import detect_swing_points, rsi
from app.strategies.base import MarketContext, Signal
from app.strategies.registry import register_strategy


@dataclass(frozen=True)
class RSIDivergenceConfig:
    instrument_symbol: str
    rsi_period: int = 14
    rsi_source: Literal["close"] = "close"
    swing_left_bars: int = 3
    swing_right_bars: int = 3
    min_pivot_separation: int = 5
    stop_loss_method: Literal["swing_extreme"] = "swing_extreme"
    stop_loss_buffer: float = 0.0
    risk_reward: float = 2.0
    max_trades_per_day: Optional[int] = None
    htf_timeframe: Optional[str] = None  # e.g. "1H"; key into MarketContext.context
    htf_filter_enabled: bool = False
    htf_sma_period: int = 20

    def __post_init__(self) -> None:
        if self.rsi_period < 2:
            raise ValueError("rsi_period must be >= 2")
        if self.swing_left_bars < 1 or self.swing_right_bars < 1:
            raise ValueError("swing_left_bars and swing_right_bars must both be >= 1")
        if self.min_pivot_separation < 1:
            raise ValueError("min_pivot_separation must be >= 1")
        if self.risk_reward <= 0:
            raise ValueError("risk_reward must be > 0")
        if self.htf_filter_enabled and not self.htf_timeframe:
            raise ValueError("htf_filter_enabled requires htf_timeframe to be set")


class RSIDivergenceStrategy:
    """RSI price divergence. See module docstring for the exact rule."""

    def __init__(self, config: RSIDivergenceConfig) -> None:
        self.config = config
        self.name = "rsi_divergence"

    def params(self) -> dict[str, Any]:
        return {
            "instrument_symbol": self.config.instrument_symbol,
            "rsi_period": self.config.rsi_period,
            "rsi_source": self.config.rsi_source,
            "swing_left_bars": self.config.swing_left_bars,
            "swing_right_bars": self.config.swing_right_bars,
            "min_pivot_separation": self.config.min_pivot_separation,
            "stop_loss_method": self.config.stop_loss_method,
            "stop_loss_buffer": self.config.stop_loss_buffer,
            "risk_reward": self.config.risk_reward,
            "max_trades_per_day": self.config.max_trades_per_day,
            "htf_timeframe": self.config.htf_timeframe,
            "htf_filter_enabled": self.config.htf_filter_enabled,
            "htf_sma_period": self.config.htf_sma_period,
        }

    def generate_signals(self, market: MarketContext) -> list[Signal]:
        df = market.execution_df
        cfg = self.config
        if len(df) < cfg.rsi_period + cfg.swing_left_bars + cfg.swing_right_bars + 1:
            return []

        rsi_series = rsi(df["close"], period=cfg.rsi_period)
        swings = detect_swing_points(df["high"], df["low"], cfg.swing_left_bars, cfg.swing_right_bars)

        htf_trend_up: Optional[pd.Series] = None
        if cfg.htf_filter_enabled and cfg.htf_timeframe:
            htf_df = market.context.get(cfg.htf_timeframe)
            if htf_df is not None and not htf_df.empty:
                close_col = f"close_{cfg.htf_timeframe}" if f"close_{cfg.htf_timeframe}" in htf_df.columns else "close_htf"
                if close_col in htf_df.columns:
                    htf_close = htf_df[close_col].reindex(df.index).ffill()
                    htf_sma = htf_close.rolling(cfg.htf_sma_period).mean()
                    htf_trend_up = htf_close > htf_sma

        highs = [p for p in swings if p.kind == "high"]
        lows = [p for p in swings if p.kind == "low"]

        signals: list[Signal] = []
        signals += self._scan_pairs(highs, rsi_series, df, "short", htf_trend_up)
        signals += self._scan_pairs(lows, rsi_series, df, "long", htf_trend_up)
        signals.sort(key=lambda s: s.signal_time)

        if cfg.max_trades_per_day is not None:
            signals = self._apply_daily_cap(signals)

        return signals

    def _scan_pairs(self, pivots, rsi_series: pd.Series, df: pd.DataFrame, direction: str, htf_trend_up):
        cfg = self.config
        out: list[Signal] = []
        for j in range(1, len(pivots)):
            p2 = pivots[j]
            # find the most recent earlier pivot of the same kind satisfying separation
            candidates = [p1 for p1 in pivots[:j] if p2.index_pos - p1.index_pos >= cfg.min_pivot_separation]
            if not candidates:
                continue
            p1 = candidates[-1]

            rsi_p1 = rsi_series.iloc[p1.index_pos]
            rsi_p2 = rsi_series.iloc[p2.index_pos]
            if pd.isna(rsi_p1) or pd.isna(rsi_p2):
                continue

            confirmed_pos = p2.confirmed_pos  # later pivot's confirmation always dominates for ascending pairs
            if confirmed_pos >= len(df):
                continue
            confirmed_ts = df.index[confirmed_pos]

            if direction == "short":
                # bearish divergence: higher high in price, lower high in RSI
                is_divergence = p2.price > p1.price and rsi_p2 < rsi_p1
                stop_loss = p2.price + cfg.stop_loss_buffer
            else:
                # bullish divergence: lower low in price, higher low in RSI
                is_divergence = p2.price < p1.price and rsi_p2 > rsi_p1
                stop_loss = p2.price - cfg.stop_loss_buffer

            if not is_divergence:
                continue

            if htf_trend_up is not None and confirmed_ts in htf_trend_up.index:
                trend_up = bool(htf_trend_up.loc[confirmed_ts]) if pd.notna(htf_trend_up.loc[confirmed_ts]) else None
                if trend_up is not None:
                    if direction == "long" and not trend_up:
                        continue
                    if direction == "short" and trend_up:
                        continue

            out.append(
                Signal(
                    signal_time=confirmed_ts,
                    direction=direction,
                    stop_loss=float(stop_loss),
                    strategy_name=self.name,
                    meta={
                        "pivot1_price": p1.price,
                        "pivot1_rsi": float(rsi_p1),
                        "pivot1_time": str(p1.timestamp),
                        "pivot2_price": p2.price,
                        "pivot2_rsi": float(rsi_p2),
                        "pivot2_time": str(p2.timestamp),
                    },
                )
            )
        return out

    def _apply_daily_cap(self, signals: list[Signal]) -> list[Signal]:
        cfg = self.config
        counts: dict[Any, int] = {}
        kept: list[Signal] = []
        for s in signals:
            day = s.signal_time.date()
            counts[day] = counts.get(day, 0)
            if counts[day] >= cfg.max_trades_per_day:  # type: ignore[operator]
                continue
            counts[day] += 1
            kept.append(s)
        return kept


@register_strategy("rsi_divergence", RSIDivergenceConfig, "RSI price divergence - nearest-prior-pivot comparison")
def _build_rsi_divergence(**kwargs) -> RSIDivergenceStrategy:
    return RSIDivergenceStrategy(RSIDivergenceConfig(**kwargs))
