"""Liquidity Sweep + FVG (multi-timeframe: 1D bias, 1H NY-session
fakeout, 5M FVG/IFVG pullback entry).

Requires `market.context` to contain two higher-timeframe frames,
already causally aligned via `app.backtesting.data_access.align_higher_timeframe`
(see that function's docstring - each value only becomes visible from
the moment its own bar has actually closed):
  - `context["1D"]`: daily candles (`open_htf`, `high_htf`, `low_htf`, `close_htf`)
  - `context["1H"]`: 1-hour candles, same column shape

RULE (objective, no discretionary judgment):

1. DAILY BIAS: on each trading day, look at the most recently CLOSED
   daily candle (i.e. yesterday's, from `context["1D"]`). Bias is
   "bullish" if that candle's close > open, "bearish" if close < open.
   A flat daily candle (close == open) has no bias and the strategy
   does not trade that day. ASSUMPTION (source rule said only "1D
   candles to check direction bull/bearish" without specifying exactly
   how): bias is read directly off the prior day's own candle body,
   not compared against an earlier day - the simplest reading of the
   stated rule, and the one used consistently for both the sweep
   reference level and the profit target below.

2. NY SESSION FAKEOUT (1H): within `ny_session_start`-`ny_session_end`
   (`session_tz`, default America/New_York 08:00-11:00 - the NY AM
   session), watch each newly-CLOSED 1H candle (from `context["1H"]`)
   for a liquidity sweep against the daily bias:
   - bullish bias: the 1H candle's low < previous day's low, AND its
     close > previous day's low (it swept below yesterday's low - a
     move *opposite* the bullish bias - then closed back above it,
     i.e. rejected the sweep)
   - bearish bias: the mirror image against previous day's high
   The first such 1H candle each day arms the setup; its swept extreme
   (the 1H candle's low, for the bullish case) becomes the stop-loss
   reference.

3. FVG/IFVG PULLBACK (5M): once armed, scan 5-minute candles (via
   `app.features.indicators.detect_fvg`) for the first Fair Value Gap,
   CONFIRMED after the fakeout candle, whose direction matches the bias
   (a bullish FVG for a long setup, bearish for a short setup) - this
   is the Area of Interest (AOI). "IFVG" (inverse FVG) support: if a
   FVG of the OPPOSITE kind exists after the fakeout and price later
   closes fully through it (past its far boundary), that zone flips
   polarity and becomes eligible as an AOI in the bias direction too
   (an inverted bearish FVG becomes a bullish AOI, and vice versa) -
   see `_active_zones_at` for the exact, mechanical inversion check.
   The most recently confirmed eligible zone as of the current bar is
   used as the active AOI (a later-forming zone supersedes an earlier
   one that price hasn't yet reached).

4. ENTRY: the first 5M bar whose range pulls back into the active AOI
   (`low[i] <= zone_high` for a long AOI, `high[i] >= zone_low` for a
   short AOI) triggers a signal in the bias direction.
   - stop_loss: the fakeout candle's swept extreme (the 1H low it swept
     below, for a long; the 1H high it swept above, for a short) minus/
     plus `stop_buffer_points` (default 0.0 - configurable rather than
     an unstated hard-coded number).
   - take_profit: the previous day's level in the bias direction -
     previous day's HIGH for a long, previous day's LOW for a short
     (`Signal.take_profit`, see app/strategies/base.py - this strategy
     targets a specific level, not a fixed R-multiple).

5. At most `max_trades_per_day` signals (default 1) are emitted per
   trading day; the setup is not re-armed after a trade is taken or
   after `fvg_lookback_bars` 5-minute bars have passed since the
   fakeout without an entry (a stale setup is dropped, not carried
   forward indefinitely).

CAUSALITY: bias and fakeout detection use only already-closed HTF
candles (`align_higher_timeframe`'s own guarantee). FVG zones use only
`confirmed_pos <= current bar position` (see `detect_fvg`). Entry uses
only the current 5M bar's own high/low/close. `signal_time` is that
bar's timestamp; the engine fills at the next bar's open (see
`Signal` docstring) exactly as every other strategy in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any, Optional

import pandas as pd

from app.features.indicators import FVGZone, detect_fvg
from app.features.sessions import SessionWindow
from app.strategies.base import Direction, MarketContext, Signal
from app.strategies.registry import register_strategy

DAILY_CONTEXT_LABEL = "1D"
SESSION_CONTEXT_LABEL = "1H"


@dataclass(frozen=True)
class LiquiditySweepFVGConfig:
    instrument_symbol: str
    session_tz: str = "America/New_York"
    ny_session_start: time = time(8, 0)
    ny_session_end: time = time(11, 0)
    fvg_lookback_bars: int = 24  # 5-minute bars a fakeout stays "armed" while waiting for an AOI + entry
    max_trades_per_day: int = 1
    min_fvg_size_points: float = 0.0
    stop_buffer_points: float = 0.0

    def __post_init__(self) -> None:
        if self.ny_session_end <= self.ny_session_start:
            raise ValueError("ny_session_end must be after ny_session_start")
        if self.fvg_lookback_bars < 3:
            raise ValueError("fvg_lookback_bars must be >= 3 (an FVG needs 3 candles to exist at all)")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be >= 1")
        if self.min_fvg_size_points < 0:
            raise ValueError("min_fvg_size_points must be >= 0")
        if self.stop_buffer_points < 0:
            raise ValueError("stop_buffer_points must be >= 0")


@dataclass
class _DayState:
    bias: Optional[Direction]
    prev_day_high: Optional[float]
    prev_day_low: Optional[float]
    fakeout_found: bool = False
    fakeout_bar_pos: Optional[int] = None
    fakeout_extreme: Optional[float] = None
    trades_taken: int = 0
    last_seen_1h_ohlc: Optional[tuple] = None  # dedupe: same closed 1H candle visible across several 5M bars


def _active_zones_at(zones: list[FVGZone], position: int, close: pd.Series) -> list[FVGZone]:
    """Zones confirmed at or before `position`, in their EFFECTIVE
    (possibly inverted) form as of `position`. A zone inverts the first
    time, after its own confirmation, that `close` fully closes past its
    far boundary (below zone_low for a bullish zone, above zone_high for
    a bearish one) - from that bar onward it is treated as the opposite
    kind, same boundaries. Only depends on `close[<=position]`, so this
    is causal by construction.
    """
    result: list[FVGZone] = []
    for zone in zones:
        if zone.confirmed_pos > position:
            continue
        inverted = False
        # Only bars strictly after confirmation can invert a zone - the
        # confirming candle itself is what defined the zone.
        check_slice = close.iloc[zone.confirmed_pos + 1 : position + 1]
        if zone.kind == "bullish" and (check_slice < zone.zone_low).any():
            inverted = True
        elif zone.kind == "bearish" and (check_slice > zone.zone_high).any():
            inverted = True

        if not inverted:
            result.append(zone)
        else:
            flipped_kind = "bearish" if zone.kind == "bullish" else "bullish"
            result.append(
                FVGZone(
                    index_pos=zone.index_pos,
                    confirmed_pos=zone.confirmed_pos,
                    timestamp=zone.timestamp,
                    confirmed_at=zone.confirmed_at,
                    zone_low=zone.zone_low,
                    zone_high=zone.zone_high,
                    kind=flipped_kind,  # type: ignore[arg-type]
                )
            )
    return result


class LiquiditySweepFVGStrategy:
    """See module docstring for the exact rule."""

    def __init__(self, config: LiquiditySweepFVGConfig) -> None:
        self.config = config
        self.name = "liquidity_sweep_fvg"
        self._session = SessionWindow(config.session_tz, time(0, 0), time(23, 59, 59))
        self._ny_window = SessionWindow(config.session_tz, config.ny_session_start, config.ny_session_end)

    def params(self) -> dict[str, Any]:
        return {
            "instrument_symbol": self.config.instrument_symbol,
            "session_tz": self.config.session_tz,
            "ny_session_start": self.config.ny_session_start.isoformat(),
            "ny_session_end": self.config.ny_session_end.isoformat(),
            "fvg_lookback_bars": self.config.fvg_lookback_bars,
            "max_trades_per_day": self.config.max_trades_per_day,
            "min_fvg_size_points": self.config.min_fvg_size_points,
            "stop_buffer_points": self.config.stop_buffer_points,
        }

    def generate_signals(self, market: MarketContext) -> list[Signal]:
        cfg = self.config
        df = market.execution_df
        if df.empty:
            return []
        if DAILY_CONTEXT_LABEL not in market.context or SESSION_CONTEXT_LABEL not in market.context:
            raise ValueError(
                f"{self.name} requires market.context to include {DAILY_CONTEXT_LABEL!r} and "
                f"{SESSION_CONTEXT_LABEL!r} higher-timeframe frames (see module docstring)"
            )

        daily_ctx = market.context[DAILY_CONTEXT_LABEL]
        session_ctx = market.context[SESSION_CONTEXT_LABEL]
        zones = detect_fvg(df["high"], df["low"], min_gap_size=cfg.min_fvg_size_points)

        signals: list[Signal] = []
        state: Optional[_DayState] = None
        current_day = None

        for i, ts in enumerate(df.index):
            day = self._session.session_date(ts)
            if day != current_day:
                current_day = day
                bias, prev_high, prev_low = self._daily_bias_at(daily_ctx, ts)
                state = _DayState(bias=bias, prev_day_high=prev_high, prev_day_low=prev_low)

            assert state is not None
            if state.bias is None or state.trades_taken >= cfg.max_trades_per_day:
                continue

            local_time = ts.tz_convert(self._ny_window_tz())
            in_ny_window = cfg.ny_session_start <= local_time.time() < cfg.ny_session_end

            # --- 1H fakeout detection (only while still looking for one) ---
            if not state.fakeout_found and in_ny_window:
                open_1h = session_ctx["open_htf"].iloc[i] if "open_htf" in session_ctx else float("nan")
                close_1h = session_ctx["close_htf"].iloc[i] if "close_htf" in session_ctx else float("nan")
                low_1h = session_ctx["low_htf"].iloc[i] if "low_htf" in session_ctx else float("nan")
                high_1h = session_ctx["high_htf"].iloc[i] if "high_htf" in session_ctx else float("nan")
                current_1h_ohlc = (open_1h, high_1h, low_1h, close_1h)
                # Compare the FULL closed OHLC tuple, not just close, to detect
                # "a new 1H candle just became visible" - comparing close alone
                # would wrongly treat two different candles that happen to
                # share the same close as "the same candle" and never re-check.
                if pd.notna(close_1h) and current_1h_ohlc != state.last_seen_1h_ohlc:
                    state.last_seen_1h_ohlc = current_1h_ohlc
                    if state.bias == "long" and low_1h < state.prev_day_low and close_1h > state.prev_day_low:
                        state.fakeout_found = True
                        state.fakeout_bar_pos = i
                        state.fakeout_extreme = float(low_1h)
                    elif state.bias == "short" and high_1h > state.prev_day_high and close_1h < state.prev_day_high:
                        state.fakeout_found = True
                        state.fakeout_bar_pos = i
                        state.fakeout_extreme = float(high_1h)

            # --- FVG/IFVG pullback entry ---
            if state.fakeout_found and state.fakeout_bar_pos is not None:
                bars_since_fakeout = i - state.fakeout_bar_pos
                if bars_since_fakeout > cfg.fvg_lookback_bars:
                    # Stale setup - drop it and wait for a fresh fakeout.
                    state.fakeout_found = False
                    state.fakeout_bar_pos = None
                    state.fakeout_extreme = None
                    continue
                if bars_since_fakeout <= 0:
                    continue  # need at least one bar after the fakeout for a post-fakeout FVG to exist

                wanted_kind = "bullish" if state.bias == "long" else "bearish"
                candidates = [
                    z
                    for z in _active_zones_at(zones, i, df["close"])
                    if z.kind == wanted_kind and z.confirmed_pos > state.fakeout_bar_pos and z.confirmed_pos <= i
                ]
                if not candidates:
                    continue
                aoi = max(candidates, key=lambda z: z.confirmed_pos)

                bar = df.iloc[i]
                if state.bias == "long" and bar["low"] <= aoi.zone_high:
                    signals.append(
                        Signal(
                            signal_time=ts,
                            direction="long",
                            stop_loss=state.fakeout_extreme - cfg.stop_buffer_points,
                            take_profit=state.prev_day_high,
                            strategy_name=self.name,
                            meta={
                                "aoi_zone_low": aoi.zone_low,
                                "aoi_zone_high": aoi.zone_high,
                                "aoi_kind": aoi.kind,
                                "fakeout_extreme": state.fakeout_extreme,
                            },
                        )
                    )
                    state.trades_taken += 1
                    state.fakeout_found = False
                elif state.bias == "short" and bar["high"] >= aoi.zone_low:
                    signals.append(
                        Signal(
                            signal_time=ts,
                            direction="short",
                            stop_loss=state.fakeout_extreme + cfg.stop_buffer_points,
                            take_profit=state.prev_day_low,
                            strategy_name=self.name,
                            meta={
                                "aoi_zone_low": aoi.zone_low,
                                "aoi_zone_high": aoi.zone_high,
                                "aoi_kind": aoi.kind,
                                "fakeout_extreme": state.fakeout_extreme,
                            },
                        )
                    )
                    state.trades_taken += 1
                    state.fakeout_found = False

        return signals

    def _ny_window_tz(self):
        from zoneinfo import ZoneInfo

        return ZoneInfo(self.config.session_tz)

    @staticmethod
    def _daily_bias_at(daily_ctx: pd.DataFrame, ts: pd.Timestamp) -> tuple[Optional[Direction], Optional[float], Optional[float]]:
        if ts not in daily_ctx.index:
            return None, None, None
        row = daily_ctx.loc[ts]
        open_p, close_p = row.get("open_htf"), row.get("close_htf")
        high_p, low_p = row.get("high_htf"), row.get("low_htf")
        if pd.isna(open_p) or pd.isna(close_p) or pd.isna(high_p) or pd.isna(low_p):
            return None, None, None
        if close_p > open_p:
            return "long", float(high_p), float(low_p)
        if close_p < open_p:
            return "short", float(high_p), float(low_p)
        return None, None, None


@register_strategy(
    "liquidity_sweep_fvg",
    LiquiditySweepFVGConfig,
    "1D bias + 1H NY-session liquidity sweep (fakeout) + 5M FVG/IFVG pullback entry, targeting previous day's level",
)
def _build_liquidity_sweep_fvg(**kwargs) -> LiquiditySweepFVGStrategy:
    return LiquiditySweepFVGStrategy(LiquiditySweepFVGConfig(**kwargs))
