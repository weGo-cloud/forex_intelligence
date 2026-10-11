"""Opening Range Breakout (ORB) strategy.

Rule (objective, no discretionary judgment):
1. On each session day, the "opening range" is the high/low spanning
   the first `opening_range_minutes` of the session (in `session_tz`).
2. After the opening range window closes, scan forward bar by bar
   (on `execution_timeframe`) within `trading_window_minutes` of the
   session start for the first bar whose CLOSE breaks beyond the
   opening range (breakout_definition="close", the default) — a
   close-based breakout is a deliberately more conservative default
   than an intrabar wick touch, since wick-based breakout definitions
   on lower timeframes produce many more false triggers from noise
   that reverses within the same bar. `breakout_definition="wick"` is
   supported for researchers who want to compare the two.
3. Stop-loss is the opposite side of the opening range
   (`stop_loss_method="opposite_range"`, the only method implemented in
   this milestone — documented default, not a hidden assumption).
4. Take-profit is derived by the engine from `risk_reward`, applied to
   the ACTUAL fill price (see `app/strategies/base.py::Signal`), not the
   breakout bar's close.
5. At most `max_trades_per_session` signals are emitted per session day
   (default 1 — the first qualifying breakout only).

Every parameter in the spec's "at minimum, make configurable" list is a
constructor argument on `ORBConfig`; there are no unstated hard-coded
values, i.e. no hidden rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any, Literal

import pandas as pd

from app.features.sessions import SessionWindow
from app.strategies.base import MarketContext, Signal
from app.strategies.registry import register_strategy


@dataclass(frozen=True)
class ORBConfig:
    instrument_symbol: str
    session_tz: str = "America/New_York"
    session_start: time = time(9, 30)
    opening_range_minutes: int = 15
    execution_timeframe_minutes: int = 5
    breakout_definition: Literal["close", "wick"] = "close"
    max_trades_per_session: int = 1
    stop_loss_method: Literal["opposite_range"] = "opposite_range"
    risk_reward: float = 2.0
    trading_window_minutes: int = 180

    def __post_init__(self) -> None:
        if self.opening_range_minutes % self.execution_timeframe_minutes != 0:
            raise ValueError(
                "opening_range_minutes must be a whole multiple of execution_timeframe_minutes "
                f"(got {self.opening_range_minutes} / {self.execution_timeframe_minutes})"
            )
        if self.trading_window_minutes <= self.opening_range_minutes:
            raise ValueError("trading_window_minutes must be greater than opening_range_minutes")
        if self.max_trades_per_session < 1:
            raise ValueError("max_trades_per_session must be >= 1")
        if self.risk_reward <= 0:
            raise ValueError("risk_reward must be > 0")


class ORBStrategy:
    """Opening Range Breakout. See module docstring for the exact rule."""

    def __init__(self, config: ORBConfig) -> None:
        self.config = config
        self.name = "orb"
        self._session = SessionWindow(config.session_tz, config.session_start, time(23, 59, 59))

    def params(self) -> dict[str, Any]:
        return {
            "instrument_symbol": self.config.instrument_symbol,
            "session_tz": self.config.session_tz,
            "session_start": self.config.session_start.isoformat(),
            "opening_range_minutes": self.config.opening_range_minutes,
            "execution_timeframe_minutes": self.config.execution_timeframe_minutes,
            "breakout_definition": self.config.breakout_definition,
            "max_trades_per_session": self.config.max_trades_per_session,
            "stop_loss_method": self.config.stop_loss_method,
            "risk_reward": self.config.risk_reward,
            "trading_window_minutes": self.config.trading_window_minutes,
        }

    def generate_signals(self, market: MarketContext) -> list[Signal]:
        df = market.execution_df
        if df.empty:
            return []

        cfg = self.config
        signals: list[Signal] = []

        for session_start_utc, _ in self._session.session_bounds_utc(df.index):
            or_end = session_start_utc + pd.Timedelta(minutes=cfg.opening_range_minutes)
            window_end = session_start_utc + pd.Timedelta(minutes=cfg.trading_window_minutes)

            opening_range = df.loc[(df.index >= session_start_utc) & (df.index < or_end)]
            if opening_range.empty:
                continue  # no data for this session in the loaded range - skip, don't guess

            or_high = float(opening_range["high"].max())
            or_low = float(opening_range["low"].min())

            post_range = df.loc[(df.index >= or_end) & (df.index < window_end)]
            trades_this_session = 0

            for ts, bar in post_range.iterrows():
                if trades_this_session >= cfg.max_trades_per_session:
                    break

                if cfg.breakout_definition == "close":
                    long_trigger = bar["close"] > or_high
                    short_trigger = bar["close"] < or_low
                else:  # "wick"
                    long_trigger = bar["high"] > or_high
                    short_trigger = bar["low"] < or_low

                if long_trigger and not short_trigger:
                    signals.append(
                        Signal(
                            signal_time=ts,
                            direction="long",
                            stop_loss=or_low,
                            strategy_name=self.name,
                            meta={
                                "opening_range_high": or_high,
                                "opening_range_low": or_low,
                                "session_date": str(self._session.session_date(session_start_utc)),
                            },
                        )
                    )
                    trades_this_session += 1
                elif short_trigger and not long_trigger:
                    signals.append(
                        Signal(
                            signal_time=ts,
                            direction="short",
                            stop_loss=or_high,
                            strategy_name=self.name,
                            meta={
                                "opening_range_high": or_high,
                                "opening_range_low": or_low,
                                "session_date": str(self._session.session_date(session_start_utc)),
                            },
                        )
                    )
                    trades_this_session += 1
                # both triggered on the same bar (high>or_high AND low<or_low,
                # only reachable with breakout_definition="wick"): ambiguous
                # same-bar direction, skip rather than guess.

        return signals


@register_strategy("orb", ORBConfig, "Opening Range Breakout - close/wick-based, opposite-range stop")
def _build_orb(**kwargs) -> ORBStrategy:
    return ORBStrategy(ORBConfig(**kwargs))
