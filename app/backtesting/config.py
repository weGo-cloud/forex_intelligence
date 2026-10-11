"""Backtest configuration. Every assumption that affects realism or
result reproducibility is a named field here, not a hard-coded constant
inside the engine — so every backtest's exact assumptions are recorded
alongside its results (see `BacktestResult.config` in results.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class BacktestConfig:
    instrument_symbol: str

    # --- Position sizing ---
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0  # % of current equity risked per trade

    # --- Transaction costs ---
    # Applied symmetrically at entry and exit as half-spread each side.
    # Units: price points (same units as the instrument's price), NOT
    # percent — spread is not naturally proportional to price the way
    # commission often is. Set per-instrument by the caller (see
    # docs/M2_BACKTESTING_ENGINE.md for why costs are never a single
    # global constant across instrument classes).
    spread_points: float = 0.0
    commission_per_trade: float = 0.0  # flat currency amount, both sides combined

    # --- Fill / exit realism policy ---
    # Entry always fills at the OPEN of the bar AFTER the signal bar,
    # never at the signal bar's own close - see Signal docstring for why.
    # This is not configurable: it is the one realism rule this engine
    # does not allow disabling, because a strategy signal is by
    # definition detected using that bar's close, and filling on the
    # same bar's close would require knowing the close before it happened.
    exits_start_bar_after_entry: bool = True  # SL/TP monitoring begins the bar AFTER entry, not the entry bar itself
    same_bar_sl_tp_policy: Literal["stop_loss_first", "take_profit_first"] = "stop_loss_first"
    max_holding_bars: Optional[int] = None  # None = hold until SL/TP/end of data

    # --- Trade management ---
    one_trade_at_a_time: bool = True
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")
        if not (0 < self.risk_per_trade_pct <= 100):
            raise ValueError("risk_per_trade_pct must be in (0, 100]")
        if self.spread_points < 0 or self.commission_per_trade < 0:
            raise ValueError("costs cannot be negative")
        if self.max_holding_bars is not None and self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1 when set")
