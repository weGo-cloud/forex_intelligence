"""Strategy-agnostic backtesting engine.

Contains NO ORB-specific or RSI-specific logic — it only knows about
the generic `Signal` shape from `app.strategies.base`. Fill and exit
policy (the realism-critical part) is centralized here exactly once, so
every strategy gets the same anti-lookahead guarantees automatically
instead of re-implementing (and potentially getting wrong) fill logic
per strategy.

Fill policy (not configurable — see `BacktestConfig` docstring):
  Entry fills at the OPEN of the bar immediately after `signal.signal_time`.
  A signal is, by construction, detected using information available up
  to and including the signal bar's close; filling on that same bar
  would require knowing its close before it happened.

Exit policy:
  SL/TP are checked starting the bar AFTER entry (when
  `exits_start_bar_after_entry=True`, the default) to avoid the
  ambiguity of "did price hit the stop before or after the entry fill,
  within the same bar." Same-bar SL+TP hits (both levels touched within
  one bar) are resolved by `same_bar_sl_tp_policy`, defaulting to
  "stop_loss_first" — the conservative assumption, since assuming the
  favorable order without intrabar data would systematically overstate
  performance.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from app.backtesting.config import BacktestConfig
from app.backtesting.results import Trade
from app.strategies.base import Signal


def _half_spread_fill(raw_price: float, direction: str, action: str, spread_points: float) -> float:
    """Apply half the spread against the trader on both entry and exit,
    for both directions — buying pays the ask (above mid), selling
    receives the bid (below mid), regardless of whether the sale opens
    a short or closes a long."""
    half = spread_points / 2.0
    buying = (direction == "long" and action == "open") or (direction == "short" and action == "close")
    return raw_price + half if buying else raw_price - half


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(
        self,
        ohlcv: pd.DataFrame,
        signals: list[Signal],
        strategy_name: str,
        strategy_params: dict[str, Any],
    ) -> list[Trade]:
        if ohlcv.empty or not signals:
            return []

        cfg = self.config
        index = ohlcv.index
        pos_by_ts = {ts: i for i, ts in enumerate(index)}
        opens = ohlcv["open"].to_numpy()
        highs = ohlcv["high"].to_numpy()
        lows = ohlcv["low"].to_numpy()
        closes = ohlcv["close"].to_numpy()
        n = len(ohlcv)

        equity = cfg.initial_equity
        trades: list[Trade] = []
        last_exit_time: Optional[pd.Timestamp] = None

        for signal in sorted(signals, key=lambda s: s.signal_time):
            if signal.direction == "short" and not cfg.allow_short:
                continue

            signal_pos = pos_by_ts.get(signal.signal_time)
            if signal_pos is None:
                continue  # signal timestamp not in this OHLCV index; nothing to fill against

            entry_pos = signal_pos + 1
            if entry_pos >= n:
                continue  # no bar exists to fill on (signal on the last available bar)

            entry_time = index[entry_pos]
            if cfg.one_trade_at_a_time and last_exit_time is not None and entry_time < last_exit_time:
                continue  # would overlap a still-open position

            raw_entry = float(opens[entry_pos])
            if pd.isna(raw_entry):
                continue

            entry_price = _half_spread_fill(raw_entry, signal.direction, "open", cfg.spread_points)
            risk_per_unit = abs(entry_price - signal.stop_loss)
            if risk_per_unit <= 0:
                continue  # degenerate signal (stop == entry): cannot size or evaluate

            direction_sign = 1.0 if signal.direction == "long" else -1.0
            # take_profit: if the strategy set an explicit level on the signal
            # (e.g. "previous day's high/low" - a fixed price target, not an
            # R-multiple), use it as-is. Otherwise, derive it from the
            # RISK/REWARD ratio the strategy asked for, applied to the ACTUAL
            # fill price and ACTUAL stop distance - never the price used to
            # detect the signal (see Signal docstring). This is the only
            # place take_profit is computed; existing strategies that never
            # set signal.take_profit are completely unaffected.
            if signal.take_profit is not None:
                take_profit = signal.take_profit
            else:
                risk_reward = strategy_params.get("risk_reward", 2.0)
                take_profit = entry_price + direction_sign * risk_reward * risk_per_unit

            # An explicit take_profit on the wrong side of entry (e.g. a long
            # whose target is below its entry) is a degenerate signal - skip
            # it rather than open a position that can never reach its target
            # in the intended direction. risk_reward-derived targets can
            # never trigger this (they're constructed on the correct side by
            # definition), so this only guards explicit-target strategies.
            if (signal.direction == "long" and take_profit <= entry_price) or (
                signal.direction == "short" and take_profit >= entry_price
            ):
                continue

            position_size = (equity * cfg.risk_per_trade_pct / 100.0) / risk_per_unit
            planned_risk_amount = risk_per_unit * position_size

            scan_start = entry_pos + 1 if cfg.exits_start_bar_after_entry else entry_pos
            exit_pos: Optional[int] = None
            exit_price_raw: Optional[float] = None
            exit_reason: str = "end_of_data"

            for p in range(scan_start, n):
                if cfg.max_holding_bars is not None and (p - entry_pos) > cfg.max_holding_bars:
                    exit_pos = p - 1
                    exit_price_raw = float(closes[exit_pos])
                    exit_reason = "max_holding_period"
                    break

                bar_high, bar_low = float(highs[p]), float(lows[p])
                if signal.direction == "long":
                    hit_sl = bar_low <= signal.stop_loss
                    hit_tp = bar_high >= take_profit
                else:
                    hit_sl = bar_high >= signal.stop_loss
                    hit_tp = bar_low <= take_profit

                if hit_sl and hit_tp:
                    if cfg.same_bar_sl_tp_policy == "stop_loss_first":
                        exit_pos, exit_price_raw, exit_reason = p, signal.stop_loss, "stop_loss"
                    else:
                        exit_pos, exit_price_raw, exit_reason = p, take_profit, "take_profit"
                    break
                if hit_sl:
                    exit_pos, exit_price_raw, exit_reason = p, signal.stop_loss, "stop_loss"
                    break
                if hit_tp:
                    exit_pos, exit_price_raw, exit_reason = p, take_profit, "take_profit"
                    break
            else:
                exit_pos = n - 1
                exit_price_raw = float(closes[exit_pos])
                exit_reason = "end_of_data"

            if exit_pos is None or exit_price_raw is None:
                continue

            exit_time = index[exit_pos]
            exit_price = _half_spread_fill(exit_price_raw, signal.direction, "close", cfg.spread_points)

            gross_pnl = (exit_price - entry_price) * position_size * direction_sign
            net_pnl = gross_pnl - cfg.commission_per_trade
            r_multiple = net_pnl / planned_risk_amount if planned_risk_amount else 0.0

            equity += net_pnl
            last_exit_time = exit_time

            trades.append(
                Trade(
                    strategy_name=strategy_name,
                    instrument_symbol=cfg.instrument_symbol,
                    direction=signal.direction,
                    entry_time=entry_time.to_pydatetime(),
                    exit_time=exit_time.to_pydatetime(),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=signal.stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    r_multiple=r_multiple,
                    exit_reason=exit_reason,  # type: ignore[arg-type]
                    holding_bars=exit_pos - entry_pos,
                    strategy_params=strategy_params,
                    signal_meta=signal.meta,
                )
            )

        return trades
