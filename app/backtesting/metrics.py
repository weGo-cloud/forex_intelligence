"""Compute standardized, strategy-independent metrics from a trade list.

Metrics that are mathematically undefined at low trade counts (e.g.
profit_factor with zero losing trades, Sharpe with fewer than 2 trades)
return `None`, never 0.0 or infinity — silently reporting an undefined
quantity as a number would misrepresent an insufficient-evidence case as
a real result, which is exactly what this whole project is trying to
avoid at the product level too.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from app.backtesting.results import BacktestMetrics, DirectionalStats, Trade


def _directional_stats(trades: list[Trade]) -> DirectionalStats:
    n = len(trades)
    if n == 0:
        return DirectionalStats(trade_count=0, win_rate=None, avg_trade=None, profit_factor=None)

    pnls = [t.net_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / n
    avg_trade = float(np.mean(pnls))
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None

    return DirectionalStats(trade_count=n, win_rate=win_rate, avg_trade=avg_trade, profit_factor=profit_factor)


def compute_metrics(
    trades: list[Trade],
    strategy_name: str,
    instrument_symbol: str,
    initial_equity: float,
    test_period_start: datetime,
    test_period_end: datetime,
) -> BacktestMetrics:
    n = len(trades)
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]

    if n == 0:
        return BacktestMetrics(
            strategy_name=strategy_name,
            instrument_symbol=instrument_symbol,
            test_period_start=test_period_start,
            test_period_end=test_period_end,
            trade_count=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=None,
            avg_trade=None,
            avg_win=None,
            avg_loss=None,
            expectancy=None,
            profit_factor=None,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=None,
            avg_holding_bars=None,
            long_stats=_directional_stats([]),
            short_stats=_directional_stats([]),
        )

    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = len(wins) / n
    avg_trade = float(pnls.mean())
    avg_win = float(wins.mean()) if len(wins) else None
    avg_loss = float(losses.mean()) if len(losses) else None
    expectancy = win_rate * (avg_win or 0.0) + (1 - win_rate) * (avg_loss or 0.0)
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None

    equity_curve = initial_equity + np.cumsum(pnls)
    equity_curve_with_start = np.concatenate([[initial_equity], equity_curve])
    total_return_pct = (equity_curve_with_start[-1] - initial_equity) / initial_equity * 100.0

    running_max = np.maximum.accumulate(equity_curve_with_start)
    drawdowns = (equity_curve_with_start - running_max) / running_max * 100.0
    max_drawdown_pct = float(drawdowns.min())

    # Trade-level Sharpe: per-trade return on initial equity, NOT
    # annualized (trade frequency varies by strategy/regime, so an
    # annualization factor would be a guess dressed up as a statistic).
    trade_returns = pnls / initial_equity
    if n >= 2 and trade_returns.std(ddof=1) > 0:
        sharpe_ratio: Optional[float] = float(trade_returns.mean() / trade_returns.std(ddof=1) * np.sqrt(n))
    else:
        sharpe_ratio = None

    avg_holding_bars = float(np.mean([t.holding_bars for t in trades]))

    return BacktestMetrics(
        strategy_name=strategy_name,
        instrument_symbol=instrument_symbol,
        test_period_start=test_period_start,
        test_period_end=test_period_end,
        trade_count=n,
        winning_trades=int(len(wins)),
        losing_trades=int(len(losses)),
        win_rate=win_rate,
        avg_trade=avg_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        profit_factor=profit_factor,
        total_return_pct=float(total_return_pct),
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        avg_holding_bars=avg_holding_bars,
        long_stats=_directional_stats(long_trades),
        short_stats=_directional_stats(short_trades),
    )
