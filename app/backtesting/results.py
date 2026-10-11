"""Standardized, strategy-independent result shapes.

Pydantic models, consistent with M1's `app/data/schemas.py` convention
(canonical shapes for anything that crosses a module boundary or might
later be serialized/returned over an API) — Trade, BacktestMetrics, and
BacktestResult are exactly that: the common contract every strategy's
backtest produces, regardless of which strategy generated it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

ExitReason = Literal["stop_loss", "take_profit", "max_holding_period", "end_of_data", "unknown"]
# "unknown" is deliberately valid, not an oversight: some execution
# backends (see app/backtesting/vbt_engine.py) cannot reliably attribute
# a trade's exit cause from their own output alone. Reporting "unknown"
# is honest; guessing from the sign of PnL would silently misclassify
# max-holding-period/end-of-data exits as stop-loss/take-profit hits.
Direction = Literal["long", "short"]


class Trade(BaseModel):
    """A single simulated trade, with enough detail to manually
    investigate it against raw market data (spec requirement: "every
    trade should contain enough information to investigate it")."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    instrument_symbol: str
    direction: Direction
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    gross_pnl: float
    net_pnl: float
    r_multiple: float  # net_pnl / initial risk amount (entry-to-stop distance x size)
    exit_reason: ExitReason
    holding_bars: int
    strategy_params: dict[str, Any]
    signal_meta: dict[str, Any] = {}


class DirectionalStats(BaseModel):
    """Metrics subset for one trade direction (long or short)."""

    trade_count: int
    win_rate: Optional[float]
    avg_trade: Optional[float]
    profit_factor: Optional[float]


class BacktestMetrics(BaseModel):
    """Standardized, strategy-independent summary metrics.

    All fields are Optional[float] (not 0.0) where a metric is undefined
    at low trade counts (e.g. profit_factor with zero losing trades) —
    reporting 0.0 for "undefined" would silently misrepresent an
    insufficient-evidence case as a real result.
    """

    strategy_name: str
    instrument_symbol: str
    test_period_start: datetime
    test_period_end: datetime

    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate: Optional[float]
    avg_trade: Optional[float]
    avg_win: Optional[float]
    avg_loss: Optional[float]
    expectancy: Optional[float]
    profit_factor: Optional[float]
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    avg_holding_bars: Optional[float]

    long_stats: DirectionalStats
    short_stats: DirectionalStats


class BacktestResult(BaseModel):
    """Everything produced by one backtest run: metrics, trade log, and
    the exact configuration used — so the run is reproducible and
    auditable (spec requirement: "a researcher can reproduce the same
    result from the same data and configuration")."""

    metrics: BacktestMetrics
    trades: list[Trade]
    strategy_params: dict[str, Any]
    backtest_config: dict[str, Any]
    assumptions: list[str]
