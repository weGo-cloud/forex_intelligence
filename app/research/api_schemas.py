"""Pydantic request/response schemas for the research API.

Kept separate from `app.research.models`/`app.research.registry` (M3
spec section 14: "keep API schemas separate from domain models") so the
wire format can evolve without touching the domain/persistence layer,
same convention as `app.data.schemas` for the market-data API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.research.models import ExperimentStatus, ExperimentType


class ExperimentCreateRequest(BaseModel):
    strategy_name: str
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    instrument_symbol: str
    timeframe: str
    test_period_start: datetime
    test_period_end: datetime

    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    spread_points: float = 0.0
    commission_per_trade: float = 0.0
    same_bar_sl_tp_policy: str = "stop_loss_first"
    max_holding_bars: Optional[int] = None
    one_trade_at_a_time: bool = True
    allow_short: bool = True

    context_timeframes: dict[str, str] = Field(default_factory=dict)
    experiment_name: Optional[str] = None
    tags: dict[str, str] = Field(default_factory=dict)


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    experiment_id: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    strategy_name: str
    instrument_symbol: str
    config: dict[str, Any]
    config_hash: str
    experiment_name: Optional[str] = None
    tags: dict[str, str]
    parent_experiment_id: Optional[str] = None
    result_summary: Optional[dict[str, Any]] = None
    failure_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class ComparisonRowResponse(BaseModel):
    experiment_id: str
    status: str
    strategy_name: str
    strategy_params: dict[str, Any]
    instrument_symbol: str
    timeframe: str
    test_period_start: str
    test_period_end: str
    trade_count: Optional[int] = None
    total_return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    expectancy: Optional[float] = None
    win_rate: Optional[float] = None
