"""Execute a single experiment: ExperimentDefinition -> M2 backtest ->
persisted ExperimentRecord with a structured result summary.

This is the one place M3 calls into M2 (`app.backtesting`,
`app.strategies`) to actually run a strategy. Every other M3 module
(sweep, robustness, comparison) is built on top of this function, never
around it - so provenance (section 12 of the M3 spec) is always
Experiment -> Configuration -> Data selection -> Strategy -> Execution
engine -> Metrics, with no side path that produces an untraceable result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.data_access import UnknownInstrumentError, align_higher_timeframe, load_ohlcv
from app.backtesting.metrics import compute_metrics
from app.backtesting.results import BacktestResult
from app.data.timeframes import Timeframe
from app.research.models import ExperimentDefinition
from app.research.registry import ExperimentRecord, ExperimentRegistry
from app.strategies.base import MarketContext
from app.strategies.registry import get_strategy


class ExperimentExecutionError(Exception):
    """Raised when an experiment fails during execution, after already
    being marked RUNNING in the registry. Callers that want the
    experiment marked FAILED (rather than left stuck RUNNING) should
    use `execute_experiment`, which catches this and calls
    `registry.mark_failed` automatically."""


@dataclass(frozen=True)
class ExperimentOutcome:
    """What `execute_experiment` returns: the (now COMPLETED or FAILED)
    registry record, plus the full BacktestResult when execution
    succeeded - the full trade log and metrics for the caller to
    inspect, even though only a summary is persisted in the registry."""

    record: ExperimentRecord
    result: Optional[BacktestResult]


def _build_backtest_config(definition: ExperimentDefinition) -> BacktestConfig:
    return BacktestConfig(
        instrument_symbol=definition.instrument_symbol,
        initial_equity=definition.initial_equity,
        risk_per_trade_pct=definition.risk_per_trade_pct,
        spread_points=definition.spread_points,
        commission_per_trade=definition.commission_per_trade,
        same_bar_sl_tp_policy=definition.same_bar_sl_tp_policy,  # type: ignore[arg-type]
        max_holding_bars=definition.max_holding_bars,
        one_trade_at_a_time=definition.one_trade_at_a_time,
        allow_short=definition.allow_short,
    )


def run_single_backtest(session: Session, definition: ExperimentDefinition) -> BacktestResult:
    """Run the backtest described by `definition` and return the raw
    `BacktestResult` - no persistence, no registry interaction. Kept
    separate from `execute_experiment` so the sweep/walk-forward/
    robustness engines can run many of these without each one
    round-tripping through registry status transitions individually."""
    timeframe = Timeframe(definition.timeframe)
    execution_df = load_ohlcv(
        session, definition.instrument_symbol, timeframe, definition.test_period_start, definition.test_period_end
    )

    context: dict[str, Any] = {}
    for label, tf_value in definition.context_timeframes.items():
        tf = Timeframe(tf_value)
        htf_df = load_ohlcv(
            session, definition.instrument_symbol, tf, definition.test_period_start, definition.test_period_end
        )
        context[label] = align_higher_timeframe(execution_df.index, htf_df, tf.minutes)

    strategy = get_strategy(definition.strategy_name, **definition.strategy_params)
    market = MarketContext(instrument_symbol=definition.instrument_symbol, execution_df=execution_df, context=context)
    signals = strategy.generate_signals(market)

    backtest_config = _build_backtest_config(definition)
    engine = BacktestEngine(backtest_config)
    trades = engine.run(execution_df, signals, strategy.name, strategy.params())

    metrics = compute_metrics(
        trades=trades,
        strategy_name=strategy.name,
        instrument_symbol=definition.instrument_symbol,
        initial_equity=backtest_config.initial_equity,
        test_period_start=definition.test_period_start,
        test_period_end=definition.test_period_end,
    )

    return BacktestResult(
        metrics=metrics,
        trades=trades,
        strategy_params=strategy.params(),
        backtest_config=backtest_config.__dict__,
        assumptions=[
            "Executed via the M3 research layer (app.research.experiment), which calls the "
            "unmodified M2 BacktestEngine/runner primitives - see app/backtesting/runner.py "
            "ASSUMPTIONS for the underlying engine's fill/exit/cost assumptions.",
        ],
    )


def result_summary(result: BacktestResult) -> dict[str, Any]:
    """The structured subset of a BacktestResult that gets persisted in
    the registry. Deliberately excludes the trade log (section 6:
    "do not store large duplicated datasets") - the full BacktestResult
    remains available to a caller that just ran it via
    `ExperimentOutcome.result`, but is not durably stored twice."""
    m = result.metrics
    return {
        "trade_count": m.trade_count,
        "winning_trades": m.winning_trades,
        "losing_trades": m.losing_trades,
        "win_rate": m.win_rate,
        "avg_trade": m.avg_trade,
        "expectancy": m.expectancy,
        "profit_factor": m.profit_factor,
        "total_return_pct": m.total_return_pct,
        "max_drawdown_pct": m.max_drawdown_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "avg_holding_bars": m.avg_holding_bars,
        "long_stats": m.long_stats.model_dump(),
        "short_stats": m.short_stats.model_dump(),
    }


def execute_experiment(session: Session, registry: ExperimentRegistry, definition: ExperimentDefinition) -> ExperimentOutcome:
    """Create, run, and persist one experiment end to end: PENDING ->
    RUNNING -> (COMPLETED with a result summary) | (FAILED with a
    reason). This is the function the API and CLI call for a single
    SINGLE_BACKTEST experiment; sweep/walk-forward/robustness call
    `run_single_backtest` + registry directly per child so they control
    the parent/child bookkeeping themselves.
    """
    record = registry.create(definition)
    registry.mark_running(record.experiment_id)

    try:
        result = run_single_backtest(session, definition)
    except UnknownInstrumentError as exc:
        failed = registry.mark_failed(record.experiment_id, f"unknown_instrument: {exc}")
        return ExperimentOutcome(record=failed, result=None)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any execution failure must be persisted, not raised past the registry
        failed = registry.mark_failed(record.experiment_id, f"execution_error: {type(exc).__name__}: {exc}")
        return ExperimentOutcome(record=failed, result=None)

    completed = registry.mark_completed(record.experiment_id, result_summary(result))
    return ExperimentOutcome(record=completed, result=result)
