"""One-call orchestration: M1 stored data -> strategy -> engine -> metrics
-> BacktestResult. This is the only module a researcher/script needs to
import to run a backtest end to end; everything it calls is independently
testable and importable on its own (matching M1's own module philosophy).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.backtesting.config import BacktestConfig
from app.backtesting.data_access import align_higher_timeframe, load_ohlcv
from app.backtesting.engine import BacktestEngine
from app.backtesting.metrics import compute_metrics
from app.backtesting.results import BacktestResult
from app.data.timeframes import Timeframe
from app.strategies.base import MarketContext, Strategy

ASSUMPTIONS: list[str] = [
    "Backtesting engine is a custom bar-by-bar simulator, not VectorBT — "
    "this environment could not install/verify VectorBT, and an "
    "unverified fill/SL-TP implementation risks exactly the look-ahead "
    "bias this project treats as a failed implementation. Interface is "
    "seamed (Signal in, Trade list out) so a VectorBT adapter could "
    "replace this engine later without touching strategy code.",
    "RSI and swing-point detection are hand-implemented rather than "
    "using TA-Lib/`ta`, for the same no-network-verification reason, "
    "plus swing confirmation-lag has no standard library implementation.",
    "Entry always fills at the next bar's open after a signal; same-bar "
    "fills at the signal's own close are never used (would require "
    "knowing the close before it happened).",
    "SL/TP monitoring begins the bar AFTER entry, not the entry bar "
    "itself, to avoid same-bar fill/exit ordering ambiguity.",
    "Same-bar SL+TP hits resolve to stop-loss-first by default "
    "(conservative; configurable to take-profit-first for sensitivity "
    "testing via BacktestConfig.same_bar_sl_tp_policy).",
    "Transaction costs are a flat spread (price points) applied as "
    "half-spread against the trader on both entry and exit, plus a "
    "flat commission per trade. No slippage-beyond-spread model exists "
    "yet — this understates costs during high-volatility/news periods.",
    "Position sizing is percent-of-current-equity risk sizing "
    "(compounding); this is a simplifying assumption, not a claim about "
    "how a live account would be sized.",
    "Candle timestamps are treated as bar-OPEN time (confirmed from "
    "M1's aggregation module), which determines both the entry-fill-next-bar "
    "rule and the higher-timeframe visibility-lag rule.",
]


@dataclass(frozen=True)
class RunSpec:
    """What to backtest and over what range."""

    symbol: str
    timeframe: Timeframe
    start: datetime
    end: datetime
    context_timeframes: dict[str, Timeframe] | None = None  # e.g. {"1H": Timeframe.H1}


def run_backtest(
    session: Session,
    strategy: Strategy,
    run_spec: RunSpec,
    backtest_config: BacktestConfig,
) -> BacktestResult:
    execution_df = load_ohlcv(session, run_spec.symbol, run_spec.timeframe, run_spec.start, run_spec.end)

    context: dict[str, "object"] = {}
    if run_spec.context_timeframes:
        for label, tf in run_spec.context_timeframes.items():
            htf_df = load_ohlcv(session, run_spec.symbol, tf, run_spec.start, run_spec.end)
            context[label] = align_higher_timeframe(execution_df.index, htf_df, tf.minutes)

    market = MarketContext(instrument_symbol=run_spec.symbol, execution_df=execution_df, context=context)
    signals = strategy.generate_signals(market)

    engine = BacktestEngine(backtest_config)
    trades = engine.run(execution_df, signals, strategy.name, strategy.params())

    metrics = compute_metrics(
        trades=trades,
        strategy_name=strategy.name,
        instrument_symbol=run_spec.symbol,
        initial_equity=backtest_config.initial_equity,
        test_period_start=run_spec.start,
        test_period_end=run_spec.end,
    )

    return BacktestResult(
        metrics=metrics,
        trades=trades,
        strategy_params=strategy.params(),
        backtest_config=backtest_config.__dict__,
        assumptions=ASSUMPTIONS,
    )
