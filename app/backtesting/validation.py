"""Minimal walk-forward validation support.

This is deliberately small: a function that carves a date range into
(train, test) windows, and a helper that runs a backtest on each test
window and logs every window as its own tagged experiment. It is NOT
the full robustness suite from the original roadmap (parameter-
neighborhood sensitivity, purging/embargo, Monte Carlo) - it's the
smallest thing that makes "Validate" step 5 of the research workflow
an actual command instead of an aspiration. Build the rest only once
this minimum is actually being used.

Why this matters even in its minimal form: a single backtest over one
date range tells you almost nothing about whether a strategy's edge is
real or a property of that one range. Walk-forward - repeatedly testing
on a window the strategy's parameters were not tuned on - is the
smallest step that turns "I ran a backtest" into "I have some evidence".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from app.backtesting.config import BacktestConfig
from app.backtesting.experiment_log import DEFAULT_LOG_PATH, ExperimentRecord, log_experiment

if TYPE_CHECKING:
    # Deferred - see module docstring in data_access.py for the same pattern
    # and reasoning: walk_forward_windows() is pure date arithmetic and must
    # stay importable/testable without a database or ORM present.
    from sqlalchemy.orm import Session

    from app.data.timeframes import Timeframe
    from app.strategies.base import Strategy


@dataclass(frozen=True)
class WalkForwardWindow:
    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def walk_forward_windows(
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
) -> list[WalkForwardWindow]:
    """Non-overlapping-by-default (step_days defaults to test_days)
    rolling windows. `train_*` bounds are returned for completeness /
    future use (e.g. parameter selection on the train window) but this
    minimal version does not do anything with the train window itself -
    it exists so a train/test split is structurally present from day
    one, rather than bolted on later once someone tries to reuse this
    for actual parameter selection and finds there's nowhere to put it.
    """
    if train_days < 1 or test_days < 1:
        raise ValueError("train_days and test_days must both be >= 1")
    step = step_days if step_days is not None else test_days
    if step < 1:
        raise ValueError("step_days must be >= 1")

    windows: list[WalkForwardWindow] = []
    train_start = start
    index = 0
    while True:
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > end:
            break
        windows.append(
            WalkForwardWindow(
                window_index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start = train_start + timedelta(days=step)
        index += 1

    return windows


def run_walk_forward(
    session: "Session",
    strategy: "Strategy",
    symbol: str,
    timeframe: "Timeframe",
    start: datetime,
    end: datetime,
    backtest_config: BacktestConfig,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> list[ExperimentRecord]:
    """Run the strategy on each walk-forward TEST window and log every
    window as its own experiment, tagged with the window index and
    train/test boundaries — so `compare_experiments` can show whether a
    strategy's numbers hold up across windows or only exist in one of
    them (the single most useful sanity check available at this stage:
    a strategy that only "works" in one window out of five is not
    evidence of an edge, and the walk-forward tag is what makes that
    visible in the comparison table instead of hidden in an averaged
    number).
    """
    from app.backtesting.runner import RunSpec, run_backtest  # deferred - see module docstring

    windows = walk_forward_windows(start, end, train_days, test_days, step_days)
    if not windows:
        raise ValueError(
            f"No walk-forward windows fit in [{start}, {end}] with train_days={train_days}, "
            f"test_days={test_days} - the range is too short for even one window."
        )

    records: list[ExperimentRecord] = []
    for window in windows:
        run_spec = RunSpec(symbol=symbol, timeframe=timeframe, start=window.test_start, end=window.test_end)
        result = run_backtest(session, strategy, run_spec, backtest_config)
        record = log_experiment(
            result,
            log_path=log_path,
            tags={
                "walk_forward_window": str(window.window_index),
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
            },
        )
        records.append(record)

    return records
