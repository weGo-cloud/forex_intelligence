"""Compare experiments with compatible result structures.

Deliberately does not rank: it exposes each experiment's identity,
configuration, and metrics in a flat, deterministically-ordered table
suitable for a human (or the future ranking engine) to read - not a
single composite score. Ranking strategies is explicitly out of scope
here (M3 spec section 11: "the future ranking engine will own strategy
ranking. M3 owns research comparison.").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.research.registry import ExperimentRecord, ExperimentRegistry


@dataclass(frozen=True)
class ComparisonRow:
    experiment_id: str
    status: str
    strategy_name: str
    strategy_params: dict
    instrument_symbol: str
    timeframe: str
    test_period_start: str
    test_period_end: str
    trade_count: Optional[int]
    total_return_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    sharpe_ratio: Optional[float]
    profit_factor: Optional[float]
    expectancy: Optional[float]
    win_rate: Optional[float]


class IncompatibleExperimentsError(Exception):
    """Raised when the requested experiments don't share a comparable
    shape - currently: different instruments or timeframes. Comparing
    a EURUSD/5M result against a GBPJPY/1H result side by side would be
    misleading, not just inconvenient, so this is a hard error rather
    than a silent best-effort comparison."""


def _to_row(record: ExperimentRecord) -> ComparisonRow:
    summary = record.result_summary or {}
    config = record.config
    return ComparisonRow(
        experiment_id=record.experiment_id,
        status=record.status,
        strategy_name=record.strategy_name,
        strategy_params=config.get("strategy_params", {}),
        instrument_symbol=record.instrument_symbol,
        timeframe=config.get("timeframe", ""),
        test_period_start=config.get("test_period_start", ""),
        test_period_end=config.get("test_period_end", ""),
        trade_count=summary.get("trade_count"),
        total_return_pct=summary.get("total_return_pct"),
        max_drawdown_pct=summary.get("max_drawdown_pct"),
        sharpe_ratio=summary.get("sharpe_ratio"),
        profit_factor=summary.get("profit_factor"),
        expectancy=summary.get("expectancy"),
        win_rate=summary.get("win_rate"),
    )


def compare_experiments(
    registry: ExperimentRegistry,
    experiment_ids: list[str],
    require_compatible: bool = True,
) -> list[ComparisonRow]:
    """Fetch and compare the given experiments.

    Missing/undefined metrics (e.g. a FAILED experiment, or a metric
    that is undefined because there were too few trades - see M2's
    `compute_metrics`) are passed through as `None`, never coerced to
    0.0 (section 17: "do not convert None/undefined into fake numerical
    confidence").

    Ordering is deterministic: rows are returned in the exact order
    `experiment_ids` was given, never re-sorted by a metric (that would
    be a ranking decision, which this module deliberately does not
    make).
    """
    records = [registry.get_or_raise(eid) for eid in experiment_ids]

    if require_compatible and records:
        first = records[0]
        for other in records[1:]:
            if other.instrument_symbol != first.instrument_symbol or other.config.get(
                "timeframe"
            ) != first.config.get("timeframe"):
                raise IncompatibleExperimentsError(
                    f"{other.experiment_id} ({other.instrument_symbol}/{other.config.get('timeframe')}) is not "
                    f"comparable to {first.experiment_id} ({first.instrument_symbol}/{first.config.get('timeframe')})"
                )

    return [_to_row(record) for record in records]
