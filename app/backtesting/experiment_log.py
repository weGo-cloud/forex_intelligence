"""Lightweight experiment log: every backtest run gets one appended
JSON line recording exactly what was run and what happened. This is
the minimum viable version of what the original roadmap called an
"Experiment/Research Registry" — deliberately NOT a database table or
a UI, because the actual problem it solves (results vanishing when a
script exits, no way to tell which config produced which numbers) is
solved by durable, structured, greppable records — a dashboard is a
nice-to-have layered on top of this later, not a prerequisite for it.

File format: one JSON object per line (JSONL) at
`experiments/log.jsonl` (git-ignored — see docs). Append-only by
design: never edit or delete a past entry - if a run was wrong, log a
new corrected run rather than rewriting history you might want to
audit later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.backtesting.results import BacktestResult

DEFAULT_LOG_PATH = Path("experiments/log.jsonl")


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    logged_at: str
    strategy_name: str
    instrument_symbol: str
    strategy_params: dict[str, Any]
    backtest_config: dict[str, Any]
    test_period_start: str
    test_period_end: str
    trade_count: int
    win_rate: Optional[float]
    expectancy: Optional[float]
    profit_factor: Optional[float]
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    assumptions: list[str]
    tags: dict[str, str] = field(default_factory=dict)  # e.g. {"walk_forward_window": "3"}


def _next_experiment_id(log_path: Path) -> str:
    """Sequential IDs (exp_0001, exp_0002, ...) rather than UUIDs — for
    a lab workflow, a human needs to say "run exp_0014 again" out loud
    or in a commit message; a UUID makes that miserable for no benefit
    at this scale."""
    if not log_path.exists():
        return "exp_0001"
    count = sum(1 for line in log_path.open() if line.strip())
    return f"exp_{count + 1:04d}"


def log_experiment(
    result: BacktestResult,
    log_path: Path = DEFAULT_LOG_PATH,
    tags: Optional[dict[str, str]] = None,
) -> ExperimentRecord:
    """Append one backtest result to the log. Returns the record
    (including its assigned experiment_id) so the caller can print or
    reference it immediately."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = ExperimentRecord(
        experiment_id=_next_experiment_id(log_path),
        logged_at=datetime.now(timezone.utc).isoformat(),
        strategy_name=result.metrics.strategy_name,
        instrument_symbol=result.metrics.instrument_symbol,
        strategy_params=result.strategy_params,
        backtest_config=_json_safe(result.backtest_config),
        test_period_start=result.metrics.test_period_start.isoformat(),
        test_period_end=result.metrics.test_period_end.isoformat(),
        trade_count=result.metrics.trade_count,
        win_rate=result.metrics.win_rate,
        expectancy=result.metrics.expectancy,
        profit_factor=result.metrics.profit_factor,
        total_return_pct=result.metrics.total_return_pct,
        max_drawdown_pct=result.metrics.max_drawdown_pct,
        sharpe_ratio=result.metrics.sharpe_ratio,
        assumptions=result.assumptions,
        tags=tags or {},
    )
    with log_path.open("a") as f:
        f.write(json.dumps(asdict(record), default=str) + "\n")
    return record


def _json_safe(d: dict[str, Any]) -> dict[str, Any]:
    """BacktestConfig.__dict__ can contain non-JSON-serializable values
    (e.g. enum members from future config fields) - stringify anything
    json can't handle natively rather than letting the whole log write
    fail on one odd field."""
    safe = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    return safe


def load_experiments(log_path: Path = DEFAULT_LOG_PATH) -> list[ExperimentRecord]:
    if not log_path.exists():
        return []
    records = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ExperimentRecord(**json.loads(line)))
    return records


def compare_experiments(
    log_path: Path = DEFAULT_LOG_PATH,
    strategy_name: Optional[str] = None,
    instrument_symbol: Optional[str] = None,
    sort_by: str = "expectancy",
) -> list[ExperimentRecord]:
    """Filter and sort logged experiments — the actual "step 7" tool.
    Records with `sort_by` = None (undefined metric, e.g. no losing
    trades yet) sort last, never crash the comparison and never get
    silently treated as zero (which would misrank an undefined result
    as worse than a real bad one)."""
    records = load_experiments(log_path)
    if strategy_name is not None:
        records = [r for r in records if r.strategy_name == strategy_name]
    if instrument_symbol is not None:
        records = [r for r in records if r.instrument_symbol == instrument_symbol]

    def sort_key(r: ExperimentRecord):
        value = getattr(r, sort_by, None)
        return (value is None, -(value or 0))

    return sorted(records, key=sort_key)


def format_comparison_table(records: list[ExperimentRecord]) -> str:
    """Plain-text table for terminal/CLI output - no dependency on a
    dashboard or notebook to see how experiments stack up."""
    if not records:
        return "(no experiments logged yet)"

    headers = ["id", "strategy", "instrument", "trades", "win_rate", "expectancy", "profit_factor", "max_dd_pct"]
    rows = []
    for r in records:
        rows.append(
            [
                r.experiment_id,
                r.strategy_name,
                r.instrument_symbol,
                str(r.trade_count),
                f"{r.win_rate:.2f}" if r.win_rate is not None else "n/a",
                f"{r.expectancy:.2f}" if r.expectancy is not None else "n/a",
                f"{r.profit_factor:.2f}" if r.profit_factor is not None else "n/a",
                f"{r.max_drawdown_pct:.2f}",
            ]
        )

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)
