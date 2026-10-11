#!/usr/bin/env python3
"""The main entrypoint for the research workflow's steps 4 and 7:

    python scripts/lab.py run --strategy orb --symbol US30 --timeframe M5 \\
        --start 2024-01-01 --end 2024-03-01 --param risk_reward=2.0

    python scripts/lab.py compare --strategy orb --sort-by expectancy

    python scripts/lab.py list-strategies

Every `run` appends one line to experiments/log.jsonl (see
app/backtesting/experiment_log.py) - nothing here is a research
decision-maker, it just makes running and comparing backtests a
one-line command instead of a hand-written script per strategy.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.backtesting.config import BacktestConfig
from app.backtesting.experiment_log import DEFAULT_LOG_PATH, compare_experiments, format_comparison_table, log_experiment
from app.backtesting.runner import RunSpec, run_backtest
from app.core.database import get_session_factory
from app.data.timeframes import Timeframe

# Importing these registers "orb" and "rsi_divergence" - see each
# module's bottom-of-file @register_strategy call. A new strategy
# module needs one line added here (see docs/ADDING_A_STRATEGY.md
# step 6) - this is the ONE place the whole project still requires a
# manual edit to make a new strategy runnable from the CLI.
import app.strategies.orb  # noqa: F401
import app.strategies.rsi_divergence  # noqa: F401
import app.strategies.mean_reversion  # noqa: F401
import app.strategies.liquidity_sweep_fvg  # noqa: F401
from app.strategies.registry import get_strategy, list_strategies


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _parse_param(s: str) -> tuple[str, object]:
    """"key=value" -> (key, value), with float/int/bool coercion so
    `--param risk_reward=2.0` doesn't arrive as the string "2.0" and
    trip a strategy config's type validation."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--param must be key=value, got {s!r}")
    key, raw = s.split("=", 1)
    for caster in (int, float):
        try:
            return key, caster(raw)
        except ValueError:
            continue
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    return key, raw


def cmd_list_strategies(_args: argparse.Namespace) -> int:
    entries = list_strategies()
    if not entries:
        print("No strategies registered.")
        return 0
    for e in entries:
        print(f"{e.name:20s} {e.description}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    params = dict(args.param or [])
    params.setdefault("instrument_symbol", args.symbol)
    strategy = get_strategy(args.strategy, **params)

    backtest_config = BacktestConfig(
        instrument_symbol=args.symbol,
        spread_points=args.spread_points,
        commission_per_trade=args.commission,
        risk_per_trade_pct=args.risk_pct,
        initial_equity=args.initial_equity,
    )
    run_spec = RunSpec(
        symbol=args.symbol,
        timeframe=Timeframe[args.timeframe],
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )

    session = get_session_factory()()
    try:
        result = run_backtest(session, strategy, run_spec, backtest_config)
    finally:
        session.close()

    m = result.metrics
    print(f"\nStrategy: {m.strategy_name}  Instrument: {m.instrument_symbol}")
    print(f"Period:   {m.test_period_start} -> {m.test_period_end}")
    print(f"Trades:   {m.trade_count}   Win rate: {m.win_rate}   Expectancy: {m.expectancy}")
    print(f"Profit factor: {m.profit_factor}   Total return: {m.total_return_pct:.2f}%   Max DD: {m.max_drawdown_pct:.2f}%")

    if m.trade_count == 0:
        print("\nINSUFFICIENT EVIDENCE: zero trades in this window - not logged as a comparable result.")
        print("(Widen the date range or check the strategy's entry conditions before treating this as a real backtest.)")
        return 0

    record = log_experiment(result, log_path=Path(args.log_path))
    print(f"\nLogged as {record.experiment_id} -> {args.log_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    records = compare_experiments(
        log_path=Path(args.log_path),
        strategy_name=args.strategy,
        instrument_symbol=args.symbol,
        sort_by=args.sort_by,
    )
    print(format_comparison_table(records))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-strategies", help="List all registered strategies").set_defaults(func=cmd_list_strategies)

    p_run = sub.add_parser("run", help="Run one backtest and log it as an experiment")
    p_run.add_argument("--strategy", required=True, help="Registered strategy name (see list-strategies)")
    p_run.add_argument("--symbol", required=True)
    p_run.add_argument("--timeframe", required=True, help="e.g. M5, M15, H1 (must match app.data.timeframes.Timeframe)")
    p_run.add_argument("--start", required=True, help="ISO date, e.g. 2024-01-01")
    p_run.add_argument("--end", required=True, help="ISO date, e.g. 2024-03-01")
    p_run.add_argument("--param", action="append", type=_parse_param, metavar="KEY=VALUE", help="Strategy config override, repeatable")
    p_run.add_argument("--spread-points", dest="spread_points", type=float, default=0.0)
    p_run.add_argument("--commission", type=float, default=0.0)
    p_run.add_argument("--risk-pct", dest="risk_pct", type=float, default=1.0)
    p_run.add_argument("--initial-equity", dest="initial_equity", type=float, default=10_000.0)
    p_run.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Compare logged experiments")
    p_cmp.add_argument("--strategy", default=None)
    p_cmp.add_argument("--symbol", default=None)
    p_cmp.add_argument("--sort-by", default="expectancy", choices=["expectancy", "win_rate", "profit_factor", "total_return_pct", "sharpe_ratio"])
    p_cmp.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
