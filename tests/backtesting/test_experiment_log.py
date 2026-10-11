from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.backtesting.experiment_log import (
    compare_experiments,
    format_comparison_table,
    load_experiments,
    log_experiment,
)
from app.backtesting.results import BacktestMetrics, BacktestResult, DirectionalStats


def _result(strategy: str, expectancy, instrument: str = "US30", trade_count: int = 10) -> BacktestResult:
    m = BacktestMetrics(
        strategy_name=strategy,
        instrument_symbol=instrument,
        test_period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        test_period_end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        trade_count=trade_count,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        avg_trade=expectancy,
        avg_win=100.0,
        avg_loss=-50.0,
        expectancy=expectancy,
        profit_factor=1.5,
        total_return_pct=5.0,
        max_drawdown_pct=-3.0,
        sharpe_ratio=1.2,
        avg_holding_bars=4.0,
        long_stats=DirectionalStats(trade_count=trade_count, win_rate=0.6, avg_trade=expectancy, profit_factor=1.5),
        short_stats=DirectionalStats(trade_count=0, win_rate=None, avg_trade=None, profit_factor=None),
    )
    return BacktestResult(
        metrics=m, trades=[], strategy_params={"risk_reward": 2.0}, backtest_config={"spread_points": 1.0}, assumptions=["a"]
    )


class TestExperimentLog:
    def test_log_and_load_roundtrip(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        record = log_experiment(_result("orb", 15.0), log_path=log_path)
        assert record.experiment_id == "exp_0001"

        loaded = load_experiments(log_path)
        assert len(loaded) == 1
        assert loaded[0].strategy_name == "orb"
        assert loaded[0].expectancy == 15.0

    def test_ids_are_sequential(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        r1 = log_experiment(_result("orb", 10.0), log_path=log_path)
        r2 = log_experiment(_result("orb", 20.0), log_path=log_path)
        r3 = log_experiment(_result("orb", 30.0), log_path=log_path)
        assert [r1.experiment_id, r2.experiment_id, r3.experiment_id] == ["exp_0001", "exp_0002", "exp_0003"]

    def test_compare_sorts_descending_by_default_metric(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_experiment(_result("orb", 10.0), log_path=log_path)
        log_experiment(_result("orb", 30.0), log_path=log_path)
        log_experiment(_result("orb", 20.0), log_path=log_path)

        ranked = compare_experiments(log_path=log_path, sort_by="expectancy")
        assert [r.expectancy for r in ranked] == [30.0, 20.0, 10.0]

    def test_compare_filters_by_strategy_and_instrument(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_experiment(_result("orb", 10.0, instrument="US30"), log_path=log_path)
        log_experiment(_result("rsi_divergence", 20.0, instrument="US30"), log_path=log_path)
        log_experiment(_result("orb", 30.0, instrument="EURUSD"), log_path=log_path)

        orb_only = compare_experiments(log_path=log_path, strategy_name="orb")
        assert len(orb_only) == 2

        orb_us30_only = compare_experiments(log_path=log_path, strategy_name="orb", instrument_symbol="US30")
        assert len(orb_us30_only) == 1
        assert orb_us30_only[0].expectancy == 10.0

    def test_undefined_metric_sorts_last_not_as_zero(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        with_pf = _result("orb", 10.0)
        log_experiment(with_pf, log_path=log_path)

        # A record with profit_factor=None (e.g. no losing trades yet)
        # must not be treated as worse than a real negative profit_factor
        # by silently coercing None to 0 - it should sort after both.
        undefined = _result("orb", 5.0)
        undefined.metrics.profit_factor = None
        negative = _result("orb", 5.0)
        negative.metrics.profit_factor = -2.0
        log_experiment(undefined, log_path=log_path)
        log_experiment(negative, log_path=log_path)

        ranked = compare_experiments(log_path=log_path, sort_by="profit_factor")
        assert ranked[-1].profit_factor is None

    def test_empty_log_returns_empty_list_and_readable_table(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        assert load_experiments(log_path) == []
        assert "no experiments" in format_comparison_table([]).lower()

    def test_format_comparison_table_handles_none_fields(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        r = _result("orb", 10.0)
        r.metrics.win_rate = None
        r.metrics.profit_factor = None
        log_experiment(r, log_path=log_path)
        table = format_comparison_table(load_experiments(log_path))
        assert "n/a" in table
