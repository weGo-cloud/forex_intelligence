from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.backtesting.metrics import compute_metrics
from app.backtesting.results import Trade

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 2, 1, tzinfo=timezone.utc)


def _trade(pnl: float, direction: str = "long", holding: int = 3) -> Trade:
    return Trade(
        strategy_name="s",
        instrument_symbol="X",
        direction=direction,
        entry_time=START,
        exit_time=START,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        stop_loss=90.0,
        take_profit=110.0,
        position_size=1.0,
        gross_pnl=pnl,
        net_pnl=pnl,
        r_multiple=pnl / 10.0,
        exit_reason="take_profit" if pnl > 0 else "stop_loss",
        holding_bars=holding,
        strategy_params={},
        signal_meta={},
    )


class TestComputeMetrics:
    def test_basic_win_loss_counts_and_rates(self) -> None:
        trades = [_trade(100), _trade(-50), _trade(100), _trade(-50), _trade(-50)]
        m = compute_metrics(trades, "s", "X", 10_000.0, START, END)

        assert m.trade_count == 5
        assert m.winning_trades == 2
        assert m.losing_trades == 3
        assert m.win_rate == pytest.approx(0.4)
        assert m.avg_win == pytest.approx(100.0)
        assert m.avg_loss == pytest.approx(-50.0)
        assert m.profit_factor == pytest.approx(200 / 150)

    def test_empty_trade_list_returns_none_not_zero_or_inf(self) -> None:
        m = compute_metrics([], "s", "X", 10_000.0, START, END)
        assert m.trade_count == 0
        assert m.win_rate is None
        assert m.profit_factor is None
        assert m.sharpe_ratio is None
        assert m.avg_win is None
        assert m.avg_loss is None
        assert m.total_return_pct == 0.0

    def test_no_losing_trades_gives_undefined_profit_factor_not_infinity(self) -> None:
        trades = [_trade(100), _trade(50), _trade(75)]
        m = compute_metrics(trades, "s", "X", 10_000.0, START, END)
        assert m.profit_factor is None
        assert m.losing_trades == 0

    def test_single_trade_gives_undefined_sharpe(self) -> None:
        m = compute_metrics([_trade(100)], "s", "X", 10_000.0, START, END)
        assert m.sharpe_ratio is None

    def test_directional_split(self) -> None:
        trades = [_trade(100, "long"), _trade(-50, "long"), _trade(80, "short")]
        m = compute_metrics(trades, "s", "X", 10_000.0, START, END)
        assert m.long_stats.trade_count == 2
        assert m.short_stats.trade_count == 1
        assert m.short_stats.win_rate == 1.0

    def test_total_return_and_drawdown_reflect_equity_path(self) -> None:
        trades = [_trade(1000), _trade(-2000), _trade(1500)]
        m = compute_metrics(trades, "s", "X", 10_000.0, START, END)
        # equity path: 10000 -> 11000 -> 9000 -> 10500
        assert m.total_return_pct == pytest.approx((10500 - 10000) / 10000 * 100)
        assert m.max_drawdown_pct < 0  # drawdown occurred at the 11000 -> 9000 dip
