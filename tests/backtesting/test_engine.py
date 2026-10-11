from __future__ import annotations

import pandas as pd
import pytest

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.strategies.base import Signal


def _frame(opens, highs, lows, closes, start="2024-01-01 09:00"):
    idx = pd.date_range(start, periods=len(opens), freq="5min", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


class TestEntryFillPolicy:
    def test_entry_fills_at_next_bar_open_not_signal_bar_close(self) -> None:
        df = _frame(
            opens=[100, 101, 103, 104, 105],
            highs=[101, 102, 104, 105, 106],
            lows=[99, 100, 102, 103, 104],
            closes=[100.5, 101.5, 102.5, 104.5, 105.5],
        )
        sig = Signal(signal_time=df.index[1], direction="long", stop_loss=90.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X")
        trades = BacktestEngine(cfg).run(df, [sig], "t", {"risk_reward": 1.0})
        assert trades[0].entry_price == 103.0  # open of the bar AFTER the signal bar

    def test_signal_on_last_bar_produces_no_trade(self) -> None:
        df = _frame([100, 101], [101, 102], [99, 100], [100.5, 101.5])
        sig = Signal(signal_time=df.index[-1], direction="long", stop_loss=90.0, strategy_name="t")
        trades = BacktestEngine(BacktestConfig(instrument_symbol="X")).run(df, [sig], "t", {"risk_reward": 1.0})
        assert trades == []


class TestExitLogic:
    def test_stop_loss_hit(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104],
            highs=[101, 102, 103, 104, 105],
            lows=[99, 100, 101, 102, 95],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[2], direction="long", stop_loss=95.0, strategy_name="t")
        trades = BacktestEngine(BacktestConfig(instrument_symbol="X")).run(df, [sig], "t", {"risk_reward": 2.0})
        assert trades[0].exit_reason == "stop_loss"
        assert trades[0].exit_price == 95.0

    def test_take_profit_hit(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104],
            highs=[101, 102, 103, 150, 105],
            lows=[99, 100, 101, 102, 103],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[1], direction="long", stop_loss=90.0, strategy_name="t")
        trades = BacktestEngine(BacktestConfig(instrument_symbol="X")).run(df, [sig], "t", {"risk_reward": 1.0})
        # entry at pos2 open=102, risk=12, RR=1 -> TP=114; bar pos4 high=150 hits TP
        assert trades[0].exit_reason == "take_profit"
        assert trades[0].exit_price == 114.0

    def test_ambiguous_same_bar_defaults_to_stop_loss_first(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104],
            highs=[101, 102, 200, 104, 105],
            lows=[99, 100, 10, 102, 103],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[0], direction="long", stop_loss=95.0, strategy_name="t")
        trades = BacktestEngine(BacktestConfig(instrument_symbol="X")).run(df, [sig], "t", {"risk_reward": 1.0})
        assert trades[0].exit_reason == "stop_loss"

    def test_ambiguous_same_bar_can_be_configured_take_profit_first(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104],
            highs=[101, 102, 200, 104, 105],
            lows=[99, 100, 10, 102, 103],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[0], direction="long", stop_loss=95.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X", same_bar_sl_tp_policy="take_profit_first")
        trades = BacktestEngine(cfg).run(df, [sig], "t", {"risk_reward": 1.0})
        assert trades[0].exit_reason == "take_profit"

    def test_short_direction_stop_and_target_are_mirrored(self) -> None:
        df = _frame(
            opens=[100, 100, 100, 100, 100],
            highs=[101, 101, 101, 101, 120],
            lows=[99, 99, 99, 60, 99],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[1], direction="short", stop_loss=110.0, strategy_name="t")
        trades = BacktestEngine(BacktestConfig(instrument_symbol="X")).run(df, [sig], "t", {"risk_reward": 1.0})
        # entry at pos2 open=100, risk=10, TP = 100-10=90; bar3 low=60 hits TP
        assert trades[0].direction == "short"
        assert trades[0].exit_reason == "take_profit"
        assert trades[0].exit_price == 90.0

    def test_end_of_data_without_sl_or_tp_hit(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103],
            highs=[101, 102, 103, 104],
            lows=[99, 100, 101, 102],
            closes=[100.5, 101.5, 102.5, 103.5],
        )
        sig = Signal(signal_time=df.index[0], direction="long", stop_loss=50.0, strategy_name="t")
        trades = BacktestEngine(BacktestConfig(instrument_symbol="X")).run(df, [sig], "t", {"risk_reward": 100.0})
        assert trades[0].exit_reason == "end_of_data"
        assert trades[0].exit_price == pytest.approx(103.5)


class TestTradeManagement:
    def test_one_trade_at_a_time_skips_overlapping_signal(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104, 105, 106, 107],
            highs=[101, 102, 103, 104, 150, 106, 107, 108],
            lows=[99, 100, 101, 102, 96, 104, 105, 106],
            closes=[100.5] * 8,
        )
        sig_a = Signal(signal_time=df.index[0], direction="long", stop_loss=90.0, strategy_name="t")
        sig_b = Signal(signal_time=df.index[1], direction="long", stop_loss=90.0, strategy_name="t")
        trades = BacktestEngine(BacktestConfig(instrument_symbol="X")).run(df, [sig_a, sig_b], "t", {"risk_reward": 1.0})
        assert len(trades) == 1

    def test_short_disallowed_when_configured(self) -> None:
        df = _frame([100, 101], [101, 102], [99, 100], [100.5, 101.5])
        sig = Signal(signal_time=df.index[0], direction="short", stop_loss=110.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X", allow_short=False)
        trades = BacktestEngine(cfg).run(df, [sig], "t", {"risk_reward": 1.0})
        assert trades == []

    def test_degenerate_signal_stop_equals_entry_is_skipped(self) -> None:
        df = _frame([100, 100], [101, 101], [99, 99], [100.5, 100.5])
        sig = Signal(signal_time=df.index[0], direction="long", stop_loss=100.0, strategy_name="t")
        trades = BacktestEngine(BacktestConfig(instrument_symbol="X")).run(df, [sig], "t", {"risk_reward": 1.0})
        assert trades == []


class TestCosts:
    def test_spread_applied_against_trader_on_entry(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104],
            highs=[101, 102, 103, 104, 105],
            lows=[99, 100, 101, 102, 95],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[1], direction="long", stop_loss=90.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X", spread_points=2.0)
        trades = BacktestEngine(cfg).run(df, [sig], "t", {"risk_reward": 1.0})
        assert trades[0].entry_price == 103.0  # raw open 102 + half-spread 1.0

    def test_commission_reduces_net_pnl_but_not_gross(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104],
            highs=[101, 102, 103, 150, 105],
            lows=[99, 100, 101, 102, 103],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[1], direction="long", stop_loss=90.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X", commission_per_trade=5.0)
        trades = BacktestEngine(cfg).run(df, [sig], "t", {"risk_reward": 1.0})
        assert trades[0].net_pnl == pytest.approx(trades[0].gross_pnl - 5.0)


class TestPositionSizing:
    def test_risk_amount_matches_configured_percent(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104],
            highs=[101, 102, 103, 104, 105],
            lows=[99, 100, 101, 102, 95],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[1], direction="long", stop_loss=95.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X", initial_equity=10_000.0, risk_per_trade_pct=1.0)
        trades = BacktestEngine(cfg).run(df, [sig], "t", {"risk_reward": 1.0})
        # A full stop-loss hit should lose ~1% of equity (100, before commission)
        assert trades[0].exit_reason == "stop_loss"
        assert trades[0].net_pnl == pytest.approx(-100.0, abs=1.0)
        assert trades[0].r_multiple == pytest.approx(-1.0, abs=0.05)
