"""Cross-validation between `BacktestEngine` (hand-rolled, executed and
trusted in this codebase's own test run) and `VectorBTEngine` (written
against documentation only, not executed here — see vbt_engine.py's
module docstring).

This test could NOT be run in the development sandbox (no network to
`pip install vectorbt`). Run it the moment you have vectorbt installed,
BEFORE trusting VectorBTEngine for anything. It is intentionally
tolerant on fields VectorBTEngine can't reconstruct (see that module's
docstring: stop_loss/take_profit/r_multiple/exit_reason) and strict on
the fields that actually determine whether the simulation is correct:
entry/exit timing and price, direction, and net P&L sign and magnitude.

A mismatch here means one of the two engines has a bug. Given
`BacktestEngine` has actually executed test coverage in this repo and
`VectorBTEngine` does not, treat `BacktestEngine`'s output as the
reference and investigate `VectorBTEngine` first — not the reverse.
"""

from __future__ import annotations

import pandas as pd
import pytest

vbt = pytest.importorskip("vectorbt", reason="vectorbt not installed - install it to run engine parity checks")

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.vbt_engine import VectorBTEngine
from app.strategies.base import Signal


def _frame(opens, highs, lows, closes, start="2024-01-01 09:00"):
    idx = pd.date_range(start, periods=len(opens), freq="5min", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


class TestEngineParity:
    def test_single_long_take_profit_agrees(self) -> None:
        df = _frame(
            opens=[100, 101, 102, 103, 104, 105],
            highs=[101, 102, 103, 150, 105, 106],
            lows=[99, 100, 101, 102, 103, 104],
            closes=[100.5] * 6,
        )
        sig = Signal(signal_time=df.index[1], direction="long", stop_loss=90.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X", initial_equity=10_000.0, risk_per_trade_pct=1.0)
        params = {"risk_reward": 1.0, "instrument_symbol": "X"}

        reference = BacktestEngine(cfg).run(df, [sig], "t", params)
        candidate = VectorBTEngine(cfg).run(df, [sig], "t", params)

        assert len(reference) == len(candidate) == 1
        r, c = reference[0], candidate[0]
        assert r.direction == c.direction
        assert r.entry_time == c.entry_time
        assert r.entry_price == pytest.approx(c.entry_price, rel=1e-6)
        assert r.exit_time == c.exit_time
        assert r.net_pnl == pytest.approx(c.net_pnl, rel=1e-3)

    def test_single_short_stop_loss_agrees(self) -> None:
        df = _frame(
            opens=[100, 100, 100, 100, 100],
            highs=[101, 101, 101, 101, 120],
            lows=[99, 99, 99, 60, 99],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[1], direction="short", stop_loss=110.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X", initial_equity=10_000.0, risk_per_trade_pct=1.0)
        params = {"risk_reward": 1.0, "instrument_symbol": "X"}

        reference = BacktestEngine(cfg).run(df, [sig], "t", params)
        candidate = VectorBTEngine(cfg).run(df, [sig], "t", params)

        assert len(reference) == len(candidate) == 1
        r, c = reference[0], candidate[0]
        assert r.direction == c.direction
        assert r.net_pnl == pytest.approx(c.net_pnl, rel=1e-3)

    def test_ambiguous_same_bar_resolves_the_same_way(self) -> None:
        """Both engines default to stop-loss-first on an ambiguous bar -
        this is the one property that must never silently diverge,
        since it's a conservatism assumption, not just an implementation
        detail."""
        df = _frame(
            opens=[100, 101, 102, 103, 104],
            highs=[101, 102, 200, 104, 105],
            lows=[99, 100, 10, 102, 103],
            closes=[100.5] * 5,
        )
        sig = Signal(signal_time=df.index[0], direction="long", stop_loss=95.0, strategy_name="t")
        cfg = BacktestConfig(instrument_symbol="X", initial_equity=10_000.0, risk_per_trade_pct=1.0)
        params = {"risk_reward": 1.0, "instrument_symbol": "X"}

        reference = BacktestEngine(cfg).run(df, [sig], "t", params)
        candidate = VectorBTEngine(cfg).run(df, [sig], "t", params)

        assert reference[0].exit_reason == "stop_loss"
        # VectorBTEngine can't attribute exit_reason (see its docstring) -
        # the price/outcome, not the label, is what must match here.
        assert candidate[0].exit_price == pytest.approx(reference[0].exit_price, rel=1e-6)
        assert candidate[0].net_pnl == pytest.approx(reference[0].net_pnl, rel=1e-3)
