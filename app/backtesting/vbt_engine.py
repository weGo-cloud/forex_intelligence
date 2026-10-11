"""VectorBT-backed backtest engine — implements the exact same interface
as `app.backtesting.engine.BacktestEngine` (Signal list in, Trade list
out), so it is a drop-in replacement wherever `BacktestEngine` is used
(including `runner.run_backtest`).

WHY THIS EXISTS ALONGSIDE THE HAND-ROLLED ENGINE, NOT INSTEAD OF IT:
The roadmap's own principle is "rent the boring infrastructure, build
the intelligence." Order/fill/SL-TP simulation and portfolio bookkeeping
are exactly the boring, well-solved part; VectorBT is mature, actively
maintained (confirmed via the project's GitHub activity — recent
releases add an optional Rust-accelerated engine), and gets you
vectorized parameter sweeps and walk-forward at a speed the hand-rolled
Python loop in `engine.py` cannot match — which the roadmap's own
robustness/walk-forward requirements will eventually need.

VERIFICATION STATUS — read this before trusting this file in production:
This module's API usage (parameter names, defaults, and the documented
same-bar SL/TP conflict resolution) was checked against current,
dated VectorBT documentation and GitHub discussions via web search, NOT
by executing it — this sandbox has no network access to `pip install
vectorbt`. That is a materially different (weaker) confidence level
than the rest of this codebase, which was actually run. Two things are
confirmed reliable from documentation:
  1. Default same-bar SL+TP conflict resolution is "stop-loss-first
     (pessimistic)" - matches this project's own engine.py default.
  2. The standard fix for same-bar-close lookahead is to shift entries
     forward and fill at the next bar's open - matches engine.py's
     fill rule.
Before relying on this engine for real research: (a) `pip install
vectorbt`, (b) run `tests/backtesting/test_engine_parity.py`, which
runs identical signals through both engines and asserts trade-level
agreement, and (c) treat any mismatch as this file being wrong, not
`engine.py` being wrong — `engine.py` is the one with executed test
coverage.

KNOWN BEHAVIORAL DIFFERENCES FROM `engine.py` (not bugs - deliberate,
documented simplifications to keep this integration tractable):
  - Position sizing here is percent-of-INITIAL-equity, not
    percent-of-CURRENT-(compounding)-equity. True dynamic compounding
    sizing in VectorBT requires the lower-level `from_order_func`
    callback API, which is materially more complex to get right
    without being able to execute it here. This is a real difference
    in trade sizes for strategies with many sequential trades, and
    thus in headline equity-curve numbers - it does not affect
    per-trade win/loss classification, R-multiples, or SL/TP timing,
    which are the correctness-critical properties the causality tests
    care about.
  - Spread is approximated as a `slippage` fraction (VectorBT has no
    native fixed-price-point spread primitive) rather than applied as
    an exact half-spread in price units the way `engine.py` does. For
    instruments where absolute spread doesn't scale with price, this
    is an approximation, not an equivalent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.backtesting.config import BacktestConfig
from app.backtesting.results import Trade
from app.strategies.base import Signal


class VectorBTEngine:
    """Same interface as `app.backtesting.engine.BacktestEngine`."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(
        self,
        ohlcv: pd.DataFrame,
        signals: list[Signal],
        strategy_name: str,
        strategy_params: dict[str, Any],
    ) -> list[Trade]:
        try:
            import vectorbt as vbt
        except ImportError as exc:
            raise ImportError(
                "vectorbt is not installed. This engine is an optional alternative to "
                "app.backtesting.engine.BacktestEngine - install with `pip install vectorbt` "
                "or use BacktestEngine instead (no extra dependency)."
            ) from exc

        if ohlcv.empty or not signals:
            return []

        cfg = self.config
        index = ohlcv.index
        n = len(index)
        pos_by_ts = {ts: i for i, ts in enumerate(index)}
        opens = ohlcv["open"].to_numpy(dtype=float)

        long_entries = np.zeros(n, dtype=bool)
        short_entries = np.zeros(n, dtype=bool)
        sl_stop = np.full(n, np.nan)
        tp_stop = np.full(n, np.nan)
        size = np.full(n, np.nan)

        risk_reward = float(strategy_params.get("risk_reward", 2.0))
        risk_amount = cfg.initial_equity * cfg.risk_per_trade_pct / 100.0  # fixed - see module docstring

        # sl_stop/tp_stop in VectorBT are FRACTIONS of the entry price,
        # not absolute price levels (confirmed via documentation - e.g.
        # sl_stop=0.025 means 2.5%), so each signal's absolute
        # stop_loss must be converted using the already-known raw open
        # of its (already-causal) fill bar. This is not a leak: by the
        # time we build these arrays we already possess the full
        # historical OHLCV: referencing a fill bar's own open to size
        # ITS OWN stop distance is exactly what engine.py does too.
        last_signal_fill_pos = -1  # one-trade-at-a-time enforcement, same rule as engine.py
        for signal in sorted(signals, key=lambda s: s.signal_time):
            if signal.direction == "short" and not cfg.allow_short:
                continue
            signal_pos = pos_by_ts.get(signal.signal_time)
            if signal_pos is None:
                continue
            entry_pos = signal_pos + 1
            if entry_pos >= n:
                continue
            if cfg.one_trade_at_a_time and entry_pos <= last_signal_fill_pos:
                continue

            raw_entry = opens[entry_pos]
            if np.isnan(raw_entry):
                continue
            risk_per_unit = abs(raw_entry - signal.stop_loss)
            if risk_per_unit <= 0:
                continue

            sl_frac = risk_per_unit / raw_entry
            tp_frac = risk_reward * sl_frac
            units = risk_amount / risk_per_unit

            if signal.direction == "long":
                long_entries[entry_pos] = True
            else:
                short_entries[entry_pos] = True
            sl_stop[entry_pos] = sl_frac
            tp_stop[entry_pos] = tp_frac
            size[entry_pos] = units
            last_signal_fill_pos = entry_pos  # approximate - true exit bar isn't known until vectorbt runs

        if not long_entries.any() and not short_entries.any():
            return []

        portfolio = vbt.Portfolio.from_signals(
            close=ohlcv["close"],
            open=ohlcv["open"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            entries=pd.Series(long_entries, index=index),
            short_entries=pd.Series(short_entries, index=index),
            size=pd.Series(size, index=index),
            size_type="amount",
            sl_stop=pd.Series(sl_stop, index=index),
            tp_stop=pd.Series(tp_stop, index=index),
            # Conservative same-bar conflict resolution to match engine.py's
            # default - confirmed via documentation this IS vectorbt's own
            # default behavior already, made explicit here for clarity.
            price=ohlcv["open"],  # entries already shifted to their fill bar - see loop above
            stop_entry_price="close",  # vectorbt default; see module docstring caveat
            stop_exit_price="stop",  # assume fill exactly at the stop/target level
            init_cash=cfg.initial_equity,
            fees=0.0,
            slippage=(cfg.spread_points / 2.0) / float(np.nanmean(opens)) if cfg.spread_points else 0.0,
            freq=None,
        )

        return self._trades_to_standard_form(portfolio, strategy_name, strategy_params)

    def _trades_to_standard_form(self, portfolio, strategy_name: str, strategy_params: dict[str, Any]) -> list[Trade]:
        """Map vectorbt's trade records into this project's standardized
        `Trade` shape.

        THIS IS THE WEAKEST PART OF THIS ADAPTER — flagged explicitly
        rather than glossed over:
        - Column names in `records_readable` have changed between
          vectorbt versions before. This reads defensively with
          fallbacks, but run `portfolio.trades.records_readable.columns`
          on your installed version and verify before trusting it.
        - `stop_loss`/`take_profit` are NOT set (NaN) - vectorbt's trade
          records don't directly expose the per-trade stop levels we
          supplied as fractions; recovering them would need tracking a
          side-table keyed by entry bar. Left as a known gap rather than
          a fabricated value.
        - `exit_reason` is NOT reliably recoverable from
          `records_readable` alone (a win could be a take-profit OR a
          discretionary exit; a loss could be a stop-loss OR an
          end-of-data exit at an unfavorable price) - set to `"unknown"`
          rather than guessed from the sign of PnL, which would silently
          misclassify max-holding-period and end-of-data exits as
          stop-loss/take-profit hits. If you need reliable exit
          attribution, use `vectorbt`'s lower-level order records or the
          `Status`/exit-type columns your installed version actually
          exposes, and wire that through explicitly - don't infer it.
        - `r_multiple` is NOT computed here for the same reason: it
          requires the original risk distance, which isn't in
          `records_readable`. Set to 0.0 as an explicit placeholder,
          not a real value - `compute_metrics` results that depend on
          R-multiples will be wrong until this is wired up properly.
        """
        records = portfolio.trades.records_readable
        trades: list[Trade] = []

        def col(row, *names, default=None):
            for name in names:
                if name in row.index:
                    return row[name]
            return default

        for _, row in records.iterrows():
            direction_raw = col(row, "Direction", "direction")
            direction = "short" if str(direction_raw).lower().startswith("short") else "long"
            entry_price = float(col(row, "Avg Entry Price", "entry_price"))
            exit_price = float(col(row, "Avg Exit Price", "exit_price"))
            size_val = float(col(row, "Size", "size"))
            pnl = float(col(row, "PnL", "pnl"))
            entry_time = pd.Timestamp(col(row, "Entry Timestamp", "entry_idx")).to_pydatetime()
            exit_time = pd.Timestamp(col(row, "Exit Timestamp", "exit_idx")).to_pydatetime()

            trades.append(
                Trade(
                    strategy_name=strategy_name,
                    instrument_symbol=strategy_params.get("instrument_symbol", ""),
                    direction=direction,  # type: ignore[arg-type]
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=float("nan"),  # not recoverable from records_readable - see docstring
                    take_profit=float("nan"),  # not recoverable from records_readable - see docstring
                    position_size=size_val,
                    gross_pnl=pnl,
                    net_pnl=pnl,
                    r_multiple=0.0,  # PLACEHOLDER, not a real value - see docstring
                    exit_reason="unknown",  # type: ignore[arg-type]  # not reliably recoverable - see docstring
                    holding_bars=int(col(row, "Duration", "duration", default=0)),
                    strategy_params=strategy_params,
                    signal_meta={},
                )
            )

        return trades
