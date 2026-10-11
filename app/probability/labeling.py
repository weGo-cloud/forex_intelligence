"""Deterministic forward outcome labeling: triple-barrier logic
(take-profit / stop-loss / time), direction-aware (one shared
implementation for long and short, per the M5 mission's explicit
"do not duplicate the entire implementation for each direction"), with
an explicit, configurable same-bar collision policy.

ENTRY SEMANTICS (M5 mission section 9): entry fills at the OPEN of the
bar AFTER `signal.signal_time`, and barrier monitoring begins the bar
AFTER THAT - identical to `app.backtesting.engine.BacktestEngine`'s own
`exits_start_bar_after_entry=True` (non-configurable) convention. This
is a DELIBERATE match, not a coincidence: a signal generated after a
candle closes must not enter at an earlier price, and the M5 mission
explicitly says to reuse M2's timing convention if it already exists.
It is a separate implementation from `BacktestEngine.run` (not a call
into it) because the two solve different problems - `BacktestEngine`
runs a whole EQUITY CURVE across many signals with position sizing and
`one_trade_at_a_time` exclusivity; M5 labels EVERY signal's forward
outcome independently (including overlapping ones - M5 mission section
23), with no equity/position-sizing concept, and needs an
`UNRESOLVED`-capable same-bar policy `BacktestConfig.same_bar_sl_tp_policy`
does not offer (it only supports `"stop_loss_first"`/`"take_profit_first"`
- both of which silently pick a winner even when OHLC data cannot
actually tell you which barrier was touched first; M5's whole purpose
is honest statistics, so guessing here would inflate probabilities
in a way M2's engine, built for a different purpose, does not need to
avoid as strictly).

CAUSALITY: a bar at position i is evaluated (for the barrier check)
using only that bar's own high/low - never a later bar. Nothing here
looks beyond position `entry_pos + evaluation_horizon_bars`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from app.backtesting.results import Direction
from app.probability.models import ExitReason, OutcomeLabel, TradeOutcome
from app.regimes.context import RegimeContext
from app.strategies.base import Signal

SameBarCollisionPolicy = Literal["stop_first", "target_first", "unresolved"]


class InvalidSignalError(ValueError):
    """Raised for a structurally degenerate signal (e.g. stop_loss ==
    entry_price, or a take_profit on the wrong side of entry) - the
    same class of problem `BacktestEngine` silently skips with
    `continue` in a bulk backtest loop; here it's raised explicitly so
    a single-signal caller finds out, and `label_outcomes_bulk` decides
    whether to skip or propagate (default: skip, matching the engine's
    own behavior)."""


@dataclass(frozen=True)
class LabelingConfig:
    evaluation_horizon_bars: int = 50
    same_bar_policy: SameBarCollisionPolicy = "unresolved"
    # Only used when a Signal doesn't set its own take_profit (mirrors
    # BacktestEngine's risk_reward fallback - see app/backtesting/engine.py).
    default_risk_reward: float = 2.0
    # A resolved (TIME_EXPIRY) r_multiple within +/- this of exactly 0.0
    # is BREAKEVEN rather than a razor-thin WIN/LOSS - guards against
    # float noise turning an economically-flat outcome into a coin-flip
    # label. Deliberately tiny by default (not a "close enough" fudge
    # factor) - override explicitly if a wider breakeven band is wanted.
    breakeven_r_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.evaluation_horizon_bars < 1:
            raise ValueError("evaluation_horizon_bars must be >= 1")
        if self.default_risk_reward <= 0:
            raise ValueError("default_risk_reward must be > 0")
        if self.breakeven_r_epsilon < 0:
            raise ValueError("breakeven_r_epsilon must be >= 0")


def _resolve_take_profit(signal: Signal, entry_price: float, risk_per_unit: float, config: LabelingConfig) -> float:
    if signal.take_profit is not None:
        return signal.take_profit
    direction_sign = 1.0 if signal.direction == "long" else -1.0
    return entry_price + direction_sign * config.default_risk_reward * risk_per_unit


def label_outcome(
    signal: Signal,
    execution_df: pd.DataFrame,
    config: LabelingConfig,
    instrument_symbol: str,
    timeframe: str,
    regime_context: Optional[RegimeContext] = None,
) -> TradeOutcome:
    """Label a single signal's forward outcome. `execution_df` must
    contain `signal.signal_time` in its index (raises `KeyError`
    otherwise - the caller passed a signal/data pair that don't belong
    together, not a labeling edge case to paper over).
    """
    if signal.signal_time not in execution_df.index:
        raise KeyError(f"signal_time {signal.signal_time} not found in execution_df index")

    signal_pos = execution_df.index.get_loc(signal.signal_time)
    if not isinstance(signal_pos, (int,)):
        # get_loc can return a slice/array for duplicate index labels -
        # a data-quality problem this module does not attempt to repair.
        raise ValueError(f"signal_time {signal.signal_time} is not uniquely located in execution_df's index")

    entry_pos = signal_pos + 1
    if entry_pos >= len(execution_df):
        # No bar exists to fill on at all - not even an entry happened.
        last_ts = execution_df.index[-1]
        return TradeOutcome(
            strategy_name=signal.strategy_name,
            instrument_symbol=instrument_symbol,
            timeframe=timeframe,
            direction=signal.direction,
            signal_timestamp=signal.signal_time,
            entry_timestamp=None,
            exit_timestamp=last_ts,
            entry_price=None,
            exit_price=None,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit if signal.take_profit is not None else signal.stop_loss,
            r_multiple=None,
            outcome_label=OutcomeLabel.UNRESOLVED,
            exit_reason=ExitReason.INSUFFICIENT_DATA,
            evaluation_horizon_bars=config.evaluation_horizon_bars,
            bars_to_exit=None,
            regime_context=regime_context,
            signal_meta=dict(signal.meta),
        )

    entry_price = float(execution_df["open"].iloc[entry_pos])
    entry_timestamp = execution_df.index[entry_pos]
    risk_per_unit = abs(entry_price - signal.stop_loss)
    if risk_per_unit == 0:
        raise InvalidSignalError(f"stop_loss ({signal.stop_loss}) equals entry_price ({entry_price})")

    take_profit = _resolve_take_profit(signal, entry_price, risk_per_unit, config)
    direction_sign = 1.0 if signal.direction == "long" else -1.0
    if (signal.direction == "long" and take_profit <= entry_price) or (
        signal.direction == "short" and take_profit >= entry_price
    ):
        raise InvalidSignalError(f"take_profit ({take_profit}) is on the wrong side of entry_price ({entry_price})")
    if (signal.direction == "long" and signal.stop_loss >= entry_price) or (
        signal.direction == "short" and signal.stop_loss <= entry_price
    ):
        raise InvalidSignalError(f"stop_loss ({signal.stop_loss}) is on the wrong side of entry_price ({entry_price})")

    horizon_end_pos = min(entry_pos + config.evaluation_horizon_bars, len(execution_df) - 1)
    scan_start = entry_pos + 1  # barrier monitoring begins the bar AFTER entry - see module docstring

    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    exit_pos: Optional[int] = None

    for pos in range(scan_start, horizon_end_pos + 1):
        bar = execution_df.iloc[pos]
        if signal.direction == "long":
            hit_tp = bar["high"] >= take_profit
            hit_sl = bar["low"] <= signal.stop_loss
        else:
            hit_tp = bar["low"] <= take_profit
            hit_sl = bar["high"] >= signal.stop_loss

        if hit_tp and hit_sl:
            if config.same_bar_policy == "target_first":
                exit_price, exit_reason = take_profit, ExitReason.TAKE_PROFIT
            elif config.same_bar_policy == "stop_first":
                exit_price, exit_reason = signal.stop_loss, ExitReason.STOP_LOSS
            else:  # "unresolved" - the honest default; see module docstring
                exit_price, exit_reason = None, ExitReason.SAME_BAR_UNRESOLVED
            exit_pos = pos
            break
        if hit_tp:
            exit_price, exit_reason = take_profit, ExitReason.TAKE_PROFIT
            exit_pos = pos
            break
        if hit_sl:
            exit_price, exit_reason = signal.stop_loss, ExitReason.STOP_LOSS
            exit_pos = pos
            break

    if exit_reason is None:
        # Neither barrier was ever hit within the scanned range.
        if horizon_end_pos < entry_pos + config.evaluation_horizon_bars:
            # We wanted a full horizon's worth of bars but real data ran out first.
            exit_reason = ExitReason.INSUFFICIENT_DATA
            exit_pos = horizon_end_pos
            exit_price = None
        else:
            exit_reason = ExitReason.TIME_EXPIRY
            exit_pos = horizon_end_pos
            exit_price = float(execution_df["close"].iloc[horizon_end_pos])

    exit_timestamp = execution_df.index[exit_pos]
    bars_to_exit = exit_pos - entry_pos

    r_multiple: Optional[float] = None
    if exit_price is not None:
        r_multiple = direction_sign * (exit_price - entry_price) / risk_per_unit

    outcome_label = _label_from_exit_reason(exit_reason, r_multiple, config)

    return TradeOutcome(
        strategy_name=signal.strategy_name,
        instrument_symbol=instrument_symbol,
        timeframe=timeframe,
        direction=signal.direction,
        signal_timestamp=signal.signal_time,
        entry_timestamp=entry_timestamp,
        exit_timestamp=exit_timestamp,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss=signal.stop_loss,
        take_profit=take_profit,
        r_multiple=r_multiple,
        outcome_label=outcome_label,
        exit_reason=exit_reason,
        evaluation_horizon_bars=config.evaluation_horizon_bars,
        bars_to_exit=bars_to_exit,
        regime_context=regime_context,
        signal_meta=dict(signal.meta),
    )


def _label_from_exit_reason(exit_reason: ExitReason, r_multiple: Optional[float], config: LabelingConfig) -> OutcomeLabel:
    if exit_reason == ExitReason.TAKE_PROFIT:
        return OutcomeLabel.WIN
    if exit_reason == ExitReason.STOP_LOSS:
        return OutcomeLabel.LOSS
    if exit_reason in (ExitReason.SAME_BAR_UNRESOLVED, ExitReason.INSUFFICIENT_DATA):
        return OutcomeLabel.UNRESOLVED
    # TIME_EXPIRY: resolved by whichever side of breakeven the closing r_multiple landed on.
    assert r_multiple is not None  # TIME_EXPIRY always has a real exit_price
    if r_multiple > config.breakeven_r_epsilon:
        return OutcomeLabel.WIN
    if r_multiple < -config.breakeven_r_epsilon:
        return OutcomeLabel.LOSS
    return OutcomeLabel.BREAKEVEN


def label_outcomes_bulk(
    signals: list[Signal],
    execution_df: pd.DataFrame,
    config: LabelingConfig,
    instrument_symbol: str,
    timeframe: str,
    regime_context_by_signal_time: Optional[dict[pd.Timestamp, RegimeContext]] = None,
    skip_invalid: bool = True,
) -> list[TradeOutcome]:
    """Label every signal in `signals` independently (overlapping
    signals are fully supported - each is evaluated against
    `execution_df` on its own, with no `one_trade_at_a_time`-style
    exclusivity, since M5 has no equity-curve/position concept to make
    exclusivity meaningful for - see module docstring). Signals whose
    `signal_time` isn't in the index, or that are structurally
    degenerate (`InvalidSignalError`), are skipped when `skip_invalid`
    is True (the default, matching `BacktestEngine`'s own skip-and-
    continue behavior for degenerate signals) - set False to propagate
    the exception instead.
    """
    outcomes: list[TradeOutcome] = []
    for signal in signals:
        regime_context = (regime_context_by_signal_time or {}).get(signal.signal_time)
        try:
            outcomes.append(
                label_outcome(signal, execution_df, config, instrument_symbol, timeframe, regime_context)
            )
        except (InvalidSignalError, KeyError, ValueError):
            if skip_invalid:
                continue
            raise
    return outcomes
