"""TEMPLATE — copy this file to app/strategies/<your_strategy_name>.py
and fill in the TODOs. Delete this docstring paragraph once done.

This exists so step 2 of the research workflow ("Implement strategy")
starts from a working skeleton that already satisfies the Strategy
protocol and is already wired for registration, instead of every new
strategy reinventing the same boilerplate (or worse, subtly deviating
from the causality rules the engine depends on).

Before writing any code here, do step 1 properly: write down the
EXACT rule in plain language - entry condition, stop-loss method,
take-profit method, what's configurable, what's a fixed default and
why. If you can't state the rule as a deterministic, objective
procedure (see app/strategies/orb.py and rsi_divergence.py module
docstrings for the standard this repo holds itself to), you're not
ready to implement it yet - "vague visual logic" was explicitly
rejected for both existing strategies for a reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.strategies.base import MarketContext, Signal
from app.strategies.registry import register_strategy


@dataclass(frozen=True)
class TemplateStrategyConfig:
    instrument_symbol: str
    # TODO: every tunable parameter of your rule goes here, with a
    # documented default. No parameter should be a hard-coded constant
    # buried in generate_signals() below - if it affects the rule's
    # behavior, it belongs here (see ORBConfig for the standard).
    risk_reward: float = 2.0

    def __post_init__(self) -> None:
        # TODO: validate every field the same way ORBConfig/
        # RSIDivergenceConfig do - fail loudly on nonsensical config
        # (negative risk_reward, contradictory flags) at construction
        # time, not silently at backtest time.
        if self.risk_reward <= 0:
            raise ValueError("risk_reward must be > 0")


class TemplateStrategy:
    def __init__(self, config: TemplateStrategyConfig) -> None:
        self.config = config
        self.name = "template"  # TODO: rename to match your strategy

    def params(self) -> dict[str, Any]:
        # TODO: return every config field - this is what gets recorded
        # with every trade and every logged experiment. Missing a field
        # here means a future comparison can't tell two runs apart.
        return {
            "instrument_symbol": self.config.instrument_symbol,
            "risk_reward": self.config.risk_reward,
        }

    def generate_signals(self, market: MarketContext) -> list[Signal]:
        """TODO: implement your rule here.

        CAUSALITY CHECKLIST — every item below has caused a real bug in
        this codebase's own strategies during development; check all of
        them before considering this method done:

        [ ] Every decision at bar i uses only data from bars <= i.
        [ ] If your rule involves a "confirmed" pattern (a pivot, a
            range, a breakout), the Signal's signal_time is the bar
            where the pattern becomes KNOWABLE, not the bar the
            pattern's own price level was set on. See
            app/features/indicators.py::detect_swing_points and its
            `confirmed_pos` field for the pattern to follow if your
            rule needs anything like a local high/low.
        [ ] You only set `take_profit` explicitly when your rule targets
            a specific price level (e.g. "previous day's high/low") -
            otherwise leave it unset and the engine derives it from
            `risk_reward` and the actual fill price. `stop_loss` is
            always an absolute price you must set.
        [ ] If you use `market.context` (higher-timeframe data), it is
            already causally aligned by data_access.align_higher_timeframe
            - do not re-derive HTF values from execution_df yourself.
        [ ] Write a bias test for this strategy modeled on
            tests/backtesting/test_bias.py before considering it done -
            not optional, see docs/ADDING_A_STRATEGY.md step 3.

        Return signals in ascending signal_time order (or don't bother
        sorting - the engine sorts them again defensively, but do it
        anyway for readability).
        """
        df = market.execution_df
        signals: list[Signal] = []

        # TODO: your rule.
        # Example skeleton:
        # for ts, bar in df.iterrows():
        #     if <your objective entry condition using bar and prior bars only>:
        #         signals.append(
        #             Signal(
        #                 signal_time=ts,
        #                 direction="long",  # or "short"
        #                 stop_loss=<absolute price>,
        #                 strategy_name=self.name,
        #                 meta={},  # anything useful for later investigation
        #             )
        #         )

        return signals


# TODO: pick a unique registry name and a one-line description.
@register_strategy("template", TemplateStrategyConfig, "TODO: describe the rule in one line")
def _build_template(**kwargs) -> TemplateStrategy:
    return TemplateStrategy(TemplateStrategyConfig(**kwargs))
