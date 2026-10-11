"""Causality and walk-forward-safety tests - the M5 mission's most
important requirement (section 10: "This is the most important M5
requirement"). Proves that a probability estimate made "as of" some
historical timestamp T cannot be changed by outcomes that only resolved
after T.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.probability.estimators import ProbabilityEstimator
from app.probability.labeling import LabelingConfig, label_outcomes_bulk
from app.probability.models import ExitReason, OutcomeLabel, TradeOutcome
from app.probability.outcomes import OutcomeDataset
from app.strategies.base import Signal


def _flat_df(n: int = 300, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(price, index=idx)
    return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close}, index=idx)


def _ts(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def _outcome(exit_ts: datetime, label: OutcomeLabel, r: float | None) -> TradeOutcome:
    reason = {OutcomeLabel.WIN: ExitReason.TAKE_PROFIT, OutcomeLabel.LOSS: ExitReason.STOP_LOSS}[label]
    return TradeOutcome(
        strategy_name="orb", instrument_symbol="EURUSD", timeframe="5M", direction="long",
        signal_timestamp=exit_ts, entry_timestamp=exit_ts, exit_timestamp=exit_ts,
        entry_price=100.0, exit_price=100.0 + (r or 0), stop_loss=98.0, take_profit=104.0,
        r_multiple=r, outcome_label=label, exit_reason=reason,
        evaluation_horizon_bars=20, bars_to_exit=5,
    )


class TestFutureOutcomesCannotAffectPastEstimate:
    def test_estimate_as_of_t_unaffected_by_outcomes_resolving_after_t(self) -> None:
        cutoff = _ts(5)
        dataset = OutcomeDataset.from_list(
            [_outcome(_ts(1), OutcomeLabel.WIN, 2.0), _outcome(_ts(3), OutcomeLabel.WIN, 2.0), _outcome(_ts(4), OutcomeLabel.LOSS, -1.0)]
        )
        baseline = ProbabilityEstimator().estimate(dataset, as_of=cutoff)

        # Append outcomes that resolve AFTER the cutoff - a flood of losses,
        # which would clearly move p_win if they leaked in.
        extended = OutcomeDataset.from_list(
            list(dataset) + [_outcome(_ts(6), OutcomeLabel.LOSS, -1.0) for _ in range(20)]
        )
        after = ProbabilityEstimator().estimate(extended, as_of=cutoff)

        assert baseline.counts == after.counts
        assert baseline.p_win == after.p_win
        assert baseline.expected_r == after.expected_r

    def test_estimate_as_of_t_includes_everything_resolved_exactly_at_t(self) -> None:
        cutoff = _ts(5)
        dataset = OutcomeDataset.from_list([_outcome(cutoff, OutcomeLabel.WIN, 2.0)])
        result = ProbabilityEstimator().estimate(dataset, as_of=cutoff)
        assert result.counts.total_observations == 1

    def test_no_as_of_uses_full_hindsight_population(self) -> None:
        dataset = OutcomeDataset.from_list([_outcome(_ts(1), OutcomeLabel.WIN, 2.0), _outcome(_ts(10), OutcomeLabel.LOSS, -1.0)])
        result = ProbabilityEstimator().estimate(dataset)  # no as_of - deliberately the full-hindsight query
        assert result.counts.total_observations == 2


class TestLabelerDoesNotUseDataBeyondHorizon:
    def test_appending_future_candles_does_not_change_an_already_resolved_outcome(self) -> None:
        df = _flat_df(n=40)
        df.loc[df.index[10], "high"] = 105.0  # TP hit at bar 10
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=104.0, strategy_name="orb")
        config = LabelingConfig(evaluation_horizon_bars=20)

        from app.probability.labeling import label_outcome

        baseline = label_outcome(sig, df, config, "EURUSD", "5M")

        extra_idx = pd.date_range(df.index[-1] + pd.Timedelta(minutes=5), periods=50, freq="5min", tz="UTC")
        shock = pd.Series(500.0, index=extra_idx)  # a wild, unrelated future shock
        shock_df = pd.DataFrame({"open": shock, "high": shock, "low": shock, "close": shock}, index=extra_idx)
        extended_df = pd.concat([df, shock_df])

        after = label_outcome(sig, extended_df, config, "EURUSD", "5M")

        assert baseline.exit_reason == after.exit_reason
        assert baseline.exit_price == after.exit_price
        assert baseline.exit_timestamp == after.exit_timestamp
        assert baseline.r_multiple == after.r_multiple

    def test_a_time_expiry_outcome_is_unaffected_by_data_after_its_horizon(self) -> None:
        df = _flat_df(n=40)
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=90.0, take_profit=200.0, strategy_name="orb")
        config = LabelingConfig(evaluation_horizon_bars=10)

        from app.probability.labeling import label_outcome

        baseline = label_outcome(sig, df, config, "EURUSD", "5M")

        modified_df = df.copy()
        # Change a bar WELL beyond the horizon (bar 30, horizon ends ~bar 15) drastically.
        modified_df.loc[modified_df.index[30], "close"] = 999.0
        after = label_outcome(sig, modified_df, config, "EURUSD", "5M")

        assert baseline.exit_price == after.exit_price
        assert baseline.outcome_label == after.outcome_label


class TestDeterminism:
    def test_identical_inputs_produce_identical_outcomes(self) -> None:
        df = _flat_df(n=100)
        df.loc[df.index[10], "high"] = 105.0
        signals = [Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=104.0, strategy_name="orb")]
        config = LabelingConfig(evaluation_horizon_bars=20)

        a = label_outcomes_bulk(signals, df, config, "EURUSD", "5M")
        b = label_outcomes_bulk(signals, df, config, "EURUSD", "5M")
        assert [o.to_dict() for o in a] == [o.to_dict() for o in b]

    def test_identical_dataset_produces_identical_estimate(self) -> None:
        dataset = OutcomeDataset.from_list(
            [_outcome(_ts(1), OutcomeLabel.WIN, 2.0), _outcome(_ts(2), OutcomeLabel.LOSS, -1.0)]
        )
        a = ProbabilityEstimator().estimate(dataset)
        b = ProbabilityEstimator().estimate(dataset)
        assert a.to_dict() == b.to_dict()
