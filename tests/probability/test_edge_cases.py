from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from app.probability.estimators import ProbabilityEstimator
from app.probability.labeling import InvalidSignalError, LabelingConfig, label_outcome, label_outcomes_bulk
from app.probability.models import ExitReason, OutcomeLabel
from app.probability.outcomes import OutcomeDataset
from app.regimes.models import StructureState, TrendState, VolatilityState
from app.strategies.base import Signal


def _flat_df(n: int = 60, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(price, index=idx)
    return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close}, index=idx)


class TestNoSignals:
    def test_empty_signal_list_produces_empty_dataset(self) -> None:
        df = _flat_df()
        outcomes = label_outcomes_bulk([], df, LabelingConfig(), "EURUSD", "5M")
        assert outcomes == []


class TestNoResolvedOutcomes:
    def test_estimator_handles_dataset_with_only_unresolved(self) -> None:
        df = _flat_df()
        df.loc[df.index[9], "high"] = 105.0
        df.loc[df.index[9], "low"] = 90.0  # same-bar collision
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=95.0, take_profit=104.0, strategy_name="t")
        outcome = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=20, same_bar_policy="unresolved"), "EURUSD", "5M")
        result = ProbabilityEstimator().estimate(OutcomeDataset.from_list([outcome]))
        assert result.p_win is None
        assert result.counts.unresolved == 1


class TestZeroAndIdenticalPrices:
    def test_zero_r_multiple_time_expiry_is_breakeven(self) -> None:
        df = _flat_df(n=40)
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=90.0, take_profit=200.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=10), "EURUSD", "5M")
        assert out.r_multiple == pytest.approx(0.0)
        assert out.outcome_label == OutcomeLabel.BREAKEVEN


class TestInvalidRiskDistance:
    def test_take_profit_equal_to_stop_loss_raises(self) -> None:
        df = _flat_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=98.0, strategy_name="t")
        with pytest.raises(InvalidSignalError):
            label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")

    def test_negative_risk_distance_short_raises(self) -> None:
        df = _flat_df()
        # short: stop_loss must be ABOVE entry - putting it below is invalid
        sig = Signal(signal_time=df.index[5], direction="short", stop_loss=90.0, take_profit=80.0, strategy_name="t")
        with pytest.raises(InvalidSignalError):
            label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")


class TestMissingAndUnknownRegimeContext:
    def test_missing_regime_context_is_handled_by_filter(self) -> None:
        from app.probability.filters import OutcomeFilter

        df = _flat_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=110.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M", regime_context=None)
        dataset = OutcomeDataset.from_list([out])
        result = ProbabilityEstimator().estimate(dataset, filter=OutcomeFilter(trend_state=TrendState.UPTREND))
        assert result.counts.total_observations == 0  # cannot match a regime filter with no context

    def test_unknown_regime_state_is_still_a_valid_filter_value(self) -> None:
        from app.probability.filters import OutcomeFilter
        from app.regimes.context import RegimeContext

        df = _flat_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=110.0, strategy_name="t")
        ctx = RegimeContext(
            timestamp=df.index[6], instrument_symbol="EURUSD", timeframe="5M",
            trend_state=TrendState.UNKNOWN, volatility_state=VolatilityState.UNKNOWN,
            structure_state=StructureState.UNKNOWN, sessions=(),
        )
        out = label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M", regime_context=ctx)
        dataset = OutcomeDataset.from_list([out])
        result = ProbabilityEstimator().estimate(dataset, filter=OutcomeFilter(trend_state=TrendState.UNKNOWN))
        assert result.counts.total_observations == 1


class TestOverlappingTrades:
    def test_overlapping_signals_labeled_independently(self) -> None:
        df = _flat_df(n=60)
        signals = [
            Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=110.0, strategy_name="t"),
            Signal(signal_time=df.index[5], direction="short", stop_loss=102.0, take_profit=90.0, strategy_name="t"),
        ]
        outcomes = label_outcomes_bulk(signals, df, LabelingConfig(), "EURUSD", "5M")
        assert len(outcomes) == 2
        assert {o.direction for o in outcomes} == {"long", "short"}


class TestDuplicateTimestamps:
    def test_duplicate_signal_time_in_index_raises_rather_than_guessing(self) -> None:
        idx = list(pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC"))
        idx[10] = idx[9]  # duplicate
        idx = pd.DatetimeIndex(idx)
        close = pd.Series(100.0, index=idx)
        df = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close}, index=idx)
        sig = Signal(signal_time=idx[9], direction="long", stop_loss=98.0, take_profit=110.0, strategy_name="t")
        with pytest.raises((ValueError, KeyError)):
            label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")


class TestTimezoneAwareTimestamps:
    def test_tz_aware_signal_and_index_work(self) -> None:
        df = _flat_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=110.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")
        assert out.entry_timestamp.tzinfo is not None
        assert out.exit_timestamp.tzinfo is not None


class TestInsufficientFutureCandles:
    def test_signal_near_end_of_data_is_unresolved(self) -> None:
        df = _flat_df(n=20)
        sig = Signal(signal_time=df.index[15], direction="long", stop_loss=90.0, take_profit=200.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=50), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.UNRESOLVED
        assert out.exit_reason == ExitReason.INSUFFICIENT_DATA
