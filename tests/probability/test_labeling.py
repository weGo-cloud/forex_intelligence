from __future__ import annotations

import pandas as pd
import pytest

from app.probability.labeling import (
    InvalidSignalError,
    LabelingConfig,
    label_outcome,
    label_outcomes_bulk,
)
from app.probability.models import ExitReason, OutcomeLabel
from app.strategies.base import Signal


def _flat_df(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(price, index=idx)
    return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close}, index=idx)


class TestConfigValidation:
    def test_rejects_zero_horizon(self) -> None:
        with pytest.raises(ValueError):
            LabelingConfig(evaluation_horizon_bars=0)

    def test_rejects_non_positive_risk_reward(self) -> None:
        with pytest.raises(ValueError):
            LabelingConfig(default_risk_reward=0)

    def test_rejects_negative_epsilon(self) -> None:
        with pytest.raises(ValueError):
            LabelingConfig(breakeven_r_epsilon=-1.0)


class TestLongTakeProfit:
    def test_tp_hit_produces_win(self) -> None:
        df = _flat_df()
        df.loc[df.index[10], "high"] = 105.0
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=104.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=20), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.WIN
        assert out.exit_reason == ExitReason.TAKE_PROFIT
        assert out.exit_price == 104.0
        assert out.r_multiple == pytest.approx(2.0)
        assert out.entry_timestamp == df.index[6]


class TestLongStopLoss:
    def test_sl_hit_produces_loss(self) -> None:
        df = _flat_df()
        df.loc[df.index[8], "low"] = 90.0
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=95.0, take_profit=110.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=20), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.LOSS
        assert out.exit_reason == ExitReason.STOP_LOSS
        assert out.r_multiple == pytest.approx(-1.0)


class TestShortTakeProfit:
    def test_tp_hit_produces_win(self) -> None:
        df = _flat_df()
        df.loc[df.index[10], "low"] = 95.0
        sig = Signal(signal_time=df.index[5], direction="short", stop_loss=102.0, take_profit=96.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=20), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.WIN
        assert out.exit_reason == ExitReason.TAKE_PROFIT
        assert out.r_multiple > 0


class TestShortStopLoss:
    def test_sl_hit_produces_loss(self) -> None:
        df = _flat_df()
        df.loc[df.index[8], "high"] = 110.0
        sig = Signal(signal_time=df.index[5], direction="short", stop_loss=105.0, take_profit=90.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=20), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.LOSS
        assert out.exit_reason == ExitReason.STOP_LOSS
        assert out.r_multiple < 0


class TestTimeExpiry:
    def test_neither_barrier_hit_resolves_at_horizon_close(self) -> None:
        df = _flat_df(n=40)
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=90.0, take_profit=200.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=10), "EURUSD", "5M")
        assert out.exit_reason == ExitReason.TIME_EXPIRY
        assert out.exit_price == 100.0

    def test_time_expiry_positive_r_is_win(self) -> None:
        df = _flat_df(n=40)
        close = df["close"].copy()
        close.iloc[16:] = 103.0  # drifts up by horizon end, never touching either barrier
        df["close"] = close
        df["open"] = close
        df["high"] = close + 0.3
        df["low"] = close - 0.3
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=90.0, take_profit=200.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=10), "EURUSD", "5M")
        assert out.exit_reason == ExitReason.TIME_EXPIRY
        assert out.outcome_label == OutcomeLabel.WIN

    def test_time_expiry_negative_r_is_loss(self) -> None:
        df = _flat_df(n=40)
        close = df["close"].copy()
        close.iloc[16:] = 97.0
        df["close"] = close
        df["open"] = close
        df["high"] = close + 0.3
        df["low"] = close - 0.3
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=90.0, take_profit=200.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=10), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.LOSS


class TestBreakeven:
    def test_flat_close_at_horizon_is_breakeven(self) -> None:
        df = _flat_df(n=40)
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=90.0, take_profit=200.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=10), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.BREAKEVEN
        assert out.r_multiple == pytest.approx(0.0)


class TestSameBarCollision:
    def _collision_df(self) -> pd.DataFrame:
        df = _flat_df(n=40)
        df.loc[df.index[9], "high"] = 105.0
        df.loc[df.index[9], "low"] = 90.0
        return df

    def test_unresolved_policy_produces_unresolved(self) -> None:
        df = self._collision_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=95.0, take_profit=104.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=20, same_bar_policy="unresolved"), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.UNRESOLVED
        assert out.exit_reason == ExitReason.SAME_BAR_UNRESOLVED
        assert out.exit_price is None
        assert out.r_multiple is None
        assert out.exit_timestamp == df.index[9]  # still knows WHEN, even though not WHAT

    def test_stop_first_policy_produces_loss(self) -> None:
        df = self._collision_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=95.0, take_profit=104.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=20, same_bar_policy="stop_first"), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.LOSS
        assert out.exit_reason == ExitReason.STOP_LOSS

    def test_target_first_policy_produces_win(self) -> None:
        df = self._collision_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=95.0, take_profit=104.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=20, same_bar_policy="target_first"), "EURUSD", "5M")
        assert out.outcome_label == OutcomeLabel.WIN
        assert out.exit_reason == ExitReason.TAKE_PROFIT


class TestMissingAndInsufficientData:
    def test_signal_time_not_in_index_raises(self) -> None:
        df = _flat_df()
        bad_time = pd.Timestamp("2099-01-01", tz="UTC")
        sig = Signal(signal_time=bad_time, direction="long", stop_loss=95.0, take_profit=105.0, strategy_name="t")
        with pytest.raises(KeyError):
            label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")

    def test_signal_on_last_bar_has_no_entry(self) -> None:
        df = _flat_df(n=20)
        sig = Signal(signal_time=df.index[-1], direction="long", stop_loss=95.0, take_profit=105.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")
        assert out.entry_timestamp is None
        assert out.entry_price is None
        assert out.outcome_label == OutcomeLabel.UNRESOLVED
        assert out.exit_reason == ExitReason.INSUFFICIENT_DATA

    def test_insufficient_horizon_before_resolution(self) -> None:
        df = _flat_df(n=15)  # not enough bars for a 50-bar horizon
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=90.0, take_profit=200.0, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(evaluation_horizon_bars=50), "EURUSD", "5M")
        assert out.exit_reason == ExitReason.INSUFFICIENT_DATA
        assert out.outcome_label == OutcomeLabel.UNRESOLVED
        assert out.exit_price is None


class TestInvalidBarrierConfiguration:
    def test_stop_loss_equal_to_entry_raises(self) -> None:
        df = _flat_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=100.0, take_profit=110.0, strategy_name="t")
        with pytest.raises(InvalidSignalError):
            label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")

    def test_take_profit_on_wrong_side_raises(self) -> None:
        df = _flat_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=99.0, strategy_name="t")
        with pytest.raises(InvalidSignalError):
            label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")

    def test_stop_loss_on_wrong_side_raises(self) -> None:
        df = _flat_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=110.0, take_profit=120.0, strategy_name="t")
        with pytest.raises(InvalidSignalError):
            label_outcome(sig, df, LabelingConfig(), "EURUSD", "5M")


class TestDefaultRiskReward:
    def test_missing_take_profit_uses_config_risk_reward(self) -> None:
        df = _flat_df()
        sig = Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=None, strategy_name="t")
        out = label_outcome(sig, df, LabelingConfig(default_risk_reward=3.0), "EURUSD", "5M")
        entry = out.entry_price
        risk = entry - 98.0
        assert out.take_profit == pytest.approx(entry + 3.0 * risk)


class TestBulkLabeling:
    def test_skips_invalid_by_default(self) -> None:
        df = _flat_df()
        signals = [
            Signal(signal_time=df.index[5], direction="long", stop_loss=100.0, take_profit=110.0, strategy_name="t"),  # degenerate
            Signal(signal_time=df.index[6], direction="long", stop_loss=98.0, take_profit=110.0, strategy_name="t"),  # valid
        ]
        outcomes = label_outcomes_bulk(signals, df, LabelingConfig(), "EURUSD", "5M")
        assert len(outcomes) == 1

    def test_propagates_when_skip_invalid_false(self) -> None:
        df = _flat_df()
        signals = [Signal(signal_time=df.index[5], direction="long", stop_loss=100.0, take_profit=110.0, strategy_name="t")]
        with pytest.raises(InvalidSignalError):
            label_outcomes_bulk(signals, df, LabelingConfig(), "EURUSD", "5M", skip_invalid=False)

    def test_supports_overlapping_signals(self) -> None:
        df = _flat_df()
        signals = [
            Signal(signal_time=df.index[5], direction="long", stop_loss=98.0, take_profit=110.0, strategy_name="t"),
            Signal(signal_time=df.index[6], direction="short", stop_loss=102.0, take_profit=90.0, strategy_name="t"),
        ]
        outcomes = label_outcomes_bulk(signals, df, LabelingConfig(), "EURUSD", "5M")
        assert len(outcomes) == 2  # both evaluated independently, no exclusivity
