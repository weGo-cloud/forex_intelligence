from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.probability.estimators import ProbabilityEstimator, SampleCounts, wilson_score_interval
from app.probability.filters import OutcomeFilter
from app.probability.models import ExitReason, OutcomeLabel, TradeOutcome
from app.probability.outcomes import OutcomeDataset

TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _outcome(label: OutcomeLabel, r: float | None = 1.0, strategy: str = "orb", exit_ts=TS) -> TradeOutcome:
    reason = {
        OutcomeLabel.WIN: ExitReason.TAKE_PROFIT,
        OutcomeLabel.LOSS: ExitReason.STOP_LOSS,
        OutcomeLabel.BREAKEVEN: ExitReason.TIME_EXPIRY,
        OutcomeLabel.UNRESOLVED: ExitReason.SAME_BAR_UNRESOLVED,
    }[label]
    return TradeOutcome(
        strategy_name=strategy, instrument_symbol="EURUSD", timeframe="5M", direction="long",
        signal_timestamp=TS, entry_timestamp=TS, exit_timestamp=exit_ts,
        entry_price=100.0, exit_price=(100.0 + r if r is not None else None),
        stop_loss=98.0, take_profit=104.0,
        r_multiple=r, outcome_label=label, exit_reason=reason,
        evaluation_horizon_bars=20, bars_to_exit=5,
    )


class TestSampleCountsValidation:
    def test_rejects_inconsistent_resolved_total(self) -> None:
        with pytest.raises(ValueError):
            SampleCounts(total_observations=10, resolved_observations=5, wins=1, losses=1, breakevens=1, unresolved=5)

    def test_valid_counts_construct(self) -> None:
        SampleCounts(total_observations=10, resolved_observations=8, wins=5, losses=2, breakevens=1, unresolved=2)


class TestWilsonInterval:
    def test_bounds_within_zero_one(self) -> None:
        interval = wilson_score_interval(50, 100)
        assert 0 <= interval.lower_bound <= interval.upper_bound <= 1

    def test_point_estimate_matches_raw_proportion(self) -> None:
        interval = wilson_score_interval(30, 100)
        assert interval.probability == pytest.approx(0.3)

    def test_small_sample_stays_bounded(self) -> None:
        interval = wilson_score_interval(1, 3)
        assert 0 <= interval.lower_bound <= interval.upper_bound <= 1

    def test_all_successes_upper_bound_below_one_for_finite_n(self) -> None:
        interval = wilson_score_interval(10, 10)
        assert interval.upper_bound <= 1.0
        assert interval.upper_bound > interval.lower_bound  # still an interval, not a point mass at 1.0

    def test_rejects_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_interval(1, 0)

    def test_rejects_successes_greater_than_n(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_interval(5, 3)


class TestEstimateSimpleProbability:
    def test_all_wins(self) -> None:
        dataset = OutcomeDataset.from_list([_outcome(OutcomeLabel.WIN) for _ in range(10)])
        result = ProbabilityEstimator().estimate(dataset)
        assert result.p_win == 1.0
        assert result.p_loss == 0.0
        assert result.counts.wins == 10

    def test_all_losses(self) -> None:
        dataset = OutcomeDataset.from_list([_outcome(OutcomeLabel.LOSS, r=-1.0) for _ in range(10)])
        result = ProbabilityEstimator().estimate(dataset)
        assert result.p_win == 0.0
        assert result.p_loss == 1.0

    def test_mixed_outcomes(self) -> None:
        dataset = OutcomeDataset.from_list(
            [_outcome(OutcomeLabel.WIN, r=2.0)] * 6 + [_outcome(OutcomeLabel.LOSS, r=-1.0)] * 4
        )
        result = ProbabilityEstimator().estimate(dataset)
        assert result.p_win == pytest.approx(0.6)
        assert result.p_loss == pytest.approx(0.4)
        assert result.expected_r == pytest.approx((6 * 2.0 + 4 * -1.0) / 10)
        assert result.avg_win_r == pytest.approx(2.0)
        assert result.avg_loss_r == pytest.approx(-1.0)

    def test_zero_observations(self) -> None:
        dataset = OutcomeDataset.from_list([])
        result = ProbabilityEstimator().estimate(dataset)
        assert result.p_win is None
        assert result.p_loss is None
        assert result.win_interval is None
        assert result.expected_r is None
        assert result.counts.total_observations == 0

    def test_all_unresolved_gives_no_probabilities_but_visible_counts(self) -> None:
        dataset = OutcomeDataset.from_list([_outcome(OutcomeLabel.UNRESOLVED, r=None) for _ in range(5)])
        result = ProbabilityEstimator().estimate(dataset)
        assert result.p_win is None  # no resolved observations - not silently 0.0
        assert result.p_unresolved == 1.0
        assert result.counts.unresolved == 5
        assert result.counts.resolved_observations == 0

    def test_breakeven_counted_separately(self) -> None:
        dataset = OutcomeDataset.from_list(
            [_outcome(OutcomeLabel.WIN, r=1.0), _outcome(OutcomeLabel.BREAKEVEN, r=0.0)]
        )
        result = ProbabilityEstimator().estimate(dataset)
        assert result.p_breakeven == pytest.approx(0.5)
        assert result.counts.breakevens == 1


class TestConditionalFiltering:
    def test_filter_narrows_the_population(self) -> None:
        dataset = OutcomeDataset.from_list(
            [_outcome(OutcomeLabel.WIN, strategy="orb")] * 5 + [_outcome(OutcomeLabel.LOSS, strategy="rsi_divergence")] * 5
        )
        result = ProbabilityEstimator().estimate(dataset, filter=OutcomeFilter(strategy_name="orb"))
        assert result.counts.total_observations == 5
        assert result.p_win == 1.0

    def test_no_matches_gives_unavailable_not_a_fallback(self) -> None:
        dataset = OutcomeDataset.from_list([_outcome(OutcomeLabel.WIN, strategy="orb")])
        result = ProbabilityEstimator().estimate(dataset, filter=OutcomeFilter(strategy_name="nonexistent"))
        assert result.counts.total_observations == 0
        assert result.p_win is None


class TestCompareGroups:
    def test_groups_estimated_independently(self) -> None:
        dataset = OutcomeDataset.from_list(
            [_outcome(OutcomeLabel.WIN, strategy="orb")] * 3 + [_outcome(OutcomeLabel.LOSS, strategy="rsi_divergence")] * 3
        )
        groups = ProbabilityEstimator().compare_groups(dataset, lambda o: o.strategy_name)
        assert groups["orb"].p_win == 1.0
        assert groups["rsi_divergence"].p_win == 0.0


class TestSerialization:
    def test_to_dict_is_json_safe(self) -> None:
        import json

        dataset = OutcomeDataset.from_list([_outcome(OutcomeLabel.WIN)] * 5)
        result = ProbabilityEstimator().estimate(dataset)
        json.dumps(result.to_dict())
