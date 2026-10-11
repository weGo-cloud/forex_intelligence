from __future__ import annotations

from app.evaluation.evaluator import StrategyEvaluator
from app.probability.filters import OutcomeFilter
from app.probability.models import OutcomeLabel
from app.probability.outcomes import OutcomeDataset
from tests.evaluation.conftest import make_outcome, ts


class TestMultipleWindows:
    def test_produces_expected_window_count(self) -> None:
        outcomes = [make_outcome(ts(i)) for i in range(30)]
        dataset = OutcomeDataset.from_list(outcomes)

        result = StrategyEvaluator().evaluate_walk_forward(
            dataset, period_start=ts(0), period_end=ts(28), train_days=7, test_days=7
        )
        assert result.walk_forward is not None
        assert result.walk_forward.stability.number_of_windows >= 1

    def test_windows_are_temporally_ordered(self) -> None:
        outcomes = [make_outcome(ts(i)) for i in range(40)]
        dataset = OutcomeDataset.from_list(outcomes)

        result = StrategyEvaluator().evaluate_walk_forward(
            dataset, period_start=ts(0), period_end=ts(35), train_days=5, test_days=5
        )
        windows = result.walk_forward.windows
        for a, b in zip(windows, windows[1:]):
            assert a.test_end <= b.test_start
            assert a.window_index < b.window_index


class TestEmptyWindows:
    def test_window_with_no_matching_outcomes_has_zero_observations(self) -> None:
        outcomes = [make_outcome(ts(i), strategy="orb") for i in range(5)]  # all in the first few days
        dataset = OutcomeDataset.from_list(outcomes)

        result = StrategyEvaluator().evaluate_walk_forward(
            dataset, period_start=ts(0), period_end=ts(60), train_days=10, test_days=10,
            filter=OutcomeFilter(strategy_name="orb"),
        )
        # Later windows should have zero observations since all outcomes are early.
        later_windows = [w for w in result.walk_forward.windows if w.test_start > ts(10)]
        assert any(w.probability.counts.total_observations == 0 for w in later_windows)


class TestInsufficientWindows:
    def test_no_windows_fit_does_not_raise(self) -> None:
        outcomes = [make_outcome(ts(i)) for i in range(3)]
        dataset = OutcomeDataset.from_list(outcomes)

        result = StrategyEvaluator().evaluate_walk_forward(
            dataset, period_start=ts(0), period_end=ts(2), train_days=100, test_days=100
        )
        assert result.walk_forward.stability.number_of_windows == 0
        assert any("No walk-forward windows fit" in w for w in result.warnings)


class TestBaselineStillComputed:
    def test_overall_evaluation_present_alongside_windows(self) -> None:
        outcomes = [make_outcome(ts(i)) for i in range(30)]
        dataset = OutcomeDataset.from_list(outcomes)
        result = StrategyEvaluator().evaluate_walk_forward(
            dataset, period_start=ts(0), period_end=ts(28), train_days=7, test_days=7
        )
        assert result.probability.counts.total_observations > 0
        assert result.evaluation_period_start == ts(0)
        assert result.evaluation_period_end == ts(28)
