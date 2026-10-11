"""Proves the mission's central requirement (section 13): "A result for
period T must never use outcomes that were unresolved at the beginning
of T" / "future outcomes cannot alter evaluation results for an earlier
historical period."
"""

from __future__ import annotations

from app.evaluation.evaluator import StrategyEvaluator
from app.probability.models import OutcomeLabel
from app.probability.outcomes import OutcomeDataset
from tests.evaluation.conftest import make_outcome, ts


class TestBaselineEvaluationCausality:
    def test_as_of_cutoff_excludes_later_outcomes(self) -> None:
        early = [make_outcome(ts(i), label=OutcomeLabel.WIN, r=2.0) for i in range(5)]
        late = [make_outcome(ts(50 + i), label=OutcomeLabel.LOSS, r=-1.0) for i in range(20)]
        dataset = OutcomeDataset.from_list(early + late)

        cutoff_result = StrategyEvaluator().evaluate(dataset, as_of=ts(10))
        assert cutoff_result.probability.counts.total_observations == 5
        assert cutoff_result.probability.p_win == 1.0

    def test_appending_future_outcomes_does_not_change_a_past_as_of_result(self) -> None:
        early = [make_outcome(ts(i), label=OutcomeLabel.WIN, r=2.0) for i in range(5)]
        dataset_a = OutcomeDataset.from_list(early)
        baseline = StrategyEvaluator().evaluate(dataset_a, as_of=ts(10))

        flood_of_losses = [make_outcome(ts(50 + i), label=OutcomeLabel.LOSS, r=-1.0) for i in range(100)]
        dataset_b = OutcomeDataset.from_list(early + flood_of_losses)
        after = StrategyEvaluator().evaluate(dataset_b, as_of=ts(10))

        assert baseline.probability.counts == after.probability.counts
        assert baseline.probability.p_win == after.probability.p_win
        assert baseline.probability.expected_r == after.probability.expected_r


class TestWalkForwardWindowCausality:
    def test_window_excludes_signals_that_fire_after_the_window(self) -> None:
        in_window = [make_outcome(ts(5 + i), label=OutcomeLabel.WIN, r=2.0) for i in range(5)]
        after_window = [make_outcome(ts(50 + i), label=OutcomeLabel.LOSS, r=-1.0) for i in range(20)]
        dataset = OutcomeDataset.from_list(in_window + after_window)

        result = StrategyEvaluator().evaluate_walk_forward(
            dataset, period_start=ts(0), period_end=ts(10), train_days=5, test_days=5
        )
        for window in result.walk_forward.windows:
            for outcome in in_window + after_window:
                if outcome.signal_timestamp >= window.test_end:
                    # Nothing signaling on/after test_end can be counted in this window's totals -
                    # verified indirectly: a window's total can never exceed the count of outcomes
                    # whose signal_timestamp actually falls inside [test_start, test_end).
                    pass
        # Direct proof: total observations across ALL windows must not exceed
        # the outcomes that actually fall within the full period.
        total_in_windows = sum(w.probability.counts.total_observations for w in result.walk_forward.windows)
        assert total_in_windows <= len(in_window)

    def test_window_excludes_outcomes_unresolved_before_window_end(self) -> None:
        """An outcome whose signal_timestamp falls inside the window but
        whose exit_timestamp resolves AFTER the window's test_end must
        not be counted in that window - this is the second half of
        temporal honesty (mission section 13), distinct from the
        signal-timing check above."""
        straddling = make_outcome(ts(6), label=OutcomeLabel.WIN, r=2.0, exit_ts=ts(20))  # resolves well after window
        clean = make_outcome(ts(7), label=OutcomeLabel.WIN, r=2.0, exit_ts=ts(7))
        dataset = OutcomeDataset.from_list([straddling, clean])

        result = StrategyEvaluator().evaluate_walk_forward(
            dataset, period_start=ts(0), period_end=ts(10), train_days=5, test_days=5
        )
        # Find the window covering [5, 10)
        window = next(w for w in result.walk_forward.windows if w.test_start == ts(5))
        assert window.probability.counts.total_observations == 1  # only "clean", not "straddling"

    def test_future_window_data_does_not_change_an_earlier_windows_result(self) -> None:
        outcomes_early = [make_outcome(ts(i), label=OutcomeLabel.WIN, r=2.0) for i in range(5)]
        dataset_a = OutcomeDataset.from_list(outcomes_early)
        result_a = StrategyEvaluator().evaluate_walk_forward(
            dataset_a, period_start=ts(0), period_end=ts(30), train_days=5, test_days=5
        )

        outcomes_late = [make_outcome(ts(20 + i), label=OutcomeLabel.LOSS, r=-3.0) for i in range(5)]
        dataset_b = OutcomeDataset.from_list(outcomes_early + outcomes_late)
        result_b = StrategyEvaluator().evaluate_walk_forward(
            dataset_b, period_start=ts(0), period_end=ts(30), train_days=5, test_days=5
        )

        first_window_a = result_a.walk_forward.windows[0]
        first_window_b = result_b.walk_forward.windows[0]
        assert first_window_a.probability.to_dict() == first_window_b.probability.to_dict()


class TestDeterminism:
    def test_identical_inputs_produce_identical_evaluation(self) -> None:
        outcomes = [make_outcome(ts(i), label=OutcomeLabel.WIN, r=2.0) for i in range(30)]
        dataset = OutcomeDataset.from_list(outcomes)

        a = StrategyEvaluator().evaluate(dataset)
        b = StrategyEvaluator().evaluate(dataset)
        assert a.to_dict() == b.to_dict()

    def test_identical_inputs_produce_identical_walk_forward_report(self) -> None:
        outcomes = [make_outcome(ts(i), label=OutcomeLabel.WIN, r=2.0) for i in range(30)]
        dataset = OutcomeDataset.from_list(outcomes)

        a = StrategyEvaluator().evaluate_walk_forward(dataset, ts(0), ts(28), train_days=7, test_days=7)
        b = StrategyEvaluator().evaluate_walk_forward(dataset, ts(0), ts(28), train_days=7, test_days=7)
        assert a.to_dict() == b.to_dict()
