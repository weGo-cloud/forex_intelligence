from __future__ import annotations

from app.evaluation.models import WalkForwardWindowEvaluation
from app.evaluation.stability import compute_stability_metrics
from app.probability.estimators import ProbabilityEstimator
from app.probability.outcomes import OutcomeDataset
from tests.evaluation.conftest import make_outcome, ts
from app.probability.models import OutcomeLabel


def _window(index: int, outcomes: list) -> WalkForwardWindowEvaluation:
    dataset = OutcomeDataset.from_list(outcomes)
    estimate = ProbabilityEstimator().estimate(dataset)
    return WalkForwardWindowEvaluation(
        window_index=index, train_start=ts(0), train_end=ts(5), test_start=ts(5), test_end=ts(10), probability=estimate
    )


class TestConsistentPositiveWindows:
    def test_all_positive_windows(self) -> None:
        windows = [
            _window(i, [make_outcome(ts(i), label=OutcomeLabel.WIN, r=2.0) for _ in range(10)])
            for i in range(3)
        ]
        stability = compute_stability_metrics(windows)
        assert stability.number_of_windows == 3
        assert stability.windows_with_positive_expected_r == 3
        assert stability.mean_expected_r == 2.0


class TestMixedWindows:
    def test_some_positive_some_negative(self) -> None:
        windows = [
            _window(0, [make_outcome(ts(0), label=OutcomeLabel.WIN, r=2.0) for _ in range(10)]),
            _window(1, [make_outcome(ts(1), label=OutcomeLabel.LOSS, r=-1.0) for _ in range(10)]),
        ]
        stability = compute_stability_metrics(windows)
        assert stability.windows_with_positive_expected_r == 1
        assert stability.min_p_win == 0.0
        assert stability.max_p_win == 1.0
        assert stability.win_rate_range == 1.0


class TestNegativeWindows:
    def test_all_negative(self) -> None:
        windows = [
            _window(i, [make_outcome(ts(i), label=OutcomeLabel.LOSS, r=-1.5) for _ in range(10)])
            for i in range(2)
        ]
        stability = compute_stability_metrics(windows)
        assert stability.windows_with_positive_expected_r == 0
        assert stability.mean_expected_r == -1.5


class TestSingleWindow:
    def test_single_window_std_is_zero_not_none(self) -> None:
        windows = [_window(0, [make_outcome(ts(0), label=OutcomeLabel.WIN, r=1.0) for _ in range(10)])]
        stability = compute_stability_metrics(windows)
        assert stability.number_of_windows == 1
        assert stability.std_expected_r == 0.0


class TestEmptyWindowList:
    def test_no_windows_gives_none_aggregates(self) -> None:
        stability = compute_stability_metrics([])
        assert stability.number_of_windows == 0
        assert stability.mean_expected_r is None
        assert stability.min_p_win is None


class TestWindowsWithNoResolvedObservations:
    def test_zero_observation_windows_excluded_from_aggregates(self) -> None:
        windows = [
            _window(0, [make_outcome(ts(0), label=OutcomeLabel.WIN, r=1.0) for _ in range(10)]),
            _window(1, []),  # empty window
        ]
        stability = compute_stability_metrics(windows)
        assert stability.number_of_windows == 2  # window count includes it
        assert stability.mean_expected_r == 1.0  # but the aggregate excludes the None
