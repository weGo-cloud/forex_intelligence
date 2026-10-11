from __future__ import annotations

from app.evaluation.evaluator import StrategyEvaluator, summarize_robustness_result
from app.evaluation.models import EvidenceTier
from app.probability.filters import OutcomeFilter
from app.probability.models import OutcomeLabel
from app.probability.outcomes import OutcomeDataset
from app.regimes.models import TrendState
from tests.evaluation.conftest import make_context, make_outcome, ts


class TestZeroObservations:
    def test_empty_dataset(self) -> None:
        dataset = OutcomeDataset.from_list([])
        result = StrategyEvaluator().evaluate(dataset)
        assert result.probability.counts.total_observations == 0
        assert result.evidence_tier == EvidenceTier.INSUFFICIENT


class TestOnlyUnresolved:
    def test_all_unresolved_outcomes(self) -> None:
        dataset = OutcomeDataset.from_list(
            [make_outcome(ts(i), label=OutcomeLabel.UNRESOLVED, r=None) for i in range(15)]
        )
        result = StrategyEvaluator().evaluate(dataset)
        assert result.probability.p_win is None
        assert result.probability.counts.unresolved == 15
        assert result.evidence_tier == EvidenceTier.INSUFFICIENT  # 0 resolved observations


class TestVerySmallSamples:
    def test_single_observation(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(1))])
        result = StrategyEvaluator().evaluate(dataset)
        assert result.probability.counts.total_observations == 1
        assert result.evidence_tier == EvidenceTier.INSUFFICIENT


class TestSingleWalkForwardWindow:
    def test_period_fits_exactly_one_window(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(i)) for i in range(5)])
        result = StrategyEvaluator().evaluate_walk_forward(dataset, ts(0), ts(10), train_days=5, test_days=5)
        assert result.walk_forward.stability.number_of_windows == 1


class TestNoValidWalkForwardWindows:
    def test_period_too_short_for_any_window(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(0))])
        result = StrategyEvaluator().evaluate_walk_forward(dataset, ts(0), ts(1), train_days=30, test_days=30)
        assert result.walk_forward.stability.number_of_windows == 0
        assert result.probability is not None  # baseline still computed


class TestMissingRegimeContext:
    def test_regime_filter_excludes_outcomes_with_no_context(self) -> None:
        dataset = OutcomeDataset.from_list(
            [make_outcome(ts(i), regime_context=None) for i in range(10)]
        )
        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(trend_state=TrendState.UPTREND))
        assert result.probability.counts.total_observations == 0


class TestUnknownRegimeState:
    def test_unknown_state_is_a_queryable_value(self) -> None:
        ctx = make_context(trend=TrendState.UNKNOWN)
        dataset = OutcomeDataset.from_list([make_outcome(ts(i), regime_context=ctx) for i in range(12)])
        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(trend_state=TrendState.UNKNOWN))
        assert result.probability.counts.total_observations == 12


class TestMixedInstrumentsAndTimeframes:
    def test_unfiltered_evaluation_pools_everything(self) -> None:
        dataset = OutcomeDataset.from_list(
            [make_outcome(ts(i), instrument="EURUSD", timeframe="5M") for i in range(5)]
            + [make_outcome(ts(20 + i), instrument="GBPUSD", timeframe="1H") for i in range(5)]
        )
        result = StrategyEvaluator().evaluate(dataset)  # no filter - pools both, honestly labeled as "all"
        assert result.probability.counts.total_observations == 10
        assert result.instrument_symbol is None  # never claims a single instrument when none was requested

    def test_filtering_isolates_one_instrument(self) -> None:
        dataset = OutcomeDataset.from_list(
            [make_outcome(ts(i), instrument="EURUSD") for i in range(5)]
            + [make_outcome(ts(20 + i), instrument="GBPUSD") for i in range(5)]
        )
        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(instrument_symbol="EURUSD"))
        assert result.probability.counts.total_observations == 5


class TestDuplicateEvaluationConfigurations:
    def test_evaluate_combinations_allows_duplicate_filters(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(i), strategy="orb") for i in range(10)])
        f = OutcomeFilter(strategy_name="orb")
        results = StrategyEvaluator().evaluate_combinations(dataset, [f, f])
        assert len(results) == 2
        assert results[0].to_dict() == results[1].to_dict()


class TestEmptyFilterCombination:
    def test_empty_filter_matches_everything(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(i)) for i in range(10)])
        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter())
        assert result.probability.counts.total_observations == 10


class TestEvaluateCombinations:
    def test_evaluates_exactly_the_supplied_filters_no_auto_search(self) -> None:
        dataset = OutcomeDataset.from_list(
            [make_outcome(ts(i), regime_context=make_context(trend=TrendState.UPTREND)) for i in range(5)]
            + [make_outcome(ts(20 + i), regime_context=make_context(trend=TrendState.DOWNTREND)) for i in range(5)]
        )
        filters = [OutcomeFilter(trend_state=TrendState.UPTREND), OutcomeFilter(trend_state=TrendState.DOWNTREND)]
        results = StrategyEvaluator().evaluate_combinations(dataset, filters)
        assert len(results) == 2
        assert results[0].probability.counts.total_observations == 5
        assert results[1].probability.counts.total_observations == 5


class TestRobustnessSummary:
    def test_summarize_robustness_result_lifts_expected_fields(self) -> None:
        class _FakeRobustnessResult:
            parent_experiment_id = "exp_000042"
            parameter_name = "risk_reward"
            values_tested = [1.5, 2.0, 2.5]
            stability_label = "stable"
            notes = "all neighbors share the same return sign"

        summary = summarize_robustness_result(_FakeRobustnessResult())
        assert summary["parent_experiment_id"] == "exp_000042"
        assert summary["stability_label"] == "stable"
        assert summary["values_tested"] == [1.5, 2.0, 2.5]

    def test_robustness_summary_attaches_to_evaluation(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(i)) for i in range(10)])
        result = StrategyEvaluator().evaluate(dataset, robustness_summary={"stability_label": "stable"})
        assert result.robustness_summary == {"stability_label": "stable"}
        assert result.to_dict()["robustness_summary"] == {"stability_label": "stable"}
