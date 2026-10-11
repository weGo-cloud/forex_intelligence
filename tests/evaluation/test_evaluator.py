from __future__ import annotations

import pytest

from app.evaluation.evaluator import StrategyEvaluator
from app.evaluation.models import EvaluationConfig, EvidenceTier
from app.probability.filters import OutcomeFilter
from app.probability.models import OutcomeLabel
from app.probability.outcomes import OutcomeDataset
from tests.evaluation.conftest import make_outcome, ts


class TestEvaluationConfigValidation:
    def test_default_is_valid(self) -> None:
        EvaluationConfig()

    def test_rejects_minimum_below_limited(self) -> None:
        with pytest.raises(ValueError):
            EvaluationConfig(minimum_resolved_observations=5, limited_resolved_observations=10)

    def test_rejects_invalid_confidence_level(self) -> None:
        with pytest.raises(ValueError):
            EvaluationConfig(confidence_level=1.5)

    def test_rejects_negative_limited(self) -> None:
        with pytest.raises(ValueError):
            EvaluationConfig(limited_resolved_observations=-1)


class TestBasicEvaluation:
    def test_all_outcomes_no_filter(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(i)) for i in range(1, 41)])
        result = StrategyEvaluator().evaluate(dataset)
        assert result.probability.counts.total_observations == 40
        assert result.evidence_tier == EvidenceTier.ADEQUATE

    def test_filtered_outcomes(self) -> None:
        dataset = OutcomeDataset.from_list(
            [make_outcome(ts(i), strategy="orb") for i in range(1, 21)]
            + [make_outcome(ts(i), strategy="rsi_divergence") for i in range(21, 41)]
        )
        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        assert result.probability.counts.total_observations == 20
        assert result.strategy_name == "orb"

    def test_no_matching_outcomes(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(1), strategy="orb")])
        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(strategy_name="nonexistent"))
        assert result.probability.counts.total_observations == 0
        assert result.evidence_tier == EvidenceTier.INSUFFICIENT
        assert any("No observations" in w for w in result.warnings)

    def test_insufficient_sample(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(i)) for i in range(1, 4)])  # 3 observations
        result = StrategyEvaluator(EvaluationConfig(minimum_resolved_observations=30, limited_resolved_observations=10)).evaluate(dataset)
        assert result.evidence_tier == EvidenceTier.INSUFFICIENT
        assert any("LIMITED threshold" in w for w in result.warnings)
        # Raw statistics are still exposed, never hidden or deleted.
        assert result.probability.p_win is not None

    def test_limited_sample(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(i)) for i in range(1, 16)])  # 15 observations
        result = StrategyEvaluator(EvaluationConfig(minimum_resolved_observations=30, limited_resolved_observations=10)).evaluate(dataset)
        assert result.evidence_tier == EvidenceTier.LIMITED

    def test_adequate_sample(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(i)) for i in range(1, 31)])
        result = StrategyEvaluator(EvaluationConfig(minimum_resolved_observations=30, limited_resolved_observations=10)).evaluate(dataset)
        assert result.evidence_tier == EvidenceTier.ADEQUATE
        assert not any("threshold" in w for w in result.warnings)

    def test_unresolved_outcomes_generate_a_warning(self) -> None:
        dataset = OutcomeDataset.from_list(
            [make_outcome(ts(i), label=OutcomeLabel.WIN) for i in range(1, 31)]
            + [make_outcome(ts(40 + i), label=OutcomeLabel.UNRESOLVED, r=None) for i in range(3)]
        )
        result = StrategyEvaluator().evaluate(dataset)
        assert any("UNRESOLVED" in w for w in result.warnings)


class TestPropertiesReflectFilter:
    def test_properties_are_read_from_context_filter(self) -> None:
        dataset = OutcomeDataset.from_list([make_outcome(ts(1))])
        f = OutcomeFilter(strategy_name="orb", instrument_symbol="EURUSD", timeframe="5M", direction="long")
        result = StrategyEvaluator().evaluate(dataset, f)
        assert result.strategy_name == "orb"
        assert result.instrument_symbol == "EURUSD"
        assert result.timeframe == "5M"
        assert result.direction == "long"
