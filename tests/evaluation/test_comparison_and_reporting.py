from __future__ import annotations

import json

from app.evaluation.comparison import MULTIPLE_TESTING_CAVEAT, compare_evaluations
from app.evaluation.evaluator import StrategyEvaluator
from app.evaluation.reporting import format_comparison_report, format_report
from app.probability.filters import OutcomeFilter
from app.probability.models import OutcomeLabel
from app.probability.outcomes import OutcomeDataset
from tests.evaluation.conftest import make_outcome, ts


def _dataset():
    return OutcomeDataset.from_list(
        [make_outcome(ts(i), strategy="orb", label=OutcomeLabel.WIN, r=2.0) for i in range(20)]
        + [make_outcome(ts(20 + i), strategy="rsi_divergence", label=OutcomeLabel.LOSS, r=-1.0) for i in range(20)]
    )


class TestCompareEvaluations:
    def test_rows_preserve_caller_order(self) -> None:
        dataset = _dataset()
        evaluator = StrategyEvaluator()
        a = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        b = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="rsi_divergence"))

        rows = compare_evaluations([b, a])
        assert rows[0].strategy_name == "rsi_divergence"
        assert rows[1].strategy_name == "orb"

    def test_does_not_rank_or_declare_a_winner(self) -> None:
        dataset = _dataset()
        evaluator = StrategyEvaluator()
        a = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        b = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="rsi_divergence"))
        rows = compare_evaluations([a, b])
        for row in rows:
            payload = row.to_dict()
            assert "rank" not in payload
            assert "score" not in payload
            assert "winner" not in payload

    def test_custom_labels(self) -> None:
        dataset = _dataset()
        evaluator = StrategyEvaluator()
        a = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        rows = compare_evaluations([a], labels=["my custom label"])
        assert rows[0].label == "my custom label"

    def test_mismatched_labels_length_raises(self) -> None:
        dataset = _dataset()
        a = StrategyEvaluator().evaluate(dataset)
        import pytest

        with pytest.raises(ValueError):
            compare_evaluations([a], labels=["one", "two"])


class TestReporting:
    def test_format_report_contains_key_sections(self) -> None:
        dataset = _dataset()
        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        report = format_report(result)
        assert "Strategy:" in report
        assert "Evidence:" in report
        assert "Status:" in report
        assert "BUY" not in report
        assert "SELL" not in report

    def test_format_comparison_report_includes_caveat_when_multiple(self) -> None:
        dataset = _dataset()
        evaluator = StrategyEvaluator()
        a = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        b = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="rsi_divergence"))
        report = format_comparison_report(compare_evaluations([a, b]))
        assert MULTIPLE_TESTING_CAVEAT in report

    def test_format_comparison_report_omits_caveat_for_single_row(self) -> None:
        dataset = _dataset()
        a = StrategyEvaluator().evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        report = format_comparison_report(compare_evaluations([a]))
        assert MULTIPLE_TESTING_CAVEAT not in report


class TestSerialization:
    def test_evaluation_to_dict_is_json_safe(self) -> None:
        dataset = _dataset()
        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        json.dumps(result.to_dict())

    def test_walk_forward_to_dict_preserves_window_and_stability_data(self) -> None:
        dataset = _dataset()
        result = StrategyEvaluator().evaluate_walk_forward(dataset, ts(0), ts(35), train_days=7, test_days=7)
        payload = result.to_dict()
        json.dumps(payload)
        assert "windows" in payload["walk_forward"]
        assert "stability" in payload["walk_forward"]
        assert payload["walk_forward"]["stability"]["number_of_windows"] == len(result.walk_forward.windows)

    def test_comparison_row_to_dict_is_json_safe(self) -> None:
        dataset = _dataset()
        a = StrategyEvaluator().evaluate(dataset, OutcomeFilter(strategy_name="orb"))
        rows = compare_evaluations([a])
        json.dumps(rows[0].to_dict())
