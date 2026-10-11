from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.evaluation.models import EvidenceTier
from app.ranking.ranking import rank_evaluations
from app.ranking.validation import OptimizationProvenance
from tests.ranking.conftest import make_evaluation, make_walk_forward


def ts(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


class TestBasicRanking:
    def test_higher_fias_ranks_first(self) -> None:
        strong = make_evaluation(wins=28, n=30, strategy="strong")
        weak = make_evaluation(wins=16, n=30, strategy="weak")
        results = rank_evaluations([weak, strong])
        assert results[0].label != results[1].label
        assert results[0].fias.score >= results[1].fias.score
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_ranks_are_sequential_starting_at_one(self) -> None:
        evals = [make_evaluation(wins=20 + i, n=30, strategy=f"s{i}") for i in range(4)]
        results = rank_evaluations(evals)
        ranks = sorted(r.rank for r in results if r.rank is not None)
        assert ranks == [1, 2, 3, 4]

    def test_empty_batch(self) -> None:
        assert rank_evaluations([]) == []

    def test_single_evaluation(self) -> None:
        results = rank_evaluations([make_evaluation(wins=20, n=30)])
        assert results[0].rank == 1


class TestExclusion:
    def test_insufficient_evidence_excluded_but_visible(self) -> None:
        good = make_evaluation(wins=25, n=30, strategy="good")
        bad = make_evaluation(wins=2, n=3, tier=EvidenceTier.INSUFFICIENT, strategy="bad")
        results = rank_evaluations([good, bad])
        excluded = [r for r in results if r.label.startswith("bad")][0]
        assert excluded.rank is None
        assert excluded.eligible is False
        assert excluded.exclusion_reasons
        assert excluded.evaluation is not None  # never deleted, just excluded

    def test_robustness_disqualification_excludes(self) -> None:
        strategies = [make_evaluation(wins=25, n=30, strategy="a"), make_evaluation(wins=25, n=30, strategy="b")]
        results = rank_evaluations(strategies, robustness_summaries=[{"stability_label": "stable"}, {"stability_label": "isolated"}])
        excluded = [r for r in results if "isolated" in " ".join(r.exclusion_reasons)]
        assert len(excluded) == 1
        assert excluded[0].rank is None

    def test_optimization_leakage_excludes(self) -> None:
        evaluation = make_evaluation(wins=25, n=30, evaluation_period_start=ts(5), evaluation_period_end=ts(15))
        provenance = OptimizationProvenance(selection_period_start=ts(1), selection_period_end=ts(10))
        results = rank_evaluations([evaluation], provenances=[provenance])
        assert results[0].rank is None
        assert any("overlaps" in r for r in results[0].exclusion_reasons)

    def test_non_overlapping_provenance_does_not_exclude(self) -> None:
        evaluation = make_evaluation(wins=25, n=30, evaluation_period_start=ts(10), evaluation_period_end=ts(20))
        provenance = OptimizationProvenance(selection_period_start=ts(1), selection_period_end=ts(10))
        results = rank_evaluations([evaluation], provenances=[provenance])
        assert results[0].rank == 1

    def test_excluded_entries_sorted_after_ranked_entries(self) -> None:
        good = make_evaluation(wins=25, n=30, strategy="good")
        bad = make_evaluation(wins=2, n=3, tier=EvidenceTier.INSUFFICIENT, strategy="bad")
        results = rank_evaluations([bad, good])  # bad first in input
        assert results[0].rank == 1
        assert results[0].label.startswith("good")
        assert results[-1].rank is None


class TestSignificance:
    def test_strong_performer_is_significant(self) -> None:
        evaluation = make_evaluation(wins=28, n=30)
        results = rank_evaluations([evaluation])
        assert results[0].p_value is not None
        assert results[0].q_value is not None
        assert results[0].significant is True

    def test_at_null_performer_not_significant(self) -> None:
        evaluation = make_evaluation(wins=15, n=30)
        results = rank_evaluations([evaluation])
        assert results[0].significant is False

    def test_zero_resolved_observations_gives_no_p_value(self) -> None:
        evaluation = make_evaluation(wins=0, n=0)
        results = rank_evaluations([evaluation])
        assert results[0].p_value is None
        assert results[0].q_value is None
        assert results[0].significant is None


class TestValidation:
    def test_mismatched_labels_length_raises(self) -> None:
        with pytest.raises(ValueError):
            rank_evaluations([make_evaluation(wins=20, n=30)], labels=["a", "b"])

    def test_mismatched_robustness_length_raises(self) -> None:
        with pytest.raises(ValueError):
            rank_evaluations([make_evaluation(wins=20, n=30)], robustness_summaries=[None, None])

    def test_mismatched_provenance_length_raises(self) -> None:
        with pytest.raises(ValueError):
            rank_evaluations([make_evaluation(wins=20, n=30)], provenances=[None, None])


class TestDeterminism:
    def test_identical_batch_identical_ranking(self) -> None:
        evals = [make_evaluation(wins=20 + i, n=30, strategy=f"s{i}") for i in range(3)]
        a = rank_evaluations(evals)
        b = rank_evaluations(evals)
        assert [r.to_dict() for r in a] == [r.to_dict() for r in b]

    def test_tie_breaking_is_deterministic_by_label(self) -> None:
        # Identical evaluations (identical FIAS) -> tie broken by label, always the same way.
        e1 = make_evaluation(wins=20, n=30, strategy="zzz")
        e2 = make_evaluation(wins=20, n=30, strategy="aaa")
        results = rank_evaluations([e1, e2])
        assert results[0].label < results[1].label


class TestSerialization:
    def test_to_dict_is_json_safe(self) -> None:
        import json

        evaluation = make_evaluation(wins=25, n=30)
        results = rank_evaluations([evaluation], robustness_summaries=[{"stability_label": "stable"}])
        json.dumps(results[0].to_dict())
