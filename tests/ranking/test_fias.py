from __future__ import annotations

import pytest

from app.evaluation.models import EvidenceTier
from app.ranking.fias import FIASConfig, compute_fias
from tests.ranking.conftest import make_evaluation, make_walk_forward


class TestFIASConfigValidation:
    def test_default_is_valid(self) -> None:
        FIASConfig()

    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ValueError):
            FIASConfig(weight_sample_adequacy=-1)

    def test_rejects_all_zero_weights(self) -> None:
        with pytest.raises(ValueError):
            FIASConfig(weight_sample_adequacy=0, weight_win_rate_confidence=0, weight_stability=0, weight_robustness=0)


class TestWithholding:
    def test_insufficient_tier_withheld(self) -> None:
        evaluation = make_evaluation(wins=3, n=5, tier=EvidenceTier.INSUFFICIENT)
        fias = compute_fias(evaluation)
        assert fias.score is None
        assert fias.withheld_reason is not None
        assert fias.components.to_dict() == {"sample_adequacy": None, "win_rate_confidence": None, "stability": None, "robustness": None}

    def test_limited_tier_not_withheld_by_default(self) -> None:
        evaluation = make_evaluation(wins=8, n=15, tier=EvidenceTier.LIMITED)
        fias = compute_fias(evaluation)
        assert fias.score is not None

    def test_custom_minimum_tier_withholds_limited(self) -> None:
        evaluation = make_evaluation(wins=8, n=15, tier=EvidenceTier.LIMITED)
        fias = compute_fias(evaluation, FIASConfig(minimum_tier_for_scoring=EvidenceTier.ADEQUATE))
        assert fias.score is None


class TestComponents:
    def test_sample_adequacy_matches_tier(self) -> None:
        for tier, expected in [(EvidenceTier.LIMITED, 0.5), (EvidenceTier.ADEQUATE, 1.0)]:
            evaluation = make_evaluation(wins=20, n=30, tier=tier)
            fias = compute_fias(evaluation)
            assert fias.components.sample_adequacy == expected

    def test_narrower_interval_scores_higher_confidence(self) -> None:
        narrow = make_evaluation(wins=20, n=30, interval_width=0.1)
        wide = make_evaluation(wins=20, n=30, interval_width=0.5)
        assert compute_fias(narrow).components.win_rate_confidence > compute_fias(wide).components.win_rate_confidence

    def test_win_rate_confidence_bounded_zero_to_one(self) -> None:
        evaluation = make_evaluation(wins=20, n=30, interval_width=1.0)
        score = compute_fias(evaluation).components.win_rate_confidence
        assert 0.0 <= score <= 1.0

    def test_stability_is_fraction_of_positive_windows(self) -> None:
        wf = make_walk_forward(positive_windows=3, total_windows=4)
        evaluation = make_evaluation(wins=20, n=30, walk_forward=wf)
        assert compute_fias(evaluation).components.stability == pytest.approx(0.75)

    def test_no_walk_forward_means_stability_excluded_not_zero(self) -> None:
        evaluation = make_evaluation(wins=20, n=30, walk_forward=None)
        fias = compute_fias(evaluation)
        assert fias.components.stability is None
        assert fias.score is not None  # still scored from the other components

    def test_robustness_maps_stable_and_sensitive(self) -> None:
        evaluation = make_evaluation(wins=20, n=30)
        assert compute_fias(evaluation, robustness_summary={"stability_label": "stable"}).components.robustness == 1.0
        assert compute_fias(evaluation, robustness_summary={"stability_label": "sensitive"}).components.robustness == 0.5

    def test_no_robustness_summary_excluded_not_zero(self) -> None:
        evaluation = make_evaluation(wins=20, n=30)
        fias = compute_fias(evaluation, robustness_summary=None)
        assert fias.components.robustness is None


class TestCompositeScore:
    def test_score_is_within_zero_one(self) -> None:
        evaluation = make_evaluation(wins=25, n=30, walk_forward=make_walk_forward(3, 4))
        fias = compute_fias(evaluation, robustness_summary={"stability_label": "stable"})
        assert 0.0 <= fias.score <= 1.0

    def test_weights_shift_the_composite(self) -> None:
        evaluation = make_evaluation(wins=25, n=30, walk_forward=make_walk_forward(1, 4))  # weak stability
        default_score = compute_fias(evaluation, robustness_summary={"stability_label": "stable"}).score
        heavy_stability = FIASConfig(weight_stability=10.0)
        weighted_score = compute_fias(evaluation, heavy_stability, robustness_summary={"stability_label": "stable"}).score
        assert weighted_score < default_score  # weak stability now dominates, pulling score down

    def test_zero_weight_excludes_a_component_entirely(self) -> None:
        evaluation = make_evaluation(wins=25, n=30, walk_forward=make_walk_forward(0, 4))  # stability = 0.0
        config = FIASConfig(weight_stability=0.0)
        fias = compute_fias(evaluation, config, robustness_summary={"stability_label": "stable"})
        # stability component is still reported for transparency even though it has zero weight
        assert fias.components.stability == 0.0


class TestDeterminism:
    def test_identical_inputs_identical_score(self) -> None:
        evaluation = make_evaluation(wins=25, n=30)
        a = compute_fias(evaluation)
        b = compute_fias(evaluation)
        assert a.to_dict() == b.to_dict()
