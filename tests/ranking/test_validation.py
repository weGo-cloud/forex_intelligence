from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.ranking.validation import (
    OptimizationProvenance,
    benjamini_hochberg_adjusted,
    check_optimization_leakage,
    one_sided_binomial_p_value,
    robustness_gate,
)


def ts(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


class TestBinomialPValue:
    def test_at_null_rate_gives_moderate_p_value(self) -> None:
        p = one_sided_binomial_p_value(50, 100, null_p=0.5)
        assert 0.3 < p < 0.7

    def test_strong_outperformance_gives_small_p_value(self) -> None:
        p = one_sided_binomial_p_value(80, 100, null_p=0.5)
        assert p < 0.001

    def test_underperformance_gives_large_p_value(self) -> None:
        p = one_sided_binomial_p_value(20, 100, null_p=0.5)
        assert p > 0.99

    def test_all_wins_gives_minimal_p_value(self) -> None:
        p = one_sided_binomial_p_value(10, 10, null_p=0.5)
        assert p == pytest.approx((0.5) ** 10)

    def test_rejects_invalid_null_p(self) -> None:
        with pytest.raises(ValueError):
            one_sided_binomial_p_value(5, 10, null_p=1.0)

    def test_rejects_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            one_sided_binomial_p_value(1, 0)

    def test_rejects_successes_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            one_sided_binomial_p_value(11, 10)


class TestBenjaminiHochberg:
    def test_empty_list(self) -> None:
        assert benjamini_hochberg_adjusted([]) == []

    def test_single_value_unchanged(self) -> None:
        assert benjamini_hochberg_adjusted([0.03]) == [0.03]

    def test_adjusted_values_never_below_raw(self) -> None:
        raw = [0.001, 0.2, 0.03, 0.5, 0.04, 0.9]
        adjusted = benjamini_hochberg_adjusted(raw)
        assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw))

    def test_adjusted_values_bounded_by_one(self) -> None:
        adjusted = benjamini_hochberg_adjusted([0.01, 0.5, 0.99])
        assert all(0 <= a <= 1 for a in adjusted)

    def test_monotone_in_sorted_order(self) -> None:
        raw = [0.2, 0.001, 0.5, 0.03]
        adjusted = benjamini_hochberg_adjusted(raw)
        order = sorted(range(len(raw)), key=lambda i: raw[i])
        sorted_adjusted = [adjusted[i] for i in order]
        assert sorted_adjusted == sorted(sorted_adjusted)  # BH q-values are non-decreasing in p-rank

    def test_all_significant_when_all_tiny(self) -> None:
        adjusted = benjamini_hochberg_adjusted([0.0001, 0.0002, 0.0003])
        assert all(a < 0.05 for a in adjusted)

    def test_rejects_out_of_range_p_value(self) -> None:
        with pytest.raises(ValueError):
            benjamini_hochberg_adjusted([0.5, 1.5])


class TestOptimizationLeakage:
    def test_no_provenance_dates_not_flagged(self) -> None:
        leaked, reason = check_optimization_leakage(OptimizationProvenance(), ts(1), ts(10))
        assert leaked is False
        assert reason is None

    def test_overlapping_periods_flagged(self) -> None:
        prov = OptimizationProvenance(selection_period_start=ts(1), selection_period_end=ts(10))
        leaked, reason = check_optimization_leakage(prov, ts(5), ts(15))
        assert leaked is True
        assert "overlaps" in reason

    def test_non_overlapping_periods_not_flagged(self) -> None:
        prov = OptimizationProvenance(selection_period_start=ts(1), selection_period_end=ts(10))
        leaked, reason = check_optimization_leakage(prov, ts(10), ts(20))
        assert leaked is False
        assert reason is None

    def test_missing_evaluation_period_not_flagged(self) -> None:
        prov = OptimizationProvenance(selection_period_start=ts(1), selection_period_end=ts(10))
        leaked, reason = check_optimization_leakage(prov, None, None)
        assert leaked is False


class TestRobustnessGate:
    def test_none_summary_is_eligible(self) -> None:
        assert robustness_gate(None) == (True, None)

    def test_stable_is_eligible(self) -> None:
        eligible, reason = robustness_gate({"stability_label": "stable"})
        assert eligible is True

    def test_sensitive_is_eligible(self) -> None:
        eligible, reason = robustness_gate({"stability_label": "sensitive"})
        assert eligible is True

    def test_isolated_is_disqualified(self) -> None:
        eligible, reason = robustness_gate({"stability_label": "isolated"})
        assert eligible is False
        assert "isolated" in reason

    def test_insufficient_data_is_disqualified(self) -> None:
        eligible, reason = robustness_gate({"stability_label": "insufficient_data"})
        assert eligible is False
