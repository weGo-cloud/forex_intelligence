from __future__ import annotations

from app.evaluation.models import EvidenceTier, StabilityMetrics, StrategyContextEvaluation, WalkForwardEvaluation
from app.probability.estimators import ProbabilityEstimate, SampleCounts, WilsonInterval
from app.probability.filters import OutcomeFilter


def make_evaluation(
    wins: int,
    n: int,
    tier: EvidenceTier = EvidenceTier.ADEQUATE,
    strategy: str = "orb",
    instrument: str = "EURUSD",
    interval_width: float = 0.3,
    expected_r: float = 0.3,
    walk_forward=None,
    evaluation_period_start=None,
    evaluation_period_end=None,
) -> StrategyContextEvaluation:
    p_win = wins / n if n > 0 else None
    counts = SampleCounts(total_observations=n, resolved_observations=n, wins=wins, losses=n - wins, breakevens=0, unresolved=0)
    interval = None
    if n > 0:
        half = interval_width / 2
        interval = WilsonInterval(
            probability=p_win, lower_bound=max(0.0, p_win - half), upper_bound=min(1.0, p_win + half),
            confidence_level=0.95, sample_size=n,
        )
    estimate = ProbabilityEstimate(
        counts=counts, p_win=p_win, p_loss=(1 - p_win) if p_win is not None else None, p_breakeven=0.0 if n > 0 else None,
        p_unresolved=0.0 if n > 0 else None, win_interval=interval, expected_r=expected_r if n > 0 else None,
        avg_win_r=1.5 if n > 0 else None, avg_loss_r=-1.0 if n > 0 else None,
    )
    return StrategyContextEvaluation(
        context_filter=OutcomeFilter(strategy_name=strategy, instrument_symbol=instrument),
        probability=estimate,
        evaluation_period_start=evaluation_period_start,
        evaluation_period_end=evaluation_period_end,
        as_of=None,
        evidence_tier=tier,
        warnings=(),
        walk_forward=walk_forward,
    )


def make_walk_forward(positive_windows: int, total_windows: int) -> WalkForwardEvaluation:
    stability = StabilityMetrics(
        number_of_windows=total_windows, windows_with_positive_expected_r=positive_windows,
        mean_expected_r=0.2, median_expected_r=0.2, std_expected_r=0.1,
        mean_p_win=0.55, min_p_win=0.4, max_p_win=0.7, win_rate_range=0.3, expected_r_range=0.4,
    )
    return WalkForwardEvaluation(windows=(), stability=stability)
