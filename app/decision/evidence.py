"""The one function that constructs `M6Evidence` - every value is read
directly off an existing `StrategyContextEvaluation` (M6), never
recomputed. If a caller needs the full M6 object (e.g. for its own
report), `StrategyContextEvaluation` is still available separately;
this function only narrows it for `app.decision`'s use.
"""

from __future__ import annotations

from app.decision.models import M6Evidence
from app.evaluation.models import StrategyContextEvaluation


def evidence_from_evaluation(evaluation: StrategyContextEvaluation) -> M6Evidence:
    probability = evaluation.probability
    win_interval = None
    if probability.win_interval is not None:
        win_interval = {
            "lower_bound": probability.win_interval.lower_bound,
            "upper_bound": probability.win_interval.upper_bound,
            "confidence_level": probability.win_interval.confidence_level,
        }

    stability = None
    window_count = None
    if evaluation.walk_forward is not None:
        stability = evaluation.walk_forward.stability.to_dict()
        window_count = evaluation.walk_forward.stability.number_of_windows

    f = evaluation.context_filter
    regime_filter = {
        "trend_state": f.trend_state.value if f.trend_state is not None else None,
        "volatility_state": f.volatility_state.value if f.volatility_state is not None else None,
        "structure_state": f.structure_state.value if f.structure_state is not None else None,
        "session": f.session,
    }

    return M6Evidence(
        strategy_name=evaluation.strategy_name,
        instrument_symbol=evaluation.instrument_symbol,
        timeframe=evaluation.timeframe,
        direction=evaluation.direction,
        total_observations=probability.counts.total_observations,
        resolved_observations=probability.counts.resolved_observations,
        evidence_tier=evaluation.evidence_tier,
        p_win=probability.p_win,
        win_interval=win_interval,
        historical_expectancy=probability.expected_r,
        stability=stability,
        walk_forward_window_count=window_count,
        regime_filter=regime_filter,
        warnings=evaluation.warnings,
    )
