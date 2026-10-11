"""Descriptive stability metrics over a set of walk-forward window
evaluations. Every value is a plain aggregate (mean/median/stdev/min/
max) over the per-window `ProbabilityEstimate.expected_r`/`p_win`
values already computed by M5 - nothing new is estimated here, this
module only summarizes. Windows whose `expected_r`/`p_win` is `None`
(zero resolved observations in that window) are excluded from the
corresponding aggregate, not treated as 0 - the same "never invent a
value" discipline as M4/M5.
"""

from __future__ import annotations

import statistics
from typing import Optional

from app.evaluation.models import StabilityMetrics, WalkForwardWindowEvaluation


def compute_stability_metrics(windows: list[WalkForwardWindowEvaluation]) -> StabilityMetrics:
    expected_rs = [w.probability.expected_r for w in windows if w.probability.expected_r is not None]
    p_wins = [w.probability.p_win for w in windows if w.probability.p_win is not None]

    positive_windows = sum(1 for r in expected_rs if r > 0)

    mean_er: Optional[float] = statistics.mean(expected_rs) if expected_rs else None
    median_er: Optional[float] = statistics.median(expected_rs) if expected_rs else None
    std_er: Optional[float] = statistics.pstdev(expected_rs) if len(expected_rs) > 1 else (0.0 if expected_rs else None)

    mean_pw: Optional[float] = statistics.mean(p_wins) if p_wins else None
    min_pw: Optional[float] = min(p_wins) if p_wins else None
    max_pw: Optional[float] = max(p_wins) if p_wins else None
    win_rate_range: Optional[float] = (max_pw - min_pw) if p_wins else None
    expected_r_range: Optional[float] = (max(expected_rs) - min(expected_rs)) if expected_rs else None

    return StabilityMetrics(
        number_of_windows=len(windows),
        windows_with_positive_expected_r=positive_windows,
        mean_expected_r=mean_er,
        median_expected_r=median_er,
        std_expected_r=std_er,
        mean_p_win=mean_pw,
        min_p_win=min_pw,
        max_p_win=max_pw,
        win_rate_range=win_rate_range,
        expected_r_range=expected_r_range,
    )
