"""M6 domain models.

`StrategyContextEvaluation` embeds M5's `ProbabilityEstimate` and
`OutcomeFilter` WHOLE rather than flattening/re-copying their fields
(`p_win`, `counts`, `win_interval`, etc.) onto itself - per the mission's
explicit instruction ("do not copy every field blindly if existing M5
objects can be embedded or reused"). M6 adds exactly what M5 doesn't
have: a minimum-sample verdict (`EvidenceTier`), an optional temporal/
walk-forward breakdown, and human-facing warnings - interpretation, not
new statistics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.probability.estimators import ProbabilityEstimate
from app.probability.filters import OutcomeFilter


class EvidenceTier(str, enum.Enum):
    """Research metadata about sample size - NOT a quality/confidence
    score (mission section 9: "These are research metadata, not quality
    scores. Do not convert tiers into numerical ratings."). Thresholds
    live on `EvaluationConfig`, are configurable, and are documented
    guesses in the same spirit as M4's regime thresholds - not tuned
    against this project's historical data.
    """

    INSUFFICIENT = "insufficient"
    LIMITED = "limited"
    ADEQUATE = "adequate"


@dataclass(frozen=True)
class EvaluationConfig:
    # At/above this many RESOLVED observations, evidence is ADEQUATE.
    minimum_resolved_observations: int = 30
    # Between this and `minimum_resolved_observations`, evidence is
    # LIMITED; below this, INSUFFICIENT. Mission's own example uses 30
    # as "a" minimum sample threshold - this project treats that as the
    # ADEQUATE bar and adds a lower LIMITED bar so a 12-observation
    # result reads differently from a 0-observation one, without
    # pretending either is adequate.
    limited_resolved_observations: int = 10
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.limited_resolved_observations < 0:
            raise ValueError("limited_resolved_observations must be >= 0")
        if self.minimum_resolved_observations < self.limited_resolved_observations:
            raise ValueError("minimum_resolved_observations must be >= limited_resolved_observations")
        if not (0 < self.confidence_level < 1):
            raise ValueError("confidence_level must be between 0 and 1")


@dataclass(frozen=True)
class WalkForwardWindowEvaluation:
    """One walk-forward window's evaluation - the window boundaries
    come directly from `app.backtesting.validation.WalkForwardWindow`
    (M3, reused as-is), and `probability` is a full M5
    `ProbabilityEstimate` computed ONLY from outcomes whose
    `signal_timestamp` falls inside `[test_start, test_end)` and whose
    `exit_timestamp` is `< test_end` (both required for temporal
    honesty - see `app.evaluation.evaluator`'s module docstring)."""

    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    probability: ProbabilityEstimate


@dataclass(frozen=True)
class StabilityMetrics:
    """Descriptive-only (mission section 11: "do not create a single
    opaque robustness score yet") - every field here is a plain
    aggregate over the per-window `expected_r`/`p_win` values, nothing
    hidden or derived through an undocumented formula."""

    number_of_windows: int
    windows_with_positive_expected_r: int
    mean_expected_r: Optional[float]
    median_expected_r: Optional[float]
    std_expected_r: Optional[float]
    mean_p_win: Optional[float]
    min_p_win: Optional[float]
    max_p_win: Optional[float]
    win_rate_range: Optional[float]
    expected_r_range: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "number_of_windows": self.number_of_windows,
            "windows_with_positive_expected_r": self.windows_with_positive_expected_r,
            "mean_expected_r": self.mean_expected_r,
            "median_expected_r": self.median_expected_r,
            "std_expected_r": self.std_expected_r,
            "mean_p_win": self.mean_p_win,
            "min_p_win": self.min_p_win,
            "max_p_win": self.max_p_win,
            "win_rate_range": self.win_rate_range,
            "expected_r_range": self.expected_r_range,
        }


@dataclass(frozen=True)
class WalkForwardEvaluation:
    windows: tuple[WalkForwardWindowEvaluation, ...]
    stability: StabilityMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows": [
                {
                    "window_index": w.window_index,
                    "train_start": w.train_start.isoformat(),
                    "train_end": w.train_end.isoformat(),
                    "test_start": w.test_start.isoformat(),
                    "test_end": w.test_end.isoformat(),
                    "probability": w.probability.to_dict(),
                }
                for w in self.windows
            ],
            "stability": self.stability.to_dict(),
        }


@dataclass(frozen=True)
class StrategyContextEvaluation:
    """One evaluation: a filter (the "strategy x context" being asked
    about), the M5 probability evidence for it, an explicit sample-size
    verdict, and - optionally - a temporal/walk-forward breakdown and a
    pointer to an M3 robustness result. `strategy_name`/
    `instrument_symbol`/`timeframe`/`direction` are NOT independently
    settable - they are read straight off `context_filter` (the single
    source of truth the evaluator was actually given), so they can
    never drift out of sync with it (mission section 6: "the evaluator
    must work from configuration/filter objects").
    """

    context_filter: OutcomeFilter
    probability: ProbabilityEstimate

    evaluation_period_start: Optional[datetime]
    evaluation_period_end: Optional[datetime]
    as_of: Optional[datetime]

    evidence_tier: EvidenceTier
    warnings: tuple[str, ...] = field(default_factory=tuple)

    walk_forward: Optional[WalkForwardEvaluation] = None

    # A pointer to an M3 RobustnessResult (app.research.robustness),
    # NOT a recomputation of it - "the original M3 mechanisms remain
    # the source of truth" (mission section 12). None of M3's
    # parameter-neighborhood math is duplicated here.
    robustness_summary: Optional[dict[str, Any]] = None

    @property
    def strategy_name(self) -> Optional[str]:
        return self.context_filter.strategy_name

    @property
    def instrument_symbol(self) -> Optional[str]:
        return self.context_filter.instrument_symbol

    @property
    def timeframe(self) -> Optional[str]:
        return self.context_filter.timeframe

    @property
    def direction(self) -> Optional[str]:
        return self.context_filter.direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "instrument_symbol": self.instrument_symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "context_filter": {
                "trend_state": self.context_filter.trend_state.value if self.context_filter.trend_state else None,
                "volatility_state": self.context_filter.volatility_state.value if self.context_filter.volatility_state else None,
                "structure_state": self.context_filter.structure_state.value if self.context_filter.structure_state else None,
                "session": self.context_filter.session,
            },
            "probability": self.probability.to_dict(),
            "evaluation_period_start": self.evaluation_period_start.isoformat() if self.evaluation_period_start else None,
            "evaluation_period_end": self.evaluation_period_end.isoformat() if self.evaluation_period_end else None,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "evidence_tier": self.evidence_tier.value,
            "warnings": list(self.warnings),
            "walk_forward": self.walk_forward.to_dict() if self.walk_forward is not None else None,
            "robustness_summary": self.robustness_summary,
        }
