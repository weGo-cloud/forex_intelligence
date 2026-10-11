"""Transparent, empirical probability estimation - the M5 mission's
explicit "statistical baseline first" requirement. No ML, no fitted
model: every number here is a direct count or a closed-form interval
over the outcomes actually in the (optionally filtered/time-cut)
dataset.

`ProbabilityEstimator.estimate()` never falls back to a different,
looser population when the requested filter matches few or zero
outcomes (M5 mission section 18: "If the requested conditioning
combination has no observations: probability = unavailable rather than
falling back silently to a different population") - a small or empty
sample produces `None` probabilities plus a visible `SampleCounts`,
not a number borrowed from a broader query the caller didn't ask for.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from app.probability.filters import OutcomeFilter
from app.probability.models import OutcomeLabel
from app.probability.outcomes import OutcomeDataset


@dataclass(frozen=True)
class SampleCounts:
    total_observations: int
    resolved_observations: int  # wins + losses + breakevens (excludes UNRESOLVED - see OutcomeLabel docstring)
    wins: int
    losses: int
    breakevens: int
    unresolved: int

    def __post_init__(self) -> None:
        if self.resolved_observations != self.wins + self.losses + self.breakevens:
            raise ValueError("resolved_observations must equal wins + losses + breakevens")
        if self.total_observations != self.resolved_observations + self.unresolved:
            raise ValueError("total_observations must equal resolved_observations + unresolved")


@dataclass(frozen=True)
class WilsonInterval:
    """Wilson score confidence interval for a binomial proportion -
    chosen specifically because, unlike the plain normal ("Wald")
    approximation, it stays well-behaved (bounded within [0, 1], sane
    coverage) at SMALL sample sizes, which is exactly where M5's
    conditional queries (M5 mission section 12) are most likely to
    land. `probability` is the raw point estimate (`successes / n`),
    included alongside the interval so a caller never has to
    reconstruct it separately.
    """

    probability: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    sample_size: int


def wilson_score_interval(successes: int, n: int, confidence_level: float = 0.95) -> WilsonInterval:
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0 <= successes <= n):
        raise ValueError("successes must be between 0 and n")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1")

    # statistics.NormalDist is Python stdlib (no scipy dependency needed).
    z = statistics.NormalDist().inv_cdf(1 - (1 - confidence_level) / 2)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom

    return WilsonInterval(
        probability=phat,
        lower_bound=max(0.0, center - margin),
        upper_bound=min(1.0, center + margin),
        confidence_level=confidence_level,
        sample_size=n,
    )


@dataclass(frozen=True)
class ProbabilityEstimate:
    counts: SampleCounts

    p_win: Optional[float]
    p_loss: Optional[float]
    p_breakeven: Optional[float]
    p_unresolved: Optional[float]

    win_interval: Optional[WilsonInterval]

    expected_r: Optional[float]
    avg_win_r: Optional[float]
    avg_loss_r: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {
                "total_observations": self.counts.total_observations,
                "resolved_observations": self.counts.resolved_observations,
                "wins": self.counts.wins,
                "losses": self.counts.losses,
                "breakevens": self.counts.breakevens,
                "unresolved": self.counts.unresolved,
            },
            "p_win": self.p_win,
            "p_loss": self.p_loss,
            "p_breakeven": self.p_breakeven,
            "p_unresolved": self.p_unresolved,
            "win_interval": (
                {
                    "probability": self.win_interval.probability,
                    "lower_bound": self.win_interval.lower_bound,
                    "upper_bound": self.win_interval.upper_bound,
                    "confidence_level": self.win_interval.confidence_level,
                    "sample_size": self.win_interval.sample_size,
                }
                if self.win_interval is not None
                else None
            ),
            "expected_r": self.expected_r,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
        }


class ProbabilityEstimator:
    def __init__(self, confidence_level: float = 0.95) -> None:
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be between 0 and 1")
        self.confidence_level = confidence_level

    def estimate(
        self,
        dataset: OutcomeDataset,
        filter: Optional[OutcomeFilter] = None,
        as_of: Optional[datetime] = None,
    ) -> ProbabilityEstimate:
        """Estimate empirical outcome probabilities over `dataset`,
        optionally restricted to a conditioning `filter` and/or a
        causal cutoff `as_of` (applied BEFORE filtering makes no
        difference to the result - both are pure predicates over the
        same population - but `as_of` is applied first here so a
        caller inspecting an intermediate `OutcomeDataset` for
        debugging sees the causally-valid population at each step).
        """
        eligible = dataset
        if as_of is not None:
            eligible = eligible.resolved_before(as_of)
        if filter is not None:
            eligible = filter.apply(eligible)

        wins = [o for o in eligible if o.outcome_label == OutcomeLabel.WIN]
        losses = [o for o in eligible if o.outcome_label == OutcomeLabel.LOSS]
        breakevens = [o for o in eligible if o.outcome_label == OutcomeLabel.BREAKEVEN]
        unresolved = [o for o in eligible if o.outcome_label == OutcomeLabel.UNRESOLVED]

        counts = SampleCounts(
            total_observations=len(eligible),
            resolved_observations=len(wins) + len(losses) + len(breakevens),
            wins=len(wins),
            losses=len(losses),
            breakevens=len(breakevens),
            unresolved=len(unresolved),
        )

        p_win = p_loss = p_breakeven = None
        win_interval = None
        expected_r = avg_win_r = avg_loss_r = None

        if counts.resolved_observations > 0:
            n = counts.resolved_observations
            p_win = counts.wins / n
            p_loss = counts.losses / n
            p_breakeven = counts.breakevens / n
            win_interval = wilson_score_interval(counts.wins, n, self.confidence_level)

            resolved_r = [o.r_multiple for o in wins + losses + breakevens if o.r_multiple is not None]
            if resolved_r:
                expected_r = statistics.mean(resolved_r)
            win_r = [o.r_multiple for o in wins if o.r_multiple is not None]
            if win_r:
                avg_win_r = statistics.mean(win_r)
            loss_r = [o.r_multiple for o in losses if o.r_multiple is not None]
            if loss_r:
                avg_loss_r = statistics.mean(loss_r)

        p_unresolved = (counts.unresolved / counts.total_observations) if counts.total_observations > 0 else None

        return ProbabilityEstimate(
            counts=counts,
            p_win=p_win,
            p_loss=p_loss,
            p_breakeven=p_breakeven,
            p_unresolved=p_unresolved,
            win_interval=win_interval,
            expected_r=expected_r,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
        )

    def compare_groups(
        self, dataset: OutcomeDataset, group_key, as_of: Optional[datetime] = None
    ) -> dict[Any, ProbabilityEstimate]:
        """Estimate probabilities independently for each group produced
        by `group_key` (e.g. `lambda o: o.regime_context.volatility_state
        if o.regime_context else None`) - a foundation for future
        calibration analysis (do outcomes actually differ across
        conditions in the direction/magnitude a model would predict?),
        without M5 itself producing any predicted probability to
        calibrate against (that's explicitly M6+'s job - see the M5
        mission's hard scope boundary on ML/decision engines). Each
        group is estimated independently via `estimate()`, so every
        group gets its own honest sample-size/uncertainty treatment -
        no pooling, no smoothing across groups.
        """
        eligible = dataset.resolved_before(as_of) if as_of is not None else dataset
        groups = eligible.group_by(group_key)
        return {key: self.estimate(group_dataset) for key, group_dataset in groups.items()}
