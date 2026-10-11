"""M7 temporal-causality audit (mission section 18).

Three layers are checked, each with the same truncation pattern used
by the M2/M4 bias tests: compute on a prefix of the data, compute on a
longer series sharing that prefix, and require everything knowable
within the prefix to be identical.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.decision.engine import DecisionEngine
from app.decision.evidence import evidence_from_evaluation
from app.evaluation.models import EvidenceTier, StrategyContextEvaluation
from app.observations.detectors import (
    fvg_observations,
    structure_observations,
    support_resistance_observations,
    swing_observations,
)
from app.observations.models import Observation, ObservationSource, ObservationType, observations_as_of
from app.probability.estimators import ProbabilityEstimate, SampleCounts, WilsonInterval
from app.probability.filters import OutcomeFilter
from app.setups.detection import SetupDetector
from app.setups.models import SetupStatus


def _frame(n: int, seed: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.08, n)) + np.sin(np.arange(n) / 15) * 4, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + rng.uniform(0.05, 0.15, n), "low": close - rng.uniform(0.05, 0.15, n), "close": close},
        index=idx,
    )


def _known_by(observations: list[Observation], cutoff) -> list[dict]:
    return [o.to_dict() for o in observations if o.timestamp <= cutoff]


class TestObservationDetectorsAreCausal:
    """Every detector's observations knowable by the truncation point
    must be identical whether or not later candles exist."""

    def _check(self, build) -> None:
        full = _frame(500)
        truncated = full.iloc[:350]
        cutoff = truncated.index[-1]

        full_obs = build(full)
        trunc_obs = build(truncated)
        assert _known_by(full_obs, cutoff) == _known_by(trunc_obs, cutoff)
        assert all(o.timestamp <= cutoff for o in trunc_obs)

    def test_swings(self) -> None:
        self._check(lambda df: swing_observations(df["high"], df["low"], "EURUSD", "5M"))

    def test_fvg(self) -> None:
        self._check(lambda df: fvg_observations(df["high"], df["low"], "EURUSD", "5M"))

    def test_structure_bos_choch(self) -> None:
        self._check(lambda df: structure_observations(df["high"], df["low"], "EURUSD", "5M"))

    def test_support_resistance(self) -> None:
        self._check(
            lambda df: support_resistance_observations(
                df["high"], df["low"], df["close"], "EURUSD", "5M", tolerance_pct=0.02, min_touches=2
            )
        )

    def test_appended_shock_does_not_change_earlier_observations(self) -> None:
        base = _frame(400)
        extra_idx = pd.date_range(base.index[-1] + pd.Timedelta(minutes=5), periods=100, freq="5min", tz="UTC")
        shock = pd.Series(base["close"].iloc[-1] + np.linspace(0, -60, 100), index=extra_idx)
        extended = pd.concat(
            [base, pd.DataFrame({"open": shock, "high": shock + 3, "low": shock - 3, "close": shock}, index=extra_idx)]
        )
        cutoff = base.index[-1]
        a = structure_observations(base["high"], base["low"], "EURUSD", "5M")
        b = structure_observations(extended["high"], extended["low"], "EURUSD", "5M")
        assert _known_by(a, cutoff) == _known_by(b, cutoff)


def _obs(kind: ObservationType, ts: datetime, direction="long", **kw) -> Observation:
    return Observation(
        observation_type=kind, instrument_symbol="EURUSD", timeframe="5M", timestamp=ts,
        source=ObservationSource.MARKET_DATA, direction=direction, price=100.0, **kw,
    )


T0 = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
T1 = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
T2 = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)


class TestSetupStateIsAsOfSafe:
    def test_setup_is_detected_as_of_before_its_confirmation_exists(self) -> None:
        sweep = _obs(ObservationType.LIQUIDITY_SWEEP, T0, metadata={"swept_level": 99.0})
        bos = _obs(ObservationType.BOS, T2)  # confirmation only becomes knowable at T2
        all_obs = [sweep, bos]

        as_of_t1 = observations_as_of(all_obs, T1)
        early = SetupDetector().detect("EURUSD", "5M", sweep_observations=[o for o in as_of_t1 if o.observation_type == ObservationType.LIQUIDITY_SWEEP],
                                       structure_observations=[o for o in as_of_t1 if o.observation_type == ObservationType.BOS])
        assert early[0].status == SetupStatus.DETECTED

        as_of_t2 = observations_as_of(all_obs, T2)
        later = SetupDetector().detect("EURUSD", "5M", sweep_observations=[o for o in as_of_t2 if o.observation_type == ObservationType.LIQUIDITY_SWEEP],
                                       structure_observations=[o for o in as_of_t2 if o.observation_type == ObservationType.BOS])
        assert later[0].status == SetupStatus.VALID

    def test_future_observations_cannot_change_an_earlier_as_of_result(self) -> None:
        sweep = _obs(ObservationType.LIQUIDITY_SWEEP, T0, metadata={"swept_level": 99.0})
        baseline_obs = observations_as_of([sweep], T1)
        with_future = observations_as_of([sweep, _obs(ObservationType.BOS, T2)], T1)
        assert [o.to_dict() for o in baseline_obs] == [o.to_dict() for o in with_future]

    def test_observations_as_of_is_inclusive_at_the_boundary(self) -> None:
        obs = _obs(ObservationType.BOS, T1)
        assert observations_as_of([obs], T1) == [obs]
        assert observations_as_of([obs], T0) == []


class TestDecisionAsOfSafe:
    def _evidence(self):
        counts = SampleCounts(total_observations=40, resolved_observations=40, wins=24, losses=16, breakevens=0, unresolved=0)
        interval = WilsonInterval(probability=0.6, lower_bound=0.45, upper_bound=0.74, confidence_level=0.95, sample_size=40)
        est = ProbabilityEstimate(counts=counts, p_win=0.6, p_loss=0.4, p_breakeven=0.0, p_unresolved=0.0,
                                  win_interval=interval, expected_r=0.3, avg_win_r=1.5, avg_loss_r=-1.0)
        ev = StrategyContextEvaluation(context_filter=OutcomeFilter(strategy_name="x"), probability=est,
                                       evaluation_period_start=None, evaluation_period_end=None, as_of=T1,
                                       evidence_tier=EvidenceTier.ADEQUATE, warnings=())
        return evidence_from_evaluation(ev)

    def test_decision_as_of_t_uses_only_setups_knowable_at_t(self) -> None:
        sweep = _obs(ObservationType.LIQUIDITY_SWEEP, T0, metadata={"swept_level": 99.0})
        bos = _obs(ObservationType.BOS, T2)
        engine = DecisionEngine()

        early_setup = SetupDetector().detect("EURUSD", "5M", sweep_observations=[sweep], structure_observations=[])[0]
        late_setup = SetupDetector().detect("EURUSD", "5M", sweep_observations=[sweep], structure_observations=[bos])[0]

        early = engine.decide(early_setup, self._evidence(), at=T1)
        late = engine.decide(late_setup, self._evidence(), at=T2)
        assert early.state.value == "wait"          # decision at T1 could not know about the T2 BOS
        assert late.state.value == "valid_setup"    # decision at T2 can

    def test_identical_inputs_identical_decision(self) -> None:
        sweep = _obs(ObservationType.LIQUIDITY_SWEEP, T0, metadata={"swept_level": 99.0})
        setup = SetupDetector().detect("EURUSD", "5M", sweep_observations=[sweep])[0]
        a = DecisionEngine().decide(setup, self._evidence(), at=T1)
        b = DecisionEngine().decide(setup, self._evidence(), at=T1)
        assert a.to_dict() == b.to_dict()
