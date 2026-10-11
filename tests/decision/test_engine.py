from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.decision.engine import DecisionConfig, DecisionEngine
from app.decision.evidence import evidence_from_evaluation
from app.decision.models import DecisionState, M6Evidence
from app.evaluation.models import EvidenceTier, StrategyContextEvaluation
from app.observations.models import Observation, ObservationSource, ObservationType
from app.probability.estimators import ProbabilityEstimate, SampleCounts, WilsonInterval
from app.probability.filters import OutcomeFilter
from app.setups.models import InvalidationCondition, Setup, SetupStatus, SetupType
from app.setups.state_machine import transition

TS = datetime(2024, 1, 1, tzinfo=timezone.utc)
TS2 = datetime(2024, 1, 2, tzinfo=timezone.utc)


def _setup(status: SetupStatus, **overrides) -> Setup:
    kwargs = dict(
        id="test:1", instrument_symbol="EURUSD", timeframe="5M", direction="long",
        setup_type=SetupType.LIQUIDITY_SWEEP_REVERSAL, created_at=TS, updated_at=TS,
        observations=(
            Observation(
                observation_type=ObservationType.LIQUIDITY_SWEEP, instrument_symbol="EURUSD", timeframe="5M",
                timestamp=TS, source=ObservationSource.MARKET_DATA, direction="long",
            ),
        ),
        levels=(), confirmations=(),
        invalidation_conditions=(InvalidationCondition(description="closes below 99", price=99.0, direction="long"),),
        status=status,
    )
    kwargs.update(overrides)
    return Setup(**kwargs)


def _evaluation(tier: EvidenceTier, resolved=40, p_win=0.6, expected_r=0.3, walk_forward=None) -> StrategyContextEvaluation:
    counts = SampleCounts(
        total_observations=resolved, resolved_observations=resolved,
        wins=int(resolved * p_win), losses=resolved - int(resolved * p_win), breakevens=0, unresolved=0,
    )
    interval = WilsonInterval(probability=p_win, lower_bound=max(0, p_win - 0.15), upper_bound=min(1, p_win + 0.15), confidence_level=0.95, sample_size=resolved)
    estimate = ProbabilityEstimate(
        counts=counts, p_win=p_win, p_loss=1 - p_win, p_breakeven=0.0, p_unresolved=0.0,
        win_interval=interval, expected_r=expected_r, avg_win_r=1.5, avg_loss_r=-1.0,
    )
    return StrategyContextEvaluation(
        context_filter=OutcomeFilter(strategy_name="liquidity_sweep_fvg", instrument_symbol="EURUSD"),
        probability=estimate, evaluation_period_start=None, evaluation_period_end=None, as_of=None,
        evidence_tier=tier, warnings=(), walk_forward=walk_forward,
    )


def _evidence(tier: EvidenceTier, **kwargs) -> M6Evidence:
    return evidence_from_evaluation(_evaluation(tier, **kwargs))


class TestSufficientEvidence:
    def test_valid_setup_with_adequate_evidence(self) -> None:
        setup = _setup(SetupStatus.VALID)
        evidence = _evidence(EvidenceTier.ADEQUATE)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.VALID_SETUP

    def test_entry_condition_met_with_adequate_evidence(self) -> None:
        setup = _setup(SetupStatus.ENTRY_CONDITION_MET)
        evidence = _evidence(EvidenceTier.ADEQUATE)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.ENTRY_CONDITION_MET


class TestInsufficientEvidence:
    def test_none_evidence(self) -> None:
        setup = _setup(SetupStatus.VALID)
        decision = DecisionEngine().decide(setup, None, at=TS2)
        assert decision.state == DecisionState.INSUFFICIENT_EVIDENCE

    def test_insufficient_tier(self) -> None:
        setup = _setup(SetupStatus.VALID)
        evidence = _evidence(EvidenceTier.INSUFFICIENT, resolved=2)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.INSUFFICIENT_EVIDENCE

    def test_limited_tier_still_decides_by_default(self) -> None:
        setup = _setup(SetupStatus.VALID)
        evidence = _evidence(EvidenceTier.LIMITED, resolved=15)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.VALID_SETUP
        assert any("LIMITED" in w for w in decision.warnings)

    def test_limited_tier_insufficient_under_strict_config(self) -> None:
        setup = _setup(SetupStatus.VALID)
        evidence = _evidence(EvidenceTier.LIMITED, resolved=15)
        decision = DecisionEngine(DecisionConfig(require_adequate_evidence=True)).decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.INSUFFICIENT_EVIDENCE

    def test_does_not_force_a_decision(self) -> None:
        """Even a structurally VALID setup with no evidence never
        becomes VALID_SETUP - insufficient evidence always wins."""
        setup = _setup(SetupStatus.VALID)
        decision = DecisionEngine().decide(setup, None, at=TS2)
        assert decision.state != DecisionState.VALID_SETUP


class TestConflictingEvidence:
    def test_negative_expected_r_still_reported_not_hidden(self) -> None:
        setup = _setup(SetupStatus.VALID)
        evidence = _evidence(EvidenceTier.ADEQUATE, p_win=0.3, expected_r=-0.4)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.VALID_SETUP  # structural state unaffected by evidence direction
        assert any("-0.4" in r or "-0.400" in r for r in decision.reasons)


class TestMissingConfirmation:
    def test_detected_setup_is_wait(self) -> None:
        setup = _setup(SetupStatus.DETECTED)
        evidence = _evidence(EvidenceTier.ADEQUATE)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.WAIT
        assert decision.missing_conditions != ()

    def test_confirming_setup_is_watch(self) -> None:
        setup = _setup(SetupStatus.CONFIRMING)
        evidence = _evidence(EvidenceTier.ADEQUATE)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.WATCH


class TestInvalidatedSetup:
    def test_invalidated_wins_over_adequate_evidence(self) -> None:
        setup = _setup(SetupStatus.INVALIDATED)
        evidence = _evidence(EvidenceTier.ADEQUATE)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.INVALIDATED

    def test_invalidated_wins_over_no_evidence(self) -> None:
        setup = _setup(SetupStatus.INVALIDATED)
        decision = DecisionEngine().decide(setup, None, at=TS2)
        assert decision.state == DecisionState.INVALIDATED

    def test_expired_maps_to_wait(self) -> None:
        setup = _setup(SetupStatus.EXPIRED)
        evidence = _evidence(EvidenceTier.ADEQUATE)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        assert decision.state == DecisionState.WAIT


class TestNoFabrication:
    def test_no_numeric_confidence_score_in_output(self) -> None:
        setup = _setup(SetupStatus.VALID)
        evidence = _evidence(EvidenceTier.ADEQUATE)
        decision = DecisionEngine().decide(setup, evidence, at=TS2)
        payload = decision.to_dict()
        assert "confidence_score" not in payload
        assert "ai_confidence" not in payload

    def test_invalidation_conditions_always_present_in_output(self) -> None:
        setup = _setup(SetupStatus.VALID)
        decision = DecisionEngine().decide(setup, _evidence(EvidenceTier.ADEQUATE), at=TS2)
        assert decision.invalidation_conditions


class TestEvidenceBridge:
    def test_no_recomputation_values_match_source(self) -> None:
        evaluation = _evaluation(EvidenceTier.ADEQUATE, p_win=0.55, expected_r=0.2)
        evidence = evidence_from_evaluation(evaluation)
        assert evidence.p_win == evaluation.probability.p_win
        assert evidence.historical_expectancy == evaluation.probability.expected_r
        assert evidence.evidence_tier == evaluation.evidence_tier

    def test_regime_filter_reflects_context_filter(self) -> None:
        from app.regimes.models import TrendState

        evaluation = _evaluation(EvidenceTier.ADEQUATE)
        evaluation = StrategyContextEvaluation(
            context_filter=OutcomeFilter(strategy_name="orb", trend_state=TrendState.UPTREND, session="london"),
            probability=evaluation.probability, evaluation_period_start=None, evaluation_period_end=None, as_of=None,
            evidence_tier=EvidenceTier.ADEQUATE, warnings=(),
        )
        evidence = evidence_from_evaluation(evaluation)
        assert evidence.regime_filter["trend_state"] == "uptrend"
        assert evidence.regime_filter["session"] == "london"


class TestSerialization:
    def test_decision_to_dict_is_json_safe(self) -> None:
        setup = _setup(SetupStatus.VALID)
        decision = DecisionEngine().decide(setup, _evidence(EvidenceTier.ADEQUATE), at=TS2)
        json.dumps(decision.to_dict())


class TestDeterminism:
    def test_identical_inputs_produce_identical_decision(self) -> None:
        setup = _setup(SetupStatus.VALID)
        evidence = _evidence(EvidenceTier.ADEQUATE)
        a = DecisionEngine().decide(setup, evidence, at=TS2)
        b = DecisionEngine().decide(setup, evidence, at=TS2)
        assert a.to_dict() == b.to_dict()
