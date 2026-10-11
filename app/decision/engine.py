"""The DecisionEngine: a deterministic, table-driven mapping from
(`Setup`, `M6Evidence`) to a `Decision`. No machine learning, no LLM
call, no numeric "confidence score" - every `Decision.state` is
justified entirely by `Setup.status` (from the deterministic state
machine, `app.setups.state_machine`) and `M6Evidence.evidence_tier`
(from M6, unmodified).

PRIORITY ORDER (checked top to bottom - the first matching rule wins):

1. `Setup.status == INVALIDATED` -> `DecisionState.INVALIDATED`.
   A structural invalidation is a fact about the setup itself; no
   amount of historical evidence changes it.
2. `Setup.status == EXPIRED` -> `DecisionState.WAIT`. Mission's
   `DecisionState` list has no dedicated "expired" action state - an
   expired setup is simply nothing to act on, which is what `WAIT`
   means everywhere else in this table too.
3. `evidence is None` or `evidence.evidence_tier == EvidenceTier.INSUFFICIENT`
   -> `DecisionState.INSUFFICIENT_EVIDENCE`. This is checked BEFORE
   the setup-status-based rules below, per the mission's explicit
   instruction (section 11): "Do not force a decision simply because
   the user asks for one" - a structurally VALID setup with no
   supporting historical sample is still insufficient evidence, not a
   valid setup.
4. `Setup.status == ENTRY_CONDITION_MET` -> `DecisionState.ENTRY_CONDITION_MET`.
5. `Setup.status == VALID` -> `DecisionState.VALID_SETUP`.
6. `Setup.status in (WATCH, CONFIRMING)` -> `DecisionState.WATCH`.
7. `Setup.status == DETECTED` -> `DecisionState.WAIT`.

No prose-generation logic lives here (mission section 12: "Do not put
prose-generation logic inside the quantitative engine") - `reasons`/
`summary` are assembled from small, fixed template strings driven by
which branch fired, not by any language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.decision.models import Decision, DecisionState, M6Evidence
from app.evaluation.models import EvidenceTier
from app.setups.models import Setup, SetupStatus


@dataclass(frozen=True)
class DecisionConfig:
    # Evidence at exactly EvidenceTier.LIMITED still produces a
    # decision (with a warning) rather than INSUFFICIENT_EVIDENCE by
    # default - set this True for a stricter policy that treats LIMITED
    # the same as INSUFFICIENT.
    require_adequate_evidence: bool = False


def _missing_conditions(setup: Setup) -> tuple[str, ...]:
    if setup.status == SetupStatus.DETECTED:
        return ("Required structure confirmation has not occurred.",)
    if setup.status in (SetupStatus.WATCH, SetupStatus.CONFIRMING):
        return ("Setup has not yet reached VALID status.",)
    if setup.status == SetupStatus.VALID:
        return ("Entry condition has not yet been met.",)
    return ()


class DecisionEngine:
    def __init__(self, config: Optional[DecisionConfig] = None) -> None:
        self.config = config or DecisionConfig()

    def decide(self, setup: Setup, evidence: Optional[M6Evidence], at: datetime) -> Decision:
        reasons: list[str] = [f"Setup {setup.setup_type.value} is currently {setup.status.value.upper()}."]
        warnings: list[str] = list(evidence.warnings) if evidence is not None else []
        invalidation_texts = tuple(ic.description for ic in setup.invalidation_conditions)
        supporting_observations = tuple(o.to_dict() for o in setup.observations)
        supporting_evidence = evidence.to_dict() if evidence is not None else None

        if setup.status == SetupStatus.INVALIDATED:
            state = DecisionState.INVALIDATED
            reasons.append("The setup's invalidation condition has occurred.")
            missing = ()
        elif setup.status == SetupStatus.EXPIRED:
            state = DecisionState.WAIT
            reasons.append("The setup has expired and is no longer temporally relevant.")
            missing = ()
        elif evidence is None:
            state = DecisionState.INSUFFICIENT_EVIDENCE
            reasons.append("No historical evidence is available for this strategy/context combination.")
            missing = ("Historical evidence (M6 evaluation) has not been supplied.",)
        elif evidence.evidence_tier == EvidenceTier.INSUFFICIENT or (
            self.config.require_adequate_evidence and evidence.evidence_tier == EvidenceTier.LIMITED
        ):
            state = DecisionState.INSUFFICIENT_EVIDENCE
            reasons.append(
                f"Historical evidence sample is {evidence.evidence_tier.value.upper()} "
                f"({evidence.resolved_observations} resolved observation(s))."
            )
            missing = ("A larger resolved historical sample for this strategy/context combination.",)
        else:
            if evidence.evidence_tier == EvidenceTier.LIMITED:
                warnings.append(
                    f"Evidence tier is LIMITED ({evidence.resolved_observations} resolved observations) - "
                    "interpret with caution."
                )
            if evidence.p_win is not None:
                reasons.append(f"Historical P(win) for this context is {evidence.p_win:.3f}.")
            if evidence.historical_expectancy is not None:
                reasons.append(f"Historical expected R is {evidence.historical_expectancy:.3f}.")

            missing = _missing_conditions(setup)
            if setup.status == SetupStatus.ENTRY_CONDITION_MET:
                state = DecisionState.ENTRY_CONDITION_MET
                reasons.append("The defined entry condition has been satisfied.")
            elif setup.status == SetupStatus.VALID:
                state = DecisionState.VALID_SETUP
                reasons.append("The setup satisfies its defined structural requirements.")
            elif setup.status in (SetupStatus.WATCH, SetupStatus.CONFIRMING):
                state = DecisionState.WATCH
                reasons.append("Conditions are developing but required confirmation is not yet complete.")
            else:  # DETECTED
                state = DecisionState.WAIT
                reasons.append("A potentially relevant setup exists, but nothing further is confirmed yet.")

        reasons.extend(f"Observation: {o.observation_type.value} at {o.timestamp.isoformat()}" for o in setup.observations)

        summary = f"{setup.instrument_symbol} {setup.timeframe} {setup.direction} {setup.setup_type.value} -> {state.value.upper()}"

        return Decision(
            state=state,
            setup_id=setup.id,
            summary=summary,
            reasons=tuple(reasons),
            supporting_observations=supporting_observations,
            supporting_evidence=supporting_evidence,
            missing_conditions=missing,
            invalidation_conditions=invalidation_texts,
            warnings=tuple(warnings),
            timestamp=at,
        )
