"""M7: the DECISION layer of DATA -> CONTEXT -> OBSERVATIONS -> SETUP ->
EVIDENCE -> DECISION -> EXPLANATION.

Combines a `Setup` (app.setups) with M6 historical evidence
(`app.evaluation.models.StrategyContextEvaluation`, bridged here
without recomputation - see `app.decision.evidence`) into a `Decision`:
a deterministic, explainable ACTION STATE, never a bare prediction or a
fabricated confidence number. See module docstrings in `models.py` and
`engine.py` for the full contract.
"""
