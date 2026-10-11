"""M7: canonical setup representation, deterministic state machine, and
setup detection - the SETUP layer of DATA -> CONTEXT -> OBSERVATIONS ->
SETUP -> EVIDENCE -> DECISION -> EXPLANATION.

A `Setup` is a candidate trading structure assembled from `Observation`s
(`app.observations`) - it carries no historical evidence and no
decision of its own; `app.decision` is what combines a `Setup` with M6
evidence to produce a `Decision`.
"""
