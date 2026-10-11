"""M7: structured market observations - the first layer of the
DATA -> CONTEXT -> OBSERVATIONS -> SETUP -> EVIDENCE -> DECISION ->
EXPLANATION pipeline.

Every detector in `app.observations.detectors` is built entirely from
existing M1-M6 primitives (`detect_swing_points`, `detect_fvg`,
`label_swings`, `RegimeContext`) - nothing here recomputes an
indicator or reclassifies a regime; this layer only packages already-
causal M1-M6 outputs into a uniform `Observation` shape multiple future
sources (market data today, screenshots later - see module docstring
in `app.observations.models`) can all produce.
"""
