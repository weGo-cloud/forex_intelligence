"""M5: Outcome Modeling & Probability Foundation.

Answers "given a historical strategy signal and its market context,
what outcomes followed historically?" - a transparent statistical
baseline, not a prediction. See docs/M5_OUTCOME_PROBABILITY.md.

Dependency direction (one-way, no cycles):

    M1 Data -> M2 Strategy -> M4 Regime -> M5 Outcome/Probability

`app.probability` imports from `app.strategies.base` (Signal, Direction)
and `app.regimes.context` (RegimeContext) - nothing in M1/M2/M4 imports
anything from `app.probability`.
"""
