"""M6: Strategy x Regime Evaluation & Robustness Layer.

Answers "how robust is the historical evidence for this strategy under
this market context?" - NOT "should we trade this right now?" (that is
a future decision layer's job). See docs/M6_STRATEGY_REGIME_EVALUATION.md.

Dependency direction (one-way, no cycles):

    M1 Data -> M2 Strategy -> M3 Research -> M4 Regime -> M5 Probability -> M6 Evaluation

`app.evaluation` orchestrates and interprets M3/M4/M5 outputs - it does
not reimplement probability math (M5), walk-forward windowing (M3), or
regime classification (M4). Nothing in M1-M5 imports from
`app.evaluation`.
"""
