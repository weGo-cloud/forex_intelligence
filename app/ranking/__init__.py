"""M8: Validated Strategy Ranking & FIAS Foundation.

    M3 Robustness/Walk-forward + M5 Probability + M6 Evaluation
            v
    M8 Validation Hardening  (closes gaps M6 only documented as caveats)
            v
    M8 FIAS (transparent, multi-factor, gated)  ->  M8 Ranking (comparative, not prescriptive)

Nothing here estimates a new statistic - every input is an existing M3/M5/M6
object (`app.research.robustness.RobustnessResult`,
`app.evaluation.models.StrategyContextEvaluation`). M8 adds a documented
correction procedure (Part A), a documented composite-score formula
(Part B), and a documented ranking/ordering procedure (Part C) on top
of them - all inspectable, none of it a fitted or learned model.

Dependency direction preserved: M8 imports from M3/M5/M6; nothing
upstream imports from `app.ranking`.
"""
