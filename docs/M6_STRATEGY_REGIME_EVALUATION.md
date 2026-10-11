# M6 — Strategy × Regime Evaluation & Robustness Layer

> **M6 evaluates historical evidence. It does not rank strategies,
> recommend trades, predict future returns, or determine position
> size.**

## Purpose

M5 can answer "what's `P(win)` for ORB, conditioned on London + high
volatility?" for one filter at a time. M6 is the orchestration layer on
top: composable evaluation of strategy × context combinations, an
explicit sample-size verdict, temporal (walk-forward) breakdowns, and
side-by-side comparison — all without inventing new statistics or a
ranking.

```
Market Data (M1)
    -> Features (M4)
    -> Regime Context (M4) + Strategy Signal (M2)
    -> Outcome Labeling (M5)
    -> Historical Outcomes (M5)
    -> M5 Probability (ProbabilityEstimator)
    -> M6 Strategy x Regime Evaluation   <- this milestone
    -> Future Risk / Ranking / Decision (M7+)
```

## Architecture

```
app/evaluation/
    models.py        EvidenceTier, EvaluationConfig, StrategyContextEvaluation,
                       WalkForwardWindowEvaluation, StabilityMetrics, WalkForwardEvaluation
    evaluator.py       StrategyEvaluator (evaluate / evaluate_combinations / evaluate_walk_forward),
                       summarize_robustness_result
    stability.py        compute_stability_metrics (pure aggregation, no new statistics)
    comparison.py        compare_evaluations, ComparisonRow, MULTIPLE_TESTING_CAVEAT
    reporting.py          format_report, format_comparison_report (presentation only)
```

M6 imports `app.probability` (M5: `ProbabilityEstimator`, `OutcomeFilter`,
`OutcomeDataset`, `TradeOutcome`), `app.regimes` (M4: `RegimeContext`,
via outcomes that already embed it), and `app.backtesting.validation`
(M3: `walk_forward_windows`) — reused as-is, never reimplemented.
**Nothing in M1–M5 imports from `app.evaluation`.**

## Reuse, not duplication

Every piece of math in M6 is imported, not rewritten:

| What | Reused from | M6's own contribution |
|---|---|---|
| Empirical probability, Wilson interval, expected R | `app.probability.estimators.ProbabilityEstimator` | Wraps the result with a sample-size verdict |
| Composable conditional filtering | `app.probability.filters.OutcomeFilter` | None — used directly as `StrategyContextEvaluation.context_filter` |
| Walk-forward window boundaries | `app.backtesting.validation.walk_forward_windows` (M3) | Applies the windows to an `OutcomeDataset` instead of running backtests |
| Regime state | `app.regimes.context.RegimeContext` (M4), already embedded in every `TradeOutcome` | None — read via `OutcomeFilter`'s existing regime fields |
| Parameter-neighborhood robustness | `app.research.robustness.RobustnessResult` (M3) | `summarize_robustness_result()` lifts a read-only summary for the report — the underlying analysis is never recomputed |

`StrategyContextEvaluation` embeds M5's `ProbabilityEstimate` and
`OutcomeFilter` **whole**, rather than flattening their fields onto
itself — per the mission's explicit "do not copy every field blindly
if existing M5 objects can be embedded or reused."

## Evaluation dimensions

`StrategyEvaluator.evaluate()` takes an `OutcomeFilter` — the same
composable spec M5 already defines, covering strategy / instrument /
timeframe / direction / trend state / volatility state / structure
state / session. There is no `evaluate_orb_london()`-style function and
none should ever be added — every combination is just a different
`OutcomeFilter` instance passed to the same method (mission section 6).

## Minimum sample policy

```python
EvaluationConfig(
    minimum_resolved_observations=30,  # ADEQUATE at/above this
    limited_resolved_observations=10,  # LIMITED between this and the line above; INSUFFICIENT below it
)
```

Both defaults are documented guesses (same convention as M4's regime
thresholds), not values tuned against this project's data — configurable
per the mission's explicit requirement. **A result below the threshold
is never deleted, hidden, or borrowed from a broader population** — the
full `ProbabilityEstimate` (counts, `p_win`, interval, everything) stays
on the result; only `evidence_tier` and `warnings` change.

`EvidenceTier` (`INSUFFICIENT` / `LIMITED` / `ADEQUATE`) is **research
metadata, not a quality score** — it is never averaged, summed, or
converted into a number (mission section 9).

## Walk-forward methodology

`StrategyEvaluator.evaluate_walk_forward()` carves `[period_start,
period_end)` into windows via M3's `walk_forward_windows` (unmodified),
then evaluates each window's own `ProbabilityEstimate` from outcomes
that satisfy **both**:

1. `test_start <= signal_timestamp < test_end` — the signal actually
   fired during this window.
2. `exit_timestamp < test_end` (via `OutcomeDataset.resolved_before`,
   M5's own causal mechanism, reused unchanged) — the outcome's fate
   was actually knowable by the time the window closed.

Both conditions are mandatory; neither alone is sufficient (a signal
that fires in the window's last bar with a long evaluation horizon can
resolve well after `test_end` — condition 2 is what excludes it). See
"Temporal honesty" below for the tests proving this.

Unlike `app.research.runner.run_walk_forward_experiment` (M3), which
**raises** when no windows fit the period (a caller error in an
execution context), `evaluate_walk_forward` **does not raise** when no
windows fit — it returns an empty `WalkForwardEvaluation` plus a
warning. This is a deliberate difference for a deliberate reason: M3's
version drives actual backtest execution, where "nothing to run" is a
mistake; M6's version drives a research report, where the mission
explicitly lists "no valid walk-forward windows" as an edge case to
*represent*, not reject.

## Stability metrics

`compute_stability_metrics()` (`stability.py`) is pure aggregation —
mean/median/stdev/min/max over each window's own `expected_r`/`p_win`,
nothing estimated fresh. **No composite "robustness score" exists** —
the mission is explicit that M6 should "preserve the underlying
evidence," not compress it into one opaque number. A window with zero
resolved observations contributes `None` (excluded from the aggregate,
not treated as `0`) to `expected_r`/`p_win`-based stats, but still
counts toward `number_of_windows`.

## Robustness integration (M3)

`summarize_robustness_result()` takes an M3 `RobustnessResult`
(structurally, not by import — see the function's own docstring for
why) and lifts `parent_experiment_id`, `parameter_name`,
`values_tested`, `stability_label`, and `notes` into a plain dict for
attachment via `StrategyEvaluator.evaluate(..., robustness_summary=...)`.
**No parameter-neighborhood math is recomputed** — the M3 experiment
remains the source of truth; M6 only carries a pointer and a summary
forward into the report.

## Comparison semantics

`compare_evaluations()` returns rows in **exactly the order supplied**
— never re-sorted by any metric (same design decision M3's own
`app.research.comparison.compare_experiments` already made, for the
identical reason: sorting by a metric is itself a ranking judgment).
`ComparisonRow` has no `rank`, `score`, or `winner` field, and none
should ever be added — comparison exposes evidence side by side; it
never declares one config better than another (mission section 16, M8's
job).

## Multiple-testing awareness

`evaluate_combinations()` never auto-searches every possible filter
combination — it evaluates exactly the list the caller supplies
(mission section 15/18, avoiding combinatorial explosion and unearned
significance). `format_comparison_report()` appends
`comparison.MULTIPLE_TESTING_CAVEAT` whenever more than one row is
being compared, making the selection-bias risk visible in the report
itself rather than only in documentation someone has to remember to
read.

## Temporal honesty

Tested directly in `tests/evaluation/test_causality.py`:

- `evaluate(..., as_of=T)`: appending outcomes that resolve after `T`
  never changes the result for `T` (`TestBaselineEvaluationCausality`).
- `evaluate_walk_forward`: a window's population never includes a
  signal that fires after the window ends
  (`TestWalkForwardWindowCausality::test_window_excludes_signals_that_fire_after_the_window`),
  **and separately** never includes an outcome that resolves after the
  window ends even if its signal fired inside the window
  (`...::test_window_excludes_outcomes_unresolved_before_window_end` —
  this is the case a signal-timestamp-only check would miss).
- Adding data far in the future never changes an earlier window's
  already-computed result
  (`...::test_future_window_data_does_not_change_an_earlier_windows_result`).
- Identical inputs produce identical evaluations, always
  (`TestDeterminism`).

## Known limitations

- Sample-size thresholds (30 / 10) are documented defaults, not
  empirically validated — same caveat as every threshold in M4/M5.
- No multiple-testing *correction* (e.g. Bonferroni) is implemented —
  the mission explicitly scopes this out ("do NOT implement an
  elaborate multiple-testing correction unless the existing
  architecture already supports it cleanly"); M6 only makes the risk
  visible via the caveat text and by exposing the evaluated-
  configuration count.
- `evaluate_walk_forward`'s windows carve the outcome dataset by
  `signal_timestamp`/`exit_timestamp`, not by re-running any
  backtest — an `OutcomeDataset` is treated as a fixed, already-labeled
  population; M6 does not re-label outcomes per window.
- No HTTP API was added — the mission explicitly permits this ("the
  core evaluator must remain usable directly from Python. Do not force
  research users through HTTP") and no genuinely useful endpoint
  presented itself beyond what direct Python calls already provide
  cleanly.

## Example

```python
from app.evaluation.evaluator import StrategyEvaluator
from app.evaluation.models import EvaluationConfig
from app.evaluation.comparison import compare_evaluations
from app.evaluation.reporting import format_report, format_comparison_report
from app.probability.filters import OutcomeFilter
from app.regimes.models import TrendState, VolatilityState

evaluator = StrategyEvaluator(EvaluationConfig(minimum_resolved_observations=30))

result = evaluator.evaluate(
    dataset,
    OutcomeFilter(strategy_name="orb", instrument_symbol="EURUSD", session="london",
                  trend_state=TrendState.UPTREND, volatility_state=VolatilityState.HIGH),
)
print(format_report(result))

wf_result = evaluator.evaluate_walk_forward(
    dataset, period_start=start, period_end=end, train_days=30, test_days=7,
    filter=OutcomeFilter(strategy_name="orb"),
)
print(format_report(wf_result))  # includes the Walk-forward: section

up = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="orb", trend_state=TrendState.UPTREND))
down = evaluator.evaluate(dataset, OutcomeFilter(strategy_name="orb", trend_state=TrendState.DOWNTREND))
print(format_comparison_report(compare_evaluations([up, down], labels=["uptrend", "downtrend"])))
```
