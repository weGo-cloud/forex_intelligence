# M5 — Outcome Modeling & Probability Foundation

> **M5 probabilities are empirical historical statistics conditioned on
> defined observations. They are not guaranteed future probabilities
> and are not trading recommendations.**

## Purpose

M4 describes market conditions. M5 answers the next question: *given a
historical signal and the market context it fired in, what actually
happened afterward?* — a transparent statistical baseline, not a
prediction.

```
Market Data (M1)
    -> Features (M4's app.features.*)
    -> Regime Context (M4) + Strategy Signal (M2)
    -> Outcome Labeling (M5, new)
    -> Historical Outcome Dataset (M5, new)
    -> Probability Estimation (M5, new)
```

## Architecture

```
app/probability/
    models.py       OutcomeLabel, ExitReason, TradeOutcome
    labeling.py      LabelingConfig, label_outcome, label_outcomes_bulk
    outcomes.py       OutcomeDataset (filter, resolved_before, group_by, serialize)
    filters.py         OutcomeFilter (composable conditional query spec)
    estimators.py       SampleCounts, WilsonInterval, ProbabilityEstimate, ProbabilityEstimator
```

Simplified from the mission's suggested layout (`calibration.py` +
`service.py` were folded into `estimators.py` - per the mission's own
"you may simplify... do not create unnecessary abstractions"; see
"Deviations" below for why).

Dependency direction (one-way, matching the mission's required
boundary):

```
M1 Data -> M2 Strategy -> M4 Regime -> M5 Outcome/Probability
```

`app.probability` imports `Signal`/`Direction` (M2) and `RegimeContext`
(M4). **Nothing in M1, M2, or M4 imports from `app.probability`** — no
circular dependency was introduced, and none was needed.

## Why `TradeOutcome` is a new model, not a reuse of `Trade`

The mission asks to "use the existing project's actual strategy/
backtest types where possible" and "not duplicate existing trade/result
models unnecessarily." `app.backtesting.results.Trade` was evaluated
first, and reused where the underlying concept is actually identical
(`Direction`). It was **not** reused as the outcome record itself: `Trade`
carries `position_size`, `gross_pnl`, `net_pnl`, and cost-application
fields that only mean something inside a full, equity-tracked backtest
run — M5 labels signals in isolation, with no equity curve or position
sizing at all (explicitly M2/M3's job, not M5's, per the mission's hard
scope boundary). Forcing `TradeOutcome` to *be* a `Trade` would mean
either leaving those fields meaninglessly unset or fabricating values
for them — worse than a smaller, honestly-scoped model. `RegimeContext`
(M4) **is** embedded as-is, unmodified, since there's no such mismatch there.

## Outcome definitions

`OutcomeLabel` (the coarse statistical bucket, per mission section 5):
`WIN` / `LOSS` / `BREAKEVEN` / `UNRESOLVED`.

`ExitReason` (the finer mechanical cause, recorded alongside the label):
`TAKE_PROFIT`, `STOP_LOSS`, `TIME_EXPIRY`, `SAME_BAR_UNRESOLVED`,
`INSUFFICIENT_DATA`. The mapping:

| ExitReason | OutcomeLabel |
|---|---|
| `TAKE_PROFIT` | `WIN` |
| `STOP_LOSS` | `LOSS` |
| `TIME_EXPIRY` | `WIN` / `LOSS` / `BREAKEVEN`, by the sign of the resolved `r_multiple` (within `breakeven_r_epsilon` of 0 → `BREAKEVEN`) |
| `SAME_BAR_UNRESOLVED` | `UNRESOLVED` |
| `INSUFFICIENT_DATA` | `UNRESOLVED` |

`UNRESOLVED` deliberately merges the two "we cannot honestly assign a
verdict" reasons into one statistical bucket (matching the mission's
minimal 4-label set) while keeping the distinct *reason* on
`exit_reason` for anyone who wants to split them back out.

## Barrier methodology (triple-barrier)

```
entry
  ├── take-profit barrier   (direction-aware: upward for LONG, downward for SHORT)
  ├── stop-loss barrier     (direction-aware: downward for LONG, upward for SHORT)
  └── time barrier          (evaluation_horizon_bars)
```

One shared, direction-aware implementation (`label_outcome` in
`labeling.py`) — not duplicated per direction, per the mission's
explicit instruction. `direction_sign` (`+1`/`-1`) and a pair of
direction-conditioned comparisons are the only place LONG/SHORT differ.

## Entry semantics

Entry fills at the **open of the bar after** `signal.signal_time`;
barrier monitoring begins the bar **after that** — identical to
`app.backtesting.engine.BacktestEngine`'s own
`exits_start_bar_after_entry=True` convention (see `engine.py`). This
is a deliberate match: a signal generated after a candle closes must
never enter at an earlier price, and the mission explicitly says to
reuse M2's timing convention where it already exists. It's a *separate*
implementation, not a call into `BacktestEngine.run`, because the two
solve different problems — see `labeling.py`'s module docstring for
the full reasoning (no equity curve/position sizing in M5; M5 needs an
`UNRESOLVED`-capable same-bar policy `BacktestConfig` doesn't offer).

## Same-bar TP/SL collision policy

`LabelingConfig.same_bar_policy`, one of:

- `"unresolved"` — **the default.** Neither barrier is assumed to have
  won; `exit_reason = SAME_BAR_UNRESOLVED`, `exit_price`/`r_multiple`
  are `None`, `outcome_label = UNRESOLVED`. This differs from M2's
  `BacktestEngine` default (`"stop_loss_first"`) *on purpose*: a full
  backtest's equity curve needs *some* deterministic choice to keep
  running trade after trade, but a probability estimate's statistical
  integrity is more important than completeness — guessing here would
  inflate (or deflate) `P(win)` in exactly the way the mission's final
  principle ("do not optimize for impressive percentages") warns
  against.
- `"stop_first"` — conservative: assume the stop was touched first.
- `"target_first"` — assume the target was touched first.

**Limitation, stated plainly**: OHLC data alone genuinely cannot tell
you which barrier was touched first within one candle. `"unresolved"`
is honest about that; `"stop_first"`/`"target_first"` exist for callers
who have a specific, defensible reason to assume one (e.g. matching a
downstream system's own convention) — pick a policy, don't silently get
one.

## Probability calculation

Empirical, not fitted — direct counts over whatever's in the
(optionally filtered/time-cut) `OutcomeDataset`:

```
P(win)  = wins / resolved_observations
P(loss) = losses / resolved_observations
P(breakeven) = breakevens / resolved_observations
P(unresolved) = unresolved / total_observations   (different denominator - see below)
```

`resolved_observations = wins + losses + breakevens` — `UNRESOLVED`
outcomes are excluded from the win/loss/breakeven denominator entirely
(you can't compute a win rate including outcomes you don't have a
verdict for), but are counted against `total_observations` for
`P(unresolved)`, which answers a different question ("of everything
that matched, how much of it was even resolvable?").

## Conditional filtering

`OutcomeFilter` (`filters.py`) — every field `Optional`, `None` meaning
"don't filter on this dimension":

```python
OutcomeFilter(strategy_name="orb")
OutcomeFilter(strategy_name="orb", instrument_symbol="EURUSD", session="london")
OutcomeFilter(strategy_name="orb", trend_state=TrendState.UPTREND, volatility_state=VolatilityState.HIGH)
```

One composable spec, not a function per combination (mission section
17). `session` is a **membership** test (`RegimeContext.sessions` is a
tuple — sessions can overlap), not equality. A regime-dimension filter
(`trend_state`/`volatility_state`/`structure_state`/`session`) can never
match an outcome with `regime_context=None` — an explicit "cannot
match" rather than silently ignoring the condition (mission section 23:
"missing regime context" is a named edge case).

## Sample-size handling

Every `ProbabilityEstimate` carries its own `SampleCounts`
(`total_observations`, `resolved_observations`, `wins`, `losses`,
`breakevens`, `unresolved`) — never hidden, never optional. If a filter
matches zero outcomes, every probability field is `None` and the counts
show `0` — **not** a fallback to a broader, unrequested population
(mission section 18).

## Uncertainty methodology

Wilson score interval (`estimators.wilson_score_interval`), computed
via `statistics.NormalDist().inv_cdf` (Python stdlib — no new
dependency). Chosen specifically over the plain normal ("Wald")
approximation because it stays well-behaved (bounded within `[0, 1]`,
sane coverage) at the small sample sizes M5's conditional queries are
likely to produce. `ProbabilityEstimate.win_interval` is `None`
whenever `resolved_observations == 0` — never computed against zero.
This is a **statistical** confidence interval, explicitly not any kind
of "trading confidence score" (mission section 14).

## Expected-R calculation

Computed only over resolved (non-`UNRESOLVED`) outcomes:

```
expected_R = mean(r_multiple for every resolved outcome)
avg_win_R  = mean(r_multiple for WIN outcomes)
avg_loss_R = mean(r_multiple for LOSS outcomes)
```

All three are `None` when `resolved_observations == 0`.

## Causality rules

The mission calls this "the most important M5 requirement" (section
10). Two distinct guarantees, both tested directly:

1. **The labeler never looks beyond its own horizon.** `label_outcome`
   only ever reads `execution_df` rows in
   `[entry_pos + 1, entry_pos + evaluation_horizon_bars]` — appending
   or modifying data outside that range (even wildly, even right after
   the horizon ends) cannot change an already-computed `TradeOutcome`
   (`tests/probability/test_causality.py::TestLabelerDoesNotUseDataBeyondHorizon`).
2. **A probability estimate "as of" time `T` cannot be changed by
   outcomes that resolve after `T`.** `TradeOutcome.exit_timestamp` is
   **always** set once labeling concludes — even for `UNRESOLVED`
   outcomes, since "this is unresolvable" is itself information that
   became known at a specific point in time. This is what makes one
   simple rule sufficient for causal filtering:
   `OutcomeDataset.resolved_before(T)` = every outcome with
   `exit_timestamp <= T`, uniformly across every `ExitReason`
   (`tests/probability/test_causality.py::TestFutureOutcomesCannotAffectPastEstimate`).
   `ProbabilityEstimator.estimate(dataset, as_of=T)` applies this
   before any other filter.

## Research integration (M3)

No changes were made to `app.research` — none were needed.
`label_outcomes_bulk` + `ProbabilityEstimator` are plain, stateless
function calls a future M3-based research script can invoke directly
(e.g. alongside `execute_experiment`), the same pattern M4's
`RegimeClassifier` already established. A future workflow evaluating
`strategy + regime + outcome definition + parameter configuration`
together is a script-level composition of M2 (signals) + M4 (regime) +
M5 (outcomes/probability) + M3 (experiment orchestration), with zero
changes to any of the four layers.

## Known limitations

- Same-bar collision is fundamentally undecidable from OHLC data alone
  — the `"unresolved"` default is honest about this, not a workaround.
- No smoothing/hierarchical pooling for sparse conditioning
  combinations (mission section 18: explicitly deferred past M5 V1) —
  a 3-observation query returns exactly that: 3 observations, a wide
  Wilson interval, and nothing borrowed from a broader population.
- `compare_groups` is a *foundation* for future calibration analysis
  (grouped empirical win rates with honest per-group sample sizes), not
  calibration itself — there is no predicted probability in M5 to
  calibrate against yet; that's explicitly M6+'s job.
- `OutcomeDataset` is an in-memory Python collection, not a database
  table — matching the mission's "do not create unnecessary database
  infrastructure." Persisting outcome datasets (if ever needed) is a
  future concern, not M5's.

## Deviations from the mission

- **Project structure simplified**: `calibration.py` and `service.py`
  (from the mission's suggested layout) were folded into
  `estimators.py` — `ProbabilityEstimator.compare_groups` covers the
  "calibration foundation" requirement, and a separate one-class
  `service.py` would have added an abstraction layer with nothing of
  its own to do. Permitted explicitly by the mission ("simplify or
  adjust... do not create unnecessary abstractions").
- **`same_bar_policy` default differs from M2's `BacktestEngine`
  default** (`"unresolved"` vs. `"stop_loss_first"`) — a deliberate,
  documented choice (see "Same-bar TP/SL collision policy" above), not
  an inconsistency.

## Example

```python
from app.probability.labeling import LabelingConfig, label_outcomes_bulk
from app.probability.outcomes import OutcomeDataset
from app.probability.filters import OutcomeFilter
from app.probability.estimators import ProbabilityEstimator
from app.regimes.models import TrendState, VolatilityState

config = LabelingConfig(evaluation_horizon_bars=20, same_bar_policy="unresolved")
outcomes = label_outcomes_bulk(signals, execution_df, config, "EURUSD", "5M",
                                regime_context_by_signal_time=ctx_by_time)
dataset = OutcomeDataset.from_list(outcomes)

estimator = ProbabilityEstimator(confidence_level=0.95)
result = estimator.estimate(
    dataset,
    filter=OutcomeFilter(strategy_name="orb", trend_state=TrendState.UPTREND, volatility_state=VolatilityState.HIGH),
)

print(result.p_win, result.win_interval, result.counts)
# 0.61, WilsonInterval(lower_bound=0.54, upper_bound=0.68, sample_size=1842, ...), SampleCounts(...)

# Walk-forward-safe: only outcomes that had already resolved by T are eligible.
past_result = estimator.estimate(dataset, as_of=some_historical_timestamp)
```
