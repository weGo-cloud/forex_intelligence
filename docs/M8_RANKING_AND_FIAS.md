# M8 — Validated Strategy Ranking & FIAS Foundation

> **M8 ranks historical evidence within a caller-supplied batch. It does
> not predict future returns, guarantee outcomes, recommend trades, or
> execute anything.**

## Purpose

M6 could evaluate one strategy/context combination at a time, honestly,
with sample-size caveats. What it couldn't do: compare *many* combinations
without the comparisons themselves introducing bias, or compress evidence
into one defensible number without hiding what went into it. M8 closes
both gaps.

```
M3 Robustness/Walk-forward + M5 Probability + M6 Evaluation
        |
M8 Part A — Validation Hardening   (closes gaps M6 only documented as caveats)
        |
M8 Part B — FIAS                    (transparent, multi-factor, gated composite)
        |
M8 Part C — Ranking                 (comparative, batch-scoped, never "the best strategy")
```

## Architecture

```
app/ranking/
    validation.py   one_sided_binomial_p_value, benjamini_hochberg_adjusted,
                     OptimizationProvenance, check_optimization_leakage, robustness_gate
    fias.py          FIASConfig, FIASComponents, FIASScore, compute_fias
    ranking.py        RankedEvaluation, rank_evaluations, RANKING_CAVEAT
    reporting.py        format_ranking_report
```

One additive change to M6: `app/evaluation/comparison.py`'s private
label helper was renamed to the public `default_label` (with a
backward-compatible `_default_label` alias) so M8 reuses it instead of
duplicating four lines of logic. No behavior changed; M6's own test
suite is unmodified and still green.

## Part A — Validation hardening

**Multiple-testing correction.** M6 only exposed `MULTIPLE_TESTING_CAVEAT`
as warning text. M8 actually corrects: each evaluation's resolved
win/loss counts are tested against a null win rate (default 0.5) via an
*exact* one-sided binomial test (`math.comb`, no `scipy` dependency,
no normal approximation — consistent with M5's Wilson-interval
reasoning for staying exact at small samples). Benjamini-Hochberg then
controls the false-discovery rate across the whole ranked batch,
producing a `q_value` and `significant` flag per entry. This is a
real statistical procedure, not a caveat — but it reduces false
positives, it does not eliminate the risk, and `RANKING_CAVEAT` says so
explicitly in every report with more than one entry.

**Optimization-leakage guard.** `OptimizationProvenance` is supplied by
the caller — M8 cannot infer on its own whether a strategy's parameters
came from an M3 sweep. When supplied, `check_optimization_leakage`
flags (and excludes) any evaluation whose period overlaps the
parameter-selection period. **Stated plainly: no provenance means no
check, not a clean bill of health** — this is documented in the
function's own docstring, not hidden in prose elsewhere.

**Robustness gate.** An M3 `RobustnessResult` summarized as
`"isolated"` or `"insufficient_data"` (via
`app.evaluation.evaluator.summarize_robustness_result` — reused
unchanged, M3's robustness math is never recomputed) disqualifies that
combination. Same caveat as above: no robustness summary attached
means no gate, not an assumption of robustness.

## Part B — FIAS

**Not one opaque number.** `FIASScore.components` (`FIASComponents`)
is always attached and always serialized alongside the composite:

| Component | Formula | Bounded by construction |
|---|---|---|
| `sample_adequacy` | `{INSUFFICIENT:0.0, LIMITED:0.5, ADEQUATE:1.0}[evidence_tier]` | yes |
| `win_rate_confidence` | `1 - (Wilson interval width)` | yes (width ∈ [0,1]) |
| `stability` | `windows_with_positive_expected_r / number_of_windows` | yes |
| `robustness` | `{"stable":1.0, "sensitive":0.5, "isolated":0.0, "insufficient_data":0.0}[label]` | yes |

No arbitrary normalization constants anywhere — every component is
naturally bounded `[0, 1]`. The composite is a configurable weighted
mean over **only the components that could actually be computed**: no
walk-forward run → `stability=None`, excluded from the average, never
treated as `0`. If *nothing* could be computed, the score is withheld
(`None`) with a reason — never defaulted.

**Withheld below a threshold.** `FIASConfig.minimum_tier_for_scoring`
(default `EvidenceTier.LIMITED`) means `INSUFFICIENT`-tier evaluations
never get a FIAS score at all, regardless of how good their raw numbers
look — matching the project-wide "never fabricate precision on a thin
sample" rule.

## Part C — Ranking

`rank_evaluations(evaluations, ...)` ranks a **caller-supplied batch
only** — never an automatic search, never "the best strategy in the
system" (mirrors M6's `evaluate_combinations`, which never auto-searches
either). Unlike M6's `compare_evaluations` (which deliberately preserves
caller order for a side-by-side view), ranking **does** reorder its
output — establishing an order is the point — but:

- Excluded entries (`INSUFFICIENT` tier, robustness-disqualified,
  leakage-flagged, or FIAS-withheld) get `rank=None` and are placed
  **after** every ranked entry — never deleted, always carrying their
  full evaluation and `exclusion_reasons`.
- Ties are broken deterministically by label (alphabetical) — never by
  insertion order, which would be arbitrary.
- Every entry carries `p_value`, `q_value`, `significant`, `eligible`,
  and the full `FIASScore` (components included) — nothing about how
  the order was produced is hidden behind the rank number itself.

## Statistical honesty, concretely

A `P(win) = 0.70` entry with `n = 12` does **not** automatically
outrank a `P(win) = 0.61` entry with `n = 1,842` — FIAS's
`win_rate_confidence` component (interval width) and `sample_adequacy`
component both penalize the small sample directly, and the small
sample is likely `LIMITED`-tier to begin with (lower `sample_adequacy`).
Nothing in this milestone ever compares raw `P(win)` values without
passing them through sample-size-aware components first.

## Known limitations

- Multiple-testing correction uses a **fixed null hypothesis** (default
  win rate 0.5) — appropriate for "does this beat a coin flip", not
  necessarily for every strategy's actual economic breakeven rate
  (which depends on its risk/reward ratio). Callers comparing
  strategies with very different R-multiples should supply a `null_p`
  that reflects each one's real breakeven win rate, not rely on the
  default for everything.
- The optimization-leakage guard and robustness gate both degrade to
  "not checkable" when no provenance/robustness data is supplied —
  this is a real gap if a caller doesn't bother attaching them, stated
  plainly rather than quietly assumed away.
- FIAS weights are configurable but the *default* weights (all equal)
  are a documented choice, not empirically validated — same caveat
  class as every other default threshold in M4–M7.
- No persistence layer — `rank_evaluations` is a pure, stateless
  function over in-memory M6 objects, consistent with M6/M7's own
  "no unnecessary database infrastructure" choice.
- No HTTP API was added for M8 — ranking is a batch research operation
  best run from a script (same reasoning M6 gave for skipping an API).

## Example

```python
from app.ranking.ranking import rank_evaluations
from app.ranking.validation import OptimizationProvenance
from app.ranking.reporting import format_ranking_report
from app.evaluation.evaluator import summarize_robustness_result

evaluations = [evaluation_orb, evaluation_rsi, evaluation_mean_reversion]
robustness_summaries = [summarize_robustness_result(r) for r in robustness_results]  # or None per entry
provenances = [OptimizationProvenance(source_experiment_id="exp_000042",
                                       selection_period_start=sweep_start, selection_period_end=sweep_end), None, None]

results = rank_evaluations(evaluations, robustness_summaries=robustness_summaries, provenances=provenances)
print(format_ranking_report(results))
```
