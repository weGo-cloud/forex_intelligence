# M3 Development Notes — Research Experiment & Robustness Foundation

Records M3-specific architecture decisions, deviations from the spec
(with reasoning), and known limitations, following the same convention
as `docs/M1_DATA_FOUNDATION.md` / `docs/M2_BACKTESTING_ENGINE.md`.

## Architecture overview

```
M2 Strategy + BacktestEngine (unchanged)
   -> app/research/models.py       ExperimentDefinition, ExperimentType, ExperimentStatus, hashing
   -> app/research/registry.py     ExperimentRecord (SQLAlchemy), ExperimentRegistry (repository)
   -> app/research/experiment.py   run_single_backtest, execute_experiment (the one place M3 calls M2)
   -> app/research/sweep.py        run_parameter_sweep (strategy-agnostic parameter grid)
   -> app/research/robustness.py   run_robustness_analysis (parameter-neighborhood, built on sweep)
   -> app/research/runner.py       run_walk_forward_experiment (built on validation.walk_forward_windows)
   -> app/research/comparison.py   compare_experiments
   -> app/research/api_schemas.py  Pydantic request/response schemas
   -> app/api/research.py          HTTP endpoints
```

`app/research/experiment.py` is the only module that imports
`app.backtesting`/`app.strategies` to actually run something — sweep,
robustness, and walk-forward are all built by calling `execute_experiment`
(or its lower-level `run_single_backtest`) repeatedly with different
`ExperimentDefinition`s, never by re-implementing execution. This is
what keeps provenance (Experiment -> Configuration -> Data selection ->
Strategy -> Execution engine -> Metrics) true by construction rather
than by convention someone has to remember to follow.

Nothing in `app/research` references ORB, RSI divergence, or any other
concrete strategy by name — `strategy_name`/`strategy_params` are always
opaque values passed through to `app.strategies.registry.get_strategy`.

## Experiment lifecycle

```
PENDING --create()--> RUNNING --mark_running()--> COMPLETED --mark_completed(summary)
                                                \-> FAILED    --mark_failed(reason)
```

`execute_experiment` drives this for a single backtest: create (PENDING)
-> mark_running -> run -> mark_completed or mark_failed. Once COMPLETED
**or** FAILED, `ExperimentRegistry._require_mutable` refuses any further
status/result mutation (`ExperimentImmutableError`) — re-running the
same configuration is a new experiment (`registry.create()` again),
optionally linked via `parent_experiment_id`.

## Experiment schema (`ExperimentDefinition` / `experiments` table)

Every field listed in spec section 4 is present on `ExperimentDefinition`
(`app/research/models.py`) except explicit Git-commit capture, which was
intentionally **not** added — see "Deviations" below. The persisted
`experiments` row (`app/research/registry.py`) stores:

- identity: `experiment_id` (sequential `exp_NNNNNN`, human-referenceable)
- `experiment_type`, `status`
- `strategy_name`, `instrument_symbol` (indexed, for fast filtering)
- `config` (the full canonical dict) + `config_hash`
- `experiment_name`, `tags` (free-form labels — excluded from the hash)
- `parent_experiment_id` (self-referential FK)
- `result_summary` (metrics only — see "No duplicated datasets" below)
- `failure_reason`
- `created_at` / `updated_at` / `completed_at`

## Configuration hashing / reproducibility

`ExperimentDefinition.canonical_dict()` produces a JSON-safe dict with a
fixed field set (no arbitrary extra keys possible). `canonical_json()`
serializes it with `sort_keys=True`, and `config_hash()` is the SHA-256
hex digest of that JSON. `_normalize()` additionally deep-sorts nested
dict keys and stringifies non-JSON-native values (e.g. a stray enum
in `strategy_params`), so hashing never raises on an odd parameter type.

**Deliberately excluded from the hash:** `created_at`, `experiment_name`,
`tags`, `parent_experiment_id`. These are labels/provenance about the
*run*, not the *configuration* — two runs of an identical configuration,
submitted at different times or under different names, must hash
identically, because `config_hash` answers "is this the same
configuration", not "is this the same database row". This is verified
by `tests/research/test_models.py::TestDeterministicSerialization`.

**No Git-commit capture.** Section 7 makes this explicitly optional
("if Git commit information can be safely captured, make it optional
metadata rather than a hard dependency") — it was left out of V1 rather
than added as a half-working stub, since the execution environment here
has no guarantee of being a Git checkout. Adding it later as an optional
field on `ExperimentDefinition` would not change `config_hash()`'s
existing values (it would need to join the exclusion list above, same
as `tags`).

## Parameter sweep (`app/research/sweep.py`)

`_combinations()` takes `dict[str, list[value]]`, sorts grid keys, does
`itertools.product` over the (caller-ordered) value lists, and drops
exact-duplicate combinations. This is the only ordering guarantee
needed for determinism (spec: "use deterministic ordering") — it does
not sort by value, since values aren't guaranteed orderable across all
possible parameter types.

Each combination becomes an independent child `ExperimentDefinition`
(`base_definition.with_overrides(...)`, never a mutation — definitions
are frozen dataclasses) executed via `execute_experiment`. Nothing about
one child's execution depends on another's, which is exactly what would
let a future executor run them concurrently (spec: "design the interface
so parallel execution can be added later without changing experiment
semantics") without any change to this module's public contract.

## Walk-forward (`app/research/runner.py`)

Reuses `app.backtesting.validation.walk_forward_windows` unchanged (pure
date arithmetic — no duplication) to carve `base_definition`'s overall
test period into `(train, test)` windows, then creates one child
`SINGLE_BACKTEST` `ExperimentDefinition` per window whose
`test_period_start/end` is the **window's** test range (never the
overall range). Train boundaries are preserved as tags/summary metadata
per window, satisfying spec section 9's requirement to preserve them,
even though — matching `validation.py`'s existing scope — no automatic
parameter optimization runs on the train window. That remains explicitly
out of scope until there's an actual consumer for it (see "Deferred"
below); what matters for now is that the train/test boundary exists
structurally and no test-window result can leak into parameter
selection, because there is no parameter selection step yet at all.

## Robustness analysis (`app/research/robustness.py`)

A robustness analysis over parameter `p` with values `[v1, v2, ...]` is,
mechanically, a sweep over the single-key grid `{p: [v1, v2, ...]}` — the
values are always caller-supplied, never hard-coded.

**No composite "robustness score"** — spec section 10 explicitly scopes
that out absent a documented mathematical justification, and none
exists yet. Instead, `classify_stability()` produces one of four plain,
fully-documented labels from the per-neighbor `total_return_pct`:

| Label | Condition |
|---|---|
| `insufficient_data` | any neighbor has < `MIN_TRADES_FOR_STABILITY_JUDGEMENT` (10) trades, or failed to run |
| `isolated` | returns don't share a majority sign across the neighborhood |
| `sensitive` | same-signed, but `stdev(returns) > mean(abs(returns))` |
| `stable` | same-signed, with `stdev(returns) <= mean(abs(returns))` |

This is intentionally simple and legible rather than clever — a human
reading `notes` can verify the classification by hand from the raw
per-neighbor numbers, which a hidden composite score would not allow.

## Comparison (`app/research/comparison.py`)

`compare_experiments()` returns rows in exactly the order of the
`experiment_ids` the caller passed — it never re-sorts by a metric,
since that would itself be a ranking decision (spec section 11: ranking
is the future ranking engine's job, not comparison's). Missing/undefined
metrics (a FAILED experiment, or a metric M2's `compute_metrics` left
`None` for insufficient trades) pass through as `None`, never coerced to
`0.0`. Comparing experiments on different instruments or timeframes
raises `IncompatibleExperimentsError` rather than silently producing a
misleading side-by-side table.

## Provenance & "no duplicated datasets"

Every persisted `ExperimentRecord` traces to exactly one
`ExperimentDefinition` (via `config`/`config_hash`), which names a
strategy + instrument + timeframe + date range but never stores OHLCV
data itself — `run_single_backtest` calls M1's `load_ohlcv`
(`app.backtesting.data_access`) fresh each time, exactly as M2 does.
`result_summary` stores metrics only (`app/research/experiment.py:
result_summary()`), explicitly excluding the trade log, per spec section
6 ("do not store large duplicated datasets") — the full `BacktestResult`
(including trades) remains available on `ExperimentOutcome.result` to
whatever code just ran the experiment, but is not written to the
registry a second time.

**Dataset fingerprinting beyond symbol+timeframe+date-range does not
exist yet** (e.g. a hash of the actual candle data used) — M1's data
layer has no such fingerprint to draw on, and inventing one here would
be exactly the kind of unearned certainty section 13 warns against
("if the current data layer cannot guarantee a stronger form of dataset
fingerprinting yet, document the limitation rather than inventing one").
If M1 candles for a given range are later corrected/backfilled, a
historical experiment's `config_hash` stays the same even though
re-running it could produce different results — this is a known,
documented limitation, not a bug.

## API (`app/api/research.py`)

```
POST   /experiments              create + immediately run a SINGLE_BACKTEST experiment
GET    /experiments               list (filterable by strategy/instrument/type/status/parent)
GET    /experiments/{id}          fetch one
GET    /experiments/{id}/results  status + result_summary + failure_reason
GET    /experiments/{id}/children direct children only, in creation order
GET    /experiments/compare       ?id=exp_000001&id=exp_000002&... (repeatable query param)
```

Only single-backtest creation is exposed via POST — sweep/walk-forward/
robustness are orchestration functions (`app.research.sweep`/`runner`/
`robustness`) callable from a script or a future endpoint, not wired to
HTTP yet, matching spec section 14's "minimal API layer" framing (the
dashboard/frontend that would actually need to trigger these from a UI
is explicitly out of scope for M3).

## Regression fixes (pre-existing failures, unrelated to M3)

Both were fixed before M3 work began; see the diffs in
`tests/data/test_models.py` and `tests/data/test_api_market_data.py`.

1. **`test_naive_timestamp_is_rejected_before_reaching_the_database`** —
   stale test: `UTCDateTime`'s `ValueError` is raised inside a
   SQLAlchemy type-decorator process step, so it reaches the caller
   wrapped in `sqlalchemy.exc.StatementError`, not as a bare
   `ValueError`. The behavior (naive timestamps are rejected) was
   always correct; only the test's assertion was checking one layer
   too deep. Fixed to assert on `StatementError` and its `.orig` cause.
2. **API market-data tests (`test_api_market_data.py`)** — genuine
   fixture bug: `TestClient` dispatches requests on a separate thread
   from the one that builds the test engine and inserts fixture rows.
   A bare `sqlite:///:memory:` engine opens a *new, empty* in-memory
   database per connection/thread, so the request thread saw none of
   the seeded data. Fixed by sharing a single connection across threads
   via `StaticPool` + `connect_args={"check_same_thread": False}` —
   the same fix `tests/research/test_api.py`'s fixture uses from the
   start.

## Known limitations

- No automatic parameter optimization on walk-forward train windows
  (deliberately deferred — see "Walk-forward" above).
- No dataset fingerprinting beyond symbol/timeframe/date-range
  (deliberately deferred — see "Provenance" above).
- No Git-commit capture (deliberately deferred — see "Hashing" above).
- Synchronous execution only for sweep/walk-forward/robustness — each
  child experiment runs one at a time in-process. The per-child
  independence documented above means parallelizing this later is an
  executor change, not a semantics change.
- POST /experiments only creates SINGLE_BACKTEST experiments; sweep/
  walk-forward/robustness have no HTTP endpoint yet (see "API" above).

## Future extension points

- A parallel/async executor for `run_parameter_sweep` /
  `run_walk_forward_experiment`, swapping the synchronous `for` loop
  for a task queue without changing either function's public signature
  or the child-experiment semantics.
- HTTP endpoints for sweep/walk-forward/robustness once a caller (CLI
  or the future dashboard) actually needs to trigger them remotely.
- The future ranking engine (M4+) building on `compare_experiments`'
  `ComparisonRow`s and `robustness`'s stability labels, without needing
  any change to this package — ranking logic stays entirely outside
  `app.research`.
- Regime/probability layers (M4+) can create experiments the same way
  any other caller does: build an `ExperimentDefinition`, call
  `execute_experiment` — no coupling to add on either side.
