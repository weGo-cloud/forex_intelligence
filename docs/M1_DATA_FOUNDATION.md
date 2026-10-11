# M1 Development Notes — Market Data Foundation

This file records the M1-specific architecture decisions, data model,
validation rules, ingestion workflow, and known limitations, per the
same "record every ambiguity resolution" rule used for M0
(see `docs/DEVELOPMENT_NOTES.md`).

## Architecture overview

```
provider (vendor-specific)
   -> app/data/providers/base.py     ProviderCandle (raw, loosely typed)
   -> app/data/normalization.py      -> CanonicalCandle (typed, UTC, Decimal)
   -> app/data/validation.py         structural checks (reject bad candles)
   -> app/data/repository.py         idempotent persistence + query
   -> app/data/ingestion.py          orchestrates all of the above,
                                      records provenance (IngestionRun)
```

Each stage is independently testable and importable on its own — the
ingestion pipeline is the only module that wires them together.

## Data model decisions

- **Instrument** identity is its `symbol` (unique, indexed). The DB
  primary key (`id`) is what foreign keys reference; `symbol` is the
  natural business key everything else (providers, API callers) uses.
- **Candle** uniqueness is enforced by a DB-level `UniqueConstraint` on
  `(instrument_id, timeframe, timestamp)`, matching the spec's
  "uniquely identifiable by instrument + timeframe + timestamp"
  requirement. A composite index on the same columns backs the common
  query pattern (candles for one instrument/timeframe over a range).
- **`volume` is nullable.** Per the spec's forex considerations, spot
  forex has no centralized-exchange volume; this column stores
  whatever (if anything) the selected provider supplies, and the
  system makes no claim about what it represents.
- **Prices are `Numeric(18, 8)`, not `Float`.** Floating-point error is
  not acceptable for stored market data that future backtesting will
  depend on. Application code represents prices as `Decimal`
  throughout (`CanonicalCandle`).
- **`asset_class` is a plain string column, not a DB enum type.** Only
  `forex` is used in M1, but a future milestone adding another asset
  class should not require a migration to alter a fixed enum type.
- **Ingestion provenance (`IngestionRun`) stores its issue list as a
  single capped JSON column**, not a normalized child table. At M1
  scale this is simpler to query and sufficient for investigation, and
  avoids designing a child-table schema that would need to anticipate
  every future quality-check type. The list is capped at 200 stored
  rejection entries per run (counts themselves are exact and
  uncapped) so a pathologically bad provider response can't grow a
  single row without bound.

## Timezone / UTC handling

- All canonical timestamps are UTC and timezone-aware; naive
  datetimes are rejected outright (in both normalization and at the
  ORM level via the `UTCDateTime` type), never silently assumed to be
  UTC.
- `UTCDateTime` (`app/data/models.py`) is a `TypeDecorator` that
  normalizes to UTC before storage and re-attaches UTC tzinfo on load,
  regardless of backend. This was necessary because PostgreSQL's
  `TIMESTAMPTZ` preserves tzinfo across a round trip but SQLite (used
  in the test suite) does not — without this, the exact same
  application code would behave differently in tests vs. production.
- No session/exchange timezone offsets (New York, London, Asian
  sessions) are hard-coded anywhere in this milestone, per the spec.
  Timeframe "nominal duration" (`Timeframe.minutes`) is used only as a
  heuristic for gap-detection thresholds and aggregation bucket sizing
  — it is not a claim about trading-session boundaries.

## Validation rules

`validate_candle` (`app/data/validation.py`) rejects a candle if:
- any of open/high/low/close is not strictly positive
- volume is present and negative
- `high < low`
- `high` does not cover both `open` and `close`
- `low` does not cover both `open` and `close`

A rejected candle is recorded (reason + timestamp) but never modified
or coerced into validity — per the spec's "do not silently modify bad
market data" rule.

## Duplicate detection

Two layers, both idempotent-safe:
1. **In-batch**: `find_duplicates_in_batch` dedupes an incoming batch
   by `(symbol, timeframe, timestamp)` before persistence is attempted.
2. **Against stored data**: `CandleRepository.bulk_insert` queries
   existing timestamps for the instrument/timeframe before inserting,
   so re-running the same historical range is a no-op for already-
   stored candles.

The DB-level `UniqueConstraint` remains as a safety net against races
between concurrent ingestion runs; it is not the primary duplicate-
prevention mechanism because relying on database-specific
`ON CONFLICT` syntax would have tied persistence code to PostgreSQL
and made it impossible to test against SQLite.

## Gap detection

`detect_gaps` is a **reporting** mechanism only. It flags any interval
between consecutive candles that exceeds 1.5x the timeframe's nominal
duration and returns a `GapIssue` per flagged interval. It:
- never rejects the candles on either side of a gap
- never manufactures a candle to fill a gap
- does not distinguish "legitimate market closure" from "missing data"
  — that distinction requires session-calendar knowledge (which days/
  hours a given instrument actually trades) that this milestone
  deliberately does not implement, per the "do not hard-code session
  timezone offsets" instruction. A future milestone that needs this
  distinction should add a session calendar as its own component
  rather than baking assumptions into the market-data layer.

## Provider abstraction

`MarketDataProvider` (`app/data/providers/base.py`) takes
`(symbol, timeframe, start, end)` and returns an iterable of
`ProviderCandle` — loosely-typed raw values, since vendors disagree on
timestamp/price representation. All vendor-specific parsing happens in
`app/data/normalization.py`, which is the only module that needs to
change to support a new provider's raw format.

`MockMarketDataProvider` (`app/data/providers/mock.py`) supports two
modes: a deterministic generated sequence (for "just give me plausible
data" tests) and a fixed list of exact `ProviderCandle` objects
(for tests that need to exercise specific — including malformed —
inputs). No commercial vendor is integrated in M1.

## Ingestion pipeline

`run_ingestion` (`app/data/ingestion.py`):
`provider -> normalize -> validate -> sort -> dedupe -> persist -> detect gaps -> record provenance`.

Key invariant: `received == accepted + rejected + duplicates`, where
`rejected` counts distinct rejected **records** (a record failing
multiple checks at once is still one rejected record; all of its
issues are still kept in the provenance detail list).

Status is derived as:
- `success` — no request made (`received == 0`), or every record
  usable (`rejected == 0`, regardless of duplicates — an idempotent
  rerun where everything was already stored is a successful no-op, not
  a failure)
- `failed` — every single received record was rejected
- `partial` — some but not all records were rejected

Out-of-order input is sorted before persistence and logged; it is not
treated as an error, since providers are not guaranteed to return data
in timestamp order.

## Timeframe aggregation

`aggregate_candles` (`app/data/aggregation.py`) buckets are aligned to
UTC epoch boundaries (not to the start of the input range), so
aggregating minutes 3–9 into 5-minute candles only emits the bucket
that minutes 5–9 complete — the incomplete 0–4 bucket is silently
dropped rather than emitted as a misleading partial candle. This is
in-memory only; it does not persist aggregated candles or get invoked
automatically by ingestion. A future milestone can decide whether/how
to materialize aggregated data.

## API surface

`app/api/market_data.py` adds two read-only endpoints
(`GET /market-data/instruments`, `GET /market-data/candles`).
Deliberately no ingestion-trigger endpoint: kicking off a provider
fetch is treated as an application/operational concern (a script or a
future scheduled job), not something an HTTP client should be able to
trigger directly.

## Database / migrations

M0 deferred Alembic because no schema existed yet to migrate. M1
introduces the first real schema, so Alembic is added now
(`alembic.ini`, `migrations/env.py`, `migrations/versions/0001_*.py`).
`migrations/env.py` sources the DB URL from
`app.core.config.get_settings()` rather than duplicating it in
`alembic.ini`, keeping one source of truth for `DATABASE_URL`.

No strategy/backtesting/ML tables are added — only the market-data
schema required for M1 (`instruments`, `candles`, `ingestion_runs`).

## Testing

`tests/data/` runs the entire suite against an **in-memory SQLite**
database rather than PostgreSQL, so it runs anywhere without a live DB
service. This was possible specifically because the repository layer
avoids PostgreSQL-specific SQL (see "Duplicate detection" above) and
because `UTCDateTime` normalizes timezone behavior across backends.
Coverage includes: timeframes, normalization (multiple timestamp
formats, malformed input), validation (each rejection rule, batch
dedup, gap detection incl. a simulated weekend-style gap), the mock
provider (deterministic + fixed modes), the repository (idempotent
insert, partial-overlap insert, date-range query, ordering, UTC
round-trip), ORM-level constraints (unique constraint, naive-timestamp
rejection), the full ingestion pipeline (happy path, idempotent rerun,
malformed-record handling, out-of-order sorting, gap reporting,
provenance recording, unknown-instrument error), and aggregation
(happy path, incomplete-bucket dropping, epoch-aligned bucketing,
mismatched-timeframe rejection).

**Known limitation:** this test suite has not been executed in the
environment M1 was developed in — see the final implementation report
for why, and treat these tests as reviewed-but-unverified until run
with `pytest` in an environment with the project's real dependencies
installed.

## Assumptions / unresolved items for future milestones

- No session/trading-calendar component exists yet, so gap detection
  cannot yet distinguish "expected weekend closure" from "genuinely
  missing data" — it reports both identically, by design (see "Gap
  detection" above).
- Aggregated candles are computed on demand, not persisted. A future
  milestone should decide whether higher timeframes get materialized
  during ingestion or computed lazily from stored lower-timeframe data.
- `IngestionRun.issues` is a capped JSON list, not a queryable
  normalized table. If a future milestone needs to query "all gaps for
  instrument X across all runs" efficiently, this may need to become
  its own table at that point.
- No real commercial market-data vendor is integrated — only the mock
  provider. Adding one is isolated to a new `app/data/providers/`
  module plus provider-specific normalization; it should not require
  changes to the ingestion pipeline, validation, or repository layers.
