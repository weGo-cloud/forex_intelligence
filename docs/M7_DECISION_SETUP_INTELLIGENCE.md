# M7 — Decision & Setup Intelligence Foundation

> **M7 produces structured, explainable action states from setups plus
> historical evidence. It does not predict returns, assign confidence
> scores, place orders, or size positions.**

## Purpose

M6 answers "how robust is the historical evidence for this strategy
under this context?" M7 is the bridge from that evidence to the
*current* market: what setup is present, what evidence supports it,
what is still missing, and what action state the evidence justifies.

```
DATA (M1)
  -> MARKET CONTEXT (M4 RegimeContext)
  -> OBSERVATIONS      app/observations
  -> SETUP             app/setups
  -> HISTORICAL EVIDENCE (M6, consumed via app.decision.evidence)
  -> DECISION STATE    app/decision
  -> EXPLANATION       Decision (structured; prose belongs to a later UI/LLM layer)
```

## Audit (done before coding)

| Concept from the mission | Already existed? | What M7 did |
|---|---|---|
| Swing highs/lows | Yes — `detect_swing_points` (confirmation-lag aware) | Reused |
| FVG | Yes — `detect_fvg` | Reused |
| Higher-high/lower-low labeling | Yes — `label_swings` | Reused as the basis for BOS/CHoCH |
| Trend / volatility / structure context | Yes — M4 `RegimeContext` | Reused as-is, packaged into observations |
| Liquidity sweep | Only inline inside the `liquidity_sweep_fvg` strategy's control flow | Extracted the same technique into a standalone detector (strategy left untouched) |
| Support/resistance | No | New — clusters confirmed swings |
| BOS / CHoCH | No | New — derived from `label_swings` |
| Liquidity pool | No formal concept | Not implemented (see limitations) |
| Setup / state machine / decision state | No | New |
| Historical evidence | Yes — M6 `StrategyContextEvaluation` | Bridged, never recomputed |
| Probability math | Yes — M5 | Not touched, not duplicated |

Nothing in M1–M6 was modified except: `app/main.py` (router registration).

## Architecture

```
app/observations/
    models.py      ObservationType, ObservationSource, Observation, observations_as_of
    detectors.py    swing / fvg / structure (BOS,CHoCH) / support_resistance /
                    liquidity_sweep / regime observations
app/setups/
    models.py       SetupType, SetupStatus, Level, InvalidationCondition, Setup
    state_machine.py  deterministic transitions (the ONLY place status changes)
    detection.py      SetupDetector (two concrete patterns)
app/decision/
    models.py       DecisionState, M6Evidence, Decision
    evidence.py      evidence_from_evaluation  (M6 -> M6Evidence, no recomputation)
    engine.py         DecisionEngine (deterministic, table-driven)
    service.py         detect_setups / evaluate_decisions (domain-level services)
    api_schemas.py     request schemas
app/api/setups.py     thin HTTP wrapper: POST /setups/detect, POST /setups/evaluate
```

Detection, evaluation, historical evidence, and presentation are separate
calls (mission section 19): observations are computed once and passed
in; evidence is computed by M6 and passed in; the decision engine only
combines them. Nothing recomputes history inside a request handler.

## Observations

`Observation` carries type, direction, price/zone, timeframe, confidence,
`source`, timestamp, and metadata. `confidence` is a `[0,1]` descriptor
of how well-evidenced the *observation* is (e.g. touch count) — it is
never a trading-confidence score and is not propagated into decisions.

Implemented from real data: `SWING_HIGH`, `SWING_LOW`, `FVG`, `BOS`,
`CHOCH`, `SUPPORT`, `RESISTANCE`, `LIQUIDITY_SWEEP`, `TREND_ALIGNMENT`,
`VOLATILITY_STATE`, `PRICE_LOCATION`. `LIQUIDITY_POOL` exists as an enum
member but has no detector — the mission says not to fabricate
detections, and no reliable definition exists in the current data layer.

**BOS / CHoCH definition** (`structure_observations`): the first labeled
swing establishes a starting bias. A later swing continuing that bias
(e.g. higher-high while bullish) is BOS; one contradicting it (e.g.
lower-low while bullish) is CHoCH and flips the tracked bias.

**Support/resistance** (`support_resistance_observations`): confirmed
swing prices within `tolerance_pct` form a cluster; a cluster reaching
`min_touches` becomes a level, reported once, at formation.

## Setups and the state machine

`Setup` fields follow the mission: id, instrument, timeframe, direction,
type, observations, levels, confirmations, invalidation conditions,
status, metadata (plus `evidence` as an optional attachment point that
`app.decision` populates — `Setup` itself never computes it).

Setup ids are deterministic (`<type>:<instrument>:<timeframe>:<trigger
timestamp>`).

States: `DETECTED → WATCH → CONFIRMING → VALID → ENTRY_CONDITION_MET`,
strictly forward one step at a time; from any non-terminal state,
`INVALIDATED` or `EXPIRED` is allowed; both are terminal. Every change
goes through `state_machine.transition(setup, target, at, reason)`,
which returns a new frozen `Setup` or raises `InvalidTransitionError`.
`at` is always caller-supplied — never `datetime.now()`.

**Detector scope**: the two implemented patterns
(`LIQUIDITY_SWEEP_REVERSAL` = sweep + later same-direction BOS;
`SUPPORT_RESISTANCE_REACTION` = level + later same-direction BOS/CHoCH)
emit only `DETECTED` (no confirmation yet) or `VALID` (confirmation
found). `WATCH`, `CONFIRMING`, `ENTRY_CONDITION_MET`, `INVALIDATED`,
`EXPIRED` are fully supported and tested in the state machine, for
incremental/live re-evaluation — a single pass over already-complete
historical observations has no natural "still forming" state.

## Decision engine

Priority order (first match wins) — see `engine.py`'s docstring:

1. setup `INVALIDATED` → `INVALIDATED`
2. setup `EXPIRED` → `WAIT` (the mission's decision states have no
   "expired"; nothing actionable = wait)
3. no evidence, or `EvidenceTier.INSUFFICIENT` (or `LIMITED` under
   `require_adequate_evidence=True`) → `INSUFFICIENT_EVIDENCE`
4. setup `ENTRY_CONDITION_MET` → `ENTRY_CONDITION_MET`
5. setup `VALID` → `VALID_SETUP`
6. setup `WATCH`/`CONFIRMING` → `WATCH`
7. setup `DETECTED` → `WAIT`

Insufficient evidence is checked *before* structural status: a
structurally valid setup with no supporting sample is not a valid setup.
Invalidation is checked first: it is a fact about the setup, not an
evidence question. Evidence direction (e.g. negative historical expected
R) is reported in `reasons`, not used to override the structural state —
M7 exposes evidence; it does not adjudicate it (ranking/decisioning on
evidence quality is M8+).

`Decision` = state, setup_id, summary, reasons, supporting_observations,
supporting_evidence, missing_conditions, invalidation_conditions,
warnings, timestamp. All structured; reasons come from fixed templates
keyed to the branch taken. No prose generation, no LLM, no numeric
confidence.

## M6 evidence consumption

`evidence_from_evaluation(StrategyContextEvaluation) -> M6Evidence` is the
only constructor. Every field is read off the existing M6 object
(sample counts, tier, `p_win`, Wilson interval, expected R, walk-forward
stability, warnings). `regime_filter` is the conditioning filter the
evaluation used — M6 records what was filtered on, not a captured
`RegimeContext`, so that is what is exposed. `M6Evidence.from_dict` /
`to_dict` allow a serialized snapshot to be passed over the API without
re-running M6.

## Temporal-causality audit

Where look-ahead could enter, and how each is closed:

| Risk | Control | Test |
|---|---|---|
| Swings known before confirmation | Observation `timestamp` = `confirmed_at`, inherited from `detect_swing_points` | `test_causality.py::test_swings`, `observations/test_detectors.py::test_timestamp_is_confirmed_at_not_pivot_time` |
| FVG, BOS/CHoCH | Same primitives, same lag | `::test_fvg`, `::test_structure_bos_choch`, `::test_appended_shock_does_not_change_earlier_observations` |
| Support/resistance levels shifting as data arrives | Incremental, confirmation-ordered clustering; SUPPORT/RESISTANCE decided against close at the confirming swing | `::test_support_resistance` |
| Confirmation preceding its trigger | Confirmations must have `timestamp` strictly after the trigger | `setups/test_detection.py::test_confirmation_must_be_after_trigger_not_before` |
| Setup status using later observations | `observations_as_of(T)` / `detect_setups(as_of=T)` | `::TestSetupStateIsAsOfSafe`, `test_service_and_api.py::test_detect_as_of_excludes_later_confirmation` |
| Decision at T using later setup state | Decision built from as-of setup | `::TestDecisionAsOfSafe` |
| Historical evidence | M6/M5 `resolved_before` (unchanged) | M5/M6 suites |

**A real defect was found and fixed by this audit**:
`support_resistance_observations` originally clustered all swings
globally, so appending later data shifted an existing level's timestamp
and price — the observation stream wasn't prefix-stable. The
truncation test caught it; the detector now processes swings
incrementally in confirmation order.

## Screenshot compatibility (future M9)

A screenshot analyzer must only emit `Observation`s with
`source=ObservationSource.SCREENSHOT` — via Python
(`Observation(...)` / `Observation.from_dict`) or `POST /setups/evaluate`.
Routing in `app.decision.service` is by `observation_type`, never by
`source`, so a screenshot-derived sweep and a market-data-derived sweep
take identical paths (`test_source_does_not_change_setup_or_decision`,
`test_screenshot_source_accepted_over_api`). There is no second
detection or decision system for screenshots to bypass or duplicate.
Screenshot-derived observations get the same validation
(timezone-aware timestamps, `confidence` in `[0,1]`, known types only).

## Minimal interface

The repo has no frontend framework, so the "minimal developer/research
interface" is the JSON API: `POST /setups/detect` returns setups;
`POST /setups/evaluate` returns each setup with its `Decision` (state,
reasons, missing conditions, invalidation conditions, warnings, and the
evidence used) — the full reasoning trace, machine-readable. No
dashboard was built.

## Known limitations / deferred

- `LIQUIDITY_POOL` has no detector (no reliable definition yet).
- Support/resistance levels are reported at formation only; later
  touches aren't re-emitted, so `touch_count` reflects the count at
  formation.
- Setup `expires_at` exists on the model but no detector sets it and
  nothing auto-expires setups; expiry is applied via `transition`.
  Time-based expiry policy is deferred.
- Invalidation conditions are represented as structured data
  (`price`, `direction`, `description`) but nothing yet *evaluates*
  them against incoming candles to trigger `INVALIDATED` — a live
  monitor is M8+.
- `ENTRY_CONDITION_MET` is reachable only via `transition`; no detector
  defines entry conditions yet.
- Only two setup patterns are implemented (by design).
- The decision engine reports evidence but does not weigh it (e.g. a
  `VALID` setup with negative historical expected R is still
  `VALID_SETUP`, with the negative expectancy in `reasons`). Weighing
  evidence quality is left to M8.
- Not implemented: screenshot analysis, LLM explanation, execution,
  position sizing, UI.
