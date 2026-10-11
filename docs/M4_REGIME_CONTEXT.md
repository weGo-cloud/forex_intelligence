# M4 — Market Regime & Context Engine

> **The M4 regime engine describes historical/current market
> conditions. It is not a predictive model and does not produce trade
> recommendations.**

## Purpose

M2 strategies generate trade signals with no awareness of broader
market conditions. M3 lets you rigorously evaluate how a strategy
performed. Neither layer can answer "was this strategy firing in a
trending market or a ranging one?" - that's what M4 adds: a
deterministic, causal description of trend/volatility/structure
conditions at any bar, usable by future research, strategy-selection,
and risk layers without being coupled to any one of them.

```
Market Data (M1)
    -> Features (app.features: indicators, trend, volatility, structure)
    -> Market Regime / Context (app.regimes)
    -> Strategy Signal (M2, unchanged - does not depend on regimes)
    -> Backtest / Research (M3, unchanged - can consume regimes, optionally)
```

## Architecture

```
app/features/
    indicators.py   (M2/M3, unchanged: rsi, adx, atr, bollinger_bands,
                      stochastic_oscillator, detect_swing_points, detect_fvg)
    sessions.py      (M2, unchanged: SessionWindow)
    trend.py         (M4, new) moving_average, ma_alignment, ma_slope, directional_return
    volatility.py    (M4, new) normalized_atr, rolling_return_volatility,
                                volatility_ratio, rolling_percentile_rank
    structure.py     (M4, new) label_swings, structural_trend_strength_series,
                                range_compression_ratio

app/regimes/
    models.py        TrendState, VolatilityState, StructureState (enums)
    config.py         RegimeClassifierConfig (every threshold, one place)
    features.py        RegimeFeatureSet + compute_feature_set (vectorized, whole-series)
    detectors.py       classify_trend / classify_volatility / classify_structure (pure functions)
    classifier.py       RegimeClassifier (orchestrator) + session labeling
    context.py          RegimeContext (the output value object) + to_dict()
```

`app/features/*` is pure MEASUREMENT (numbers in, numbers out, no
thresholds, no enums). `app/regimes/*` is INTERPRETATION (numbers in,
`TrendState`/`VolatilityState`/`StructureState` out). This mirrors the
mission's explicit layering:

```
FEATURES = measurements
REGIME ENGINE = interpretation of measurements
STRATEGY ENGINE = trade signal generation   (M2, untouched)
RESEARCH ENGINE = evaluation                (M3, untouched)
```

No M2 strategy (ORB, RSI Divergence, Mean Reversion, Liquidity
Sweep/FVG) imports anything from `app.regimes`, and nothing in
`app.regimes` imports from `app.strategies` or `app.research` - the
dependency only ever flows regimes -> features, features -> M1 data
shapes. A future decision layer combining `Signal + RegimeContext` is
free to depend on both without either depending on it.

## Naming deviation: `RegimeContext`, not `MarketContext`

The mission's own pseudocode calls the output object "MarketContext".
It is named `RegimeContext` instead, because `app.strategies.base`
already defines a class called `MarketContext` with a completely
different shape and purpose (a strategy's *input*:
`execution_df` + HTF context frames - not a description of market
conditions). Reusing the same name for two unrelated shapes in one
codebase is a real, avoidable cost (ambiguous imports, "which
MarketContext?" in code review, accidental wrong-type usage) for a name
that isn't load-bearing anywhere in the mission. `RegimeContext`
describes exactly what the object is and cannot be confused with the
existing class. See `app/regimes/context.py`'s module docstring for
the same note in code.

## Feature definitions

Every feature function's own docstring documents its **inputs, output,
warm-up period, NaN behavior, and causal behavior** individually (per
the mission's explicit per-feature documentation requirement) - this
section is a summary, not a replacement for reading them.

**Trend** (`app/features/trend.py`):
| Feature | What it measures |
|---|---|
| `moving_average` | Simple moving average |
| `ma_alignment` | +1/-1/0: fast MA above/below/equal to slow MA |
| `ma_slope` | Fractional change in an MA over N bars |
| `directional_return` | Simple % return over N bars |

Trend strength/direction from the DMI system (`+DI`, `-DI`, `ADX`) is
**reused from `app.features.indicators.adx`** (M2/M3), not duplicated.

**Volatility** (`app/features/volatility.py`):
| Feature | What it measures |
|---|---|
| `normalized_atr` | ATR as a fraction of price (cross-instrument comparable) |
| `rolling_return_volatility` | Rolling stdev of bar-to-bar returns |
| `volatility_ratio` | Current normalized ATR vs. its own rolling mean |
| `rolling_percentile_rank` | Where a value ranks (0-100) within its own trailing window |

Raw ATR (`app.features.indicators.atr`, M3) is reused, not duplicated.

**Structure** (`app/features/structure.py`):
| Feature | What it measures |
|---|---|
| `label_swings` | Each swing labeled higher/lower vs. the prior swing of the same kind |
| `structural_trend_strength_series` | Per-bar `[-1, 1]` score from the last N labeled swings |
| `range_compression_ratio` | Current N-bar range vs. its own rolling mean |

Swing detection (`app.features.indicators.detect_swing_points`, M2) is
reused as-is, **confirmation lag and all** - see "Causality" below.

## Regime definitions

Three independent dimensions (`app/regimes/models.py`), each with an
explicit `UNKNOWN` member alongside its normal states - see the
module's own docstring for why `UNKNOWN` exists and why it is kept
distinct from `NEUTRAL`/etc. (in short: `UNKNOWN` = "not enough valid
data", the *measured* states = "we looked, and here's what we found").

- **`TrendState`**: `UPTREND` / `DOWNTREND` / `NEUTRAL` / `UNKNOWN`
- **`VolatilityState`**: `LOW` / `NORMAL` / `HIGH` / `UNKNOWN`
- **`StructureState`**: `TRENDING` / `RANGING` / `TRANSITION` / `UNKNOWN`

## Classifier logic (`app/regimes/detectors.py`)

**Trend** (`classify_trend`): three independent directional votes -
price vs. the slow MA, the slow MA's own slope sign, +DI vs. -DI - with
ADX used separately as a *strength gate* (ADX has no direction of its
own). `>= trend_min_agreeing_votes` (default 2 of 3) must agree **and**
`adx >= trend_adx_threshold`, or the result is `NEUTRAL`.

**Volatility** (`classify_volatility`): where the current normalized
ATR ranks (0-100 percentile) within its own trailing
`volatility_percentile_window`-bar history. `<= volatility_low_percentile`
→ `LOW`, `>= volatility_high_percentile` → `HIGH`, else `NORMAL`. This is
**relative to the instrument's own recent history**, never a fixed
absolute ATR value - per the mission's explicit requirement ("Raw ATR
alone should not determine whether volatility is 'high' or 'low' across
different instruments").

**Structure** (`classify_structure`): requires **both** a consistent
swing sequence (`|structural_trend_strength| >= structure_trend_strength_threshold`)
**and** range expansion for `TRENDING`; **both** a weak/absent swing
sequence **and** range compression for `RANGING`. Anything else (the
two signals disagree) is `TRANSITION` - genuinely mixed evidence, not
an arbitrary tie-break. `structural_trend_strength`'s *magnitude* (not
sign) drives this, keeping structure independent of trend *direction*
(a strong downtrend and a strong uptrend are both `TRENDING` here; which
direction is `TrendState`'s job).

## Default thresholds and their limitations

All defaults in `app/regimes/config.py` are **documented guesses, not
values tuned against this project's historical data** - per the
mission (sections 8/9): "Do not optimize these thresholds against
historical performance in M4... Future research milestones may
evaluate and optimize parameters using proper out-of-sample
methodology." `RegimeClassifierConfig` is a plain immutable dataclass
specifically so a future M3 parameter sweep / robustness analysis (both
already exist - see `docs/M3_RESEARCH_EXPERIMENTS.md`) can evaluate
alternative configurations mechanically, the same way it evaluates
strategy parameters, with zero source changes:

```python
RegimeClassifier(config=RegimeClassifierConfig(trend_adx_threshold=25.0))
```

## Warm-up behavior

Every feature has its own warm-up window (documented on the function
itself); the *effective* warm-up for a full `RegimeContext` is the
longest of all of them (dominated by `trend_slow_ma_period` (50) +
`volatility_percentile_window` (100) + the ADX ~2x-period warm-up, by
default). Bars before warm-up produce `UNKNOWN` on every dimension -
never a fabricated real state (`TestB` in `tests/regimes/test_bias.py`
checks this directly).

## Causality guarantees

- Every rolling/shift-based feature (`trend.py`, `volatility.py`, most
  of `structure.py`) depends only on bars `<= i` by construction
  (pandas `rolling`/`shift`/`pct_change` are all backward-looking).
- Swing-based structure features preserve `detect_swing_points`'s own
  confirmation lag exactly - a labeled swing's `available_at` is never
  earlier than its underlying swing's `confirmed_pos`
  (`tests/regimes/test_bias.py::TestC`).
- If higher-timeframe context is layered on top (see "Multi-timeframe
  readiness" below), the M2/M3 `align_higher_timeframe` utility's own
  "no incomplete HTF candle is ever visible early" guarantee is reused
  as-is, not reimplemented (`TestD`).
- Identical input + identical config → identical output, always
  (`TestE`) - no hidden state anywhere in `RegimeClassifier`,
  `RegimeClassifierConfig`, or the detector functions.
- Missing/NaN features → `UNKNOWN`, never an invented real state
  (`TestF`).

All six (Tests A-F, per the mission's own naming) live in
`tests/regimes/test_bias.py`.

**A real look-ahead-adjacent bug was caught by writing these tests**:
`structural_trend_strength_series`'s original implementation broke
(raised `ValueError: cannot reindex on an axis with duplicate labels`)
when two labeled swings became available on the same bar (a high and a
low both confirming on the same position) - fixed by keeping only the
most recent score for a duplicated position before the causal
forward-fill. Left as a permanent regression test
(`TestE_Determinism` exercises the same code path that originally
crashed).

## Multi-timeframe readiness

`RegimeClassifier` takes one OHLC frame and returns one `RegimeContext`
per bar of *that* frame - it does not assume "one symbol = one
timeframe = one context". For `execution timeframe = 15M, context
timeframe = 1H`:

1. Run `RegimeClassifier().classify_series(df_1h, symbol, "1H")` to get
   per-1H-bar contexts.
2. To align those onto a 15M execution index causally, reuse
   `app.backtesting.data_access.align_higher_timeframe` (M2/M3) exactly
   as strategies already do for HTF OHLC - M4 does **not** ship a
   second alignment mechanism (per the mission's explicit instruction).

## Session integration

`app.regimes.classifier.DEFAULT_SESSIONS` composes
`app.features.sessions.SessionWindow` (M2, unchanged) - no second
timezone system. `session_labels_at(ts, sessions)` returns a tuple of
every session name whose window covers `ts` (zero, one, or more than
one - sessions legitimately overlap). Pass `sessions={}` (or
`RegimeClassifier(sessions={})`) to get an explicit empty tuple on
every context rather than a guessed session. The four defaults
(Sydney/Tokyo/London/New York) are **approximate, commonly-used
hours**, not an authoritative source - override with a custom dict for
anything consequential.

## Research integration

No changes were made to `app.research`'s experiment registry, models,
or API - none were needed. `RegimeClassifier.classify_series(...)` is a
plain, stateless, reusable function call; a future M3-based research
workflow that wants regime-aware analysis calls it directly (e.g.
inside a script that also calls `execute_experiment`), the same way it
would call any other pure computation. M4 does not persist regime
context to the database - per the mission ("Do not persist large
regime datasets merely because persistence is possible"), the
classifier stays a pure, cheap-to-recompute function.

## Known limitations

- Thresholds are documented defaults, not empirically validated (see
  above) - do not treat a `HIGH`/`TRENDING` label as validated signal
  quality without running it through M3's research tools first.
- `DEFAULT_SESSIONS` hours are approximate and don't wrap midnight
  (consistent with `SessionWindow`'s own same-day model) - a session
  genuinely spanning midnight in its local zone needs a custom
  `SessionWindow`-based definition, which this module doesn't provide.
- `structural_trend_strength_series` only considers *labeled* swings
  (a swing needs a same-kind predecessor to get a label) - a series
  with no repeated same-kind swings (e.g. a single unbroken trend with
  no pullback swings) stays `UNKNOWN` on structure indefinitely, which
  is correct (there's genuinely no structural evidence yet) but worth
  knowing going in.
- `RegimeFeatureSet`/`classify_series` recompute every feature over the
  whole frame on every call - fine for research/backtest-scale data,
  not optimized for a tight per-tick live loop (out of scope for M4;
  `classify_at` exists precisely so a future live consumer can compute
  the feature set incrementally instead, without changing the
  classification logic itself).

## Example

```python
from app.regimes.classifier import RegimeClassifier
from app.regimes.config import RegimeClassifierConfig

classifier = RegimeClassifier(RegimeClassifierConfig(trend_adx_threshold=25.0))
contexts = classifier.classify_series(df, instrument_symbol="EURUSD", timeframe="5M")

latest = contexts[-1]
print(latest.trend_state, latest.volatility_state, latest.structure_state, latest.sessions)
# TrendState.UPTREND VolatilityState.NORMAL StructureState.TRENDING ('london',)

latest.to_dict()  # JSON-safe dict, ready for an API response or a log line
```

## Future integration points

- A decision/probability layer (M5+) combining `Signal` (M2) +
  `RegimeContext` (M4) - e.g. "only act on ORB signals when
  `structure_state == TRENDING`" - without either layer depending on
  the other today.
- Strategy ranking (future milestone) segmenting M3 experiment results
  by the regime that was active when each trade fired, by running
  `RegimeClassifier` alongside a backtest and joining on timestamp -
  no M3 changes required for this; it's a script-level composition.
- A risk engine (future milestone) reading `volatility_state` for
  position sizing - `RegimeContext.feature_values` already exposes the
  raw `normalized_atr` a risk model would need directly, without
  recomputing it.
