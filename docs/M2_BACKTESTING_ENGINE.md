# M2 Development Notes — Strategy + Backtesting Engine

This file records M2-specific architecture decisions, deviations from
the original spec (with reasoning), and known limitations, following
the same "record every ambiguity resolution" convention used for M1
(see `docs/M1_DATA_FOUNDATION.md`).

## Architecture overview

```
M1 CandleRepository (unchanged)
   -> app/backtesting/data_access.py     load_ohlcv, align_higher_timeframe
   -> app/strategies/base.py             MarketContext, Signal, Strategy protocol
   -> app/strategies/orb.py              ORBStrategy (implements Strategy)
   -> app/strategies/rsi_divergence.py   RSIDivergenceStrategy (implements Strategy)
   -> app/backtesting/engine.py          BacktestEngine (strategy-agnostic)
   -> app/backtesting/metrics.py         compute_metrics (strategy-agnostic)
   -> app/backtesting/runner.py          run_backtest() - one-call orchestration
```

`app/features/indicators.py` (RSI, swing-point detection) and
`app/features/sessions.py` (IANA-timezone session windows) are shared
primitives both strategies depend on but the engine does not.

Every module above `data_access.py` is pure-pandas/stdlib and
independently importable/testable without a database — `data_access.py`
itself defers its SQLAlchemy import to inside `load_ohlcv` (see that
module's docstring) so that even the M1 integration boundary doesn't
force a database dependency onto code that doesn't need one.

## Deviations from the original spec, and why

**No VectorBT in the default engine, but a VectorBT-backed alternative
exists.** `app/backtesting/engine.py` remains the default, dependency-free
engine — this development environment had no network access to install
or execute VectorBT, and the project's own principle ("a backtest that
produces impressive results but contains look-ahead bias is considered
a failed implementation") made an unverified execution backend too
risky to ship as the only option. `app/backtesting/vbt_engine.py` is a
second, optional implementation of the exact same interface
(`Signal[] in, Trade[] out`), install with the `vectorbt` extra
(`pip install -e .[vectorbt]`). Two things worth knowing before you
choose between them:

- Live current documentation (checked via web search, not execution)
  confirms VectorBT's own default same-bar SL+TP conflict resolution is
  "stop-loss-first (pessimistic)" and its documented fix for same-bar
  lookahead is "shift entries forward, fill at next bar's open" — both
  identical to what `engine.py` already does independently. That's
  real validation of the causal design regardless of which engine you
  run it on.
- `vbt_engine.py` has NOT been executed anywhere in this project (no
  network to `pip install vectorbt` in this sandbox) and has known,
  documented gaps — see its module docstring: position sizing is
  non-compounding (a real behavioral difference from `engine.py`),
  `exit_reason`/`stop_loss`/`take_profit`/`r_multiple` aren't
  reconstructable from vectorbt's trade records and are left as
  explicit placeholders/"unknown" rather than guessed. Run
  `tests/backtesting/test_engine_parity.py` the moment you have
  vectorbt installed — it runs identical signals through both engines
  and checks they agree on the fields that actually matter (timing,
  price, direction, net P&L). Until that test has actually passed,
  treat `engine.py`'s output as the trustworthy one; it has real
  executed test coverage in this repo and `vbt_engine.py` does not.

VectorBT genuinely earns its place here for one thing `engine.py`
structurally can't do well: fast parameter sweeps and walk-forward
across many strategy/instrument/window combinations, which the
roadmap's robustness testing (§5, §9 of the original brainstorming
brief) will eventually require at a scale a per-bar Python loop won't
handle. For a single backtest of a single strategy, the two should be
functionally interchangeable once parity is confirmed.

**No TA-Lib / `ta` package for RSI**, kept as a deliberate choice, not
merely an unresolved deviation: RSI is small and exactly defined
(Wilder's original recurrence), and it has been verified byte-correct
against the classic textbook reference values in this repo's own
executed test suite — swapping to a library implementation trades a
verified, zero-dependency function for an unverified one, for a
commodity indicator where the swap adds little. If you want strict
open-source-first compliance regardless, VectorBT itself bundles
TA-Lib access (`data.run("talib_func:rsi")`) once the `vectorbt` extra
is installed — a one-line swap in `app/features/indicators.py` if you
want it, but not currently wired up, since custom swing-point detection
with confirmation-lag semantics (the harder half of this module) has
no off-the-shelf equivalent regardless of that choice.

**ICT Silver Bullet is not one of the two strategies implemented.** The
prompt for this milestone specified ORB and RSI Divergence, superseding
the earlier three-strategy roadmap discussion; this is not a deviation,
just noted for continuity with that earlier discussion.

## Anti-lookahead mechanisms (the most important part of this milestone)

Three separate places can leak future information into a backtest, and
each has its own explicit guard, independently tested:

1. **Higher-timeframe context.** M1 candle timestamps mark bar OPEN
   time (confirmed from `app/data/aggregation.py`'s epoch-bucket-start
   convention). `align_higher_timeframe` shifts each HTF candle's
   visibility forward by its own bar duration before joining onto the
   execution timeframe, so a 1H candle opened at 09:00 is invisible to
   the base timeframe until 10:00, not 09:00.
   Tested in `tests/backtesting/test_bias.py::TestHigherTimeframeAlignmentBias`.

2. **Swing-point confirmation.** `detect_swing_points` returns a
   `confirmed_pos` distinct from a pivot's own position — a pivot at
   bar i cannot be confirmed until bar i+right_bars exists, because
   confirming it requires knowing no larger swing occurred in between.
   The RSI divergence strategy signals at the LATER pivot's
   confirmation time, never at either pivot's own timestamp.
   Tested in `tests/features/test_indicators.py::TestSwingPoints` and
   `tests/backtesting/test_bias.py::TestRSIDivergenceSignalBias`.

3. **Entry fill timing.** Every signal fills at the OPEN of the bar
   AFTER the bar that triggered it — never at that bar's own close,
   which by construction wasn't knowable at detection time. This is
   the one policy in `BacktestConfig` that is not configurable/optional
   (see `engine.py` module docstring for why).
   Tested in `tests/backtesting/test_engine.py::TestEntryFillPolicy`.

## Realism assumptions (all recorded in `BacktestResult.assumptions`)

- SL/TP monitoring begins the bar after entry, not the entry bar
  itself, to avoid same-bar fill-then-exit ordering ambiguity.
- A bar that touches both SL and TP levels resolves via
  `BacktestConfig.same_bar_sl_tp_policy`, defaulting to
  `"stop_loss_first"` (the conservative assumption — assuming the
  favorable order without intrabar data would systematically overstate
  results). `"take_profit_first"` exists for sensitivity comparison.
- Transaction costs are a flat spread (price points, not percent —
  spread does not scale with price the way commission does) applied as
  half-spread against the trader on both entry and exit, plus an
  optional flat commission. There is no slippage-beyond-spread model;
  this understates real costs during high-volatility/news periods.
  Per the roadmap's cost-realism requirement, spread should be set
  per-instrument-class by the caller (majors vs. XAU vs. index CFDs
  have structurally different spread behavior) — this milestone
  provides the mechanism, not instrument-specific default values.
- Position sizing is percent-of-current-equity risk sizing
  (compounding trade to trade) — a simplifying research assumption, not
  a claim about how a live account would size positions.
- Take-profit is always derived from the strategy's configured
  risk/reward ratio applied to the ACTUAL fill price and actual
  entry-to-stop distance, never from the price that triggered signal
  detection — so a signal's TP reflects what was actually tradeable,
  not what was observed at detection time.

## Strategy design notes

**ORB**: opening range = high/low over the first `opening_range_minutes`
of each session day (session boundary computed via IANA timezone
conversion per calendar day — no hard-coded UTC offset, consistent with
M1's own stance on this). Breakout definition defaults to close-based
(more conservative than wick-based; both are configurable). Stop-loss
is always the opposite side of the opening range — the only
`stop_loss_method` implemented this milestone, documented as a default
rather than a hidden assumption.

**RSI Divergence**: swing highs/lows are detected on PRICE; RSI's value
at those same price-pivot bars is what's compared (the standard
operational definition), rather than independently detecting RSI's own
local extrema. A signal compares each new pivot only against the
NEAREST EARLIER pivot of the same kind satisfying
`min_pivot_separation` — not against any historically qualifying pivot
— deliberately, because searching all history for whichever earlier
pivot happens to produce a divergence pattern would be a data-mining
risk in itself (cherry-picking the comparison point that confirms a
signal, rather than using the plain, nearest-neighbor definition of
"divergence"). An optional HTF trend filter exists but defaults OFF,
since the spec did not define what an HTF filter rule should be, and
shipping an invented rule as a silent default would itself be a hidden
assumption.

## Testing

Every pure-pandas/stdlib module (`app/features/*`, `app/strategies/*`,
`app/backtesting/engine.py`, `app/backtesting/metrics.py`) was actually
executed against its test suite in the development environment
(pandas/numpy are available there) and passes — this is not merely
"written but unverified" the way M1's own suite was noted to be. Numeric
correctness of RSI was checked against the classic Wilder textbook
example (values match to two decimal places).

Modules that touch SQLAlchemy or Pydantic
(`app/backtesting/data_access.py::load_ohlcv`, `app/backtesting/results.py`,
`app/backtesting/runner.py`, and `tests/backtesting/test_integration.py`)
could NOT be executed in this environment (no network access to install
those packages here, even though they are already real project
dependencies used throughout M1). These were reviewed line-by-line
against M1's actual repository/schema code and the integration test's
expected values were independently hand-computed against the exact
fixture data before being written — but per the same standard M1's own
docs applied to itself, treat them as reviewed-but-unverified until run
with `pytest` in an environment with the project's real dependencies
installed.

## Assumptions / unresolved items for future milestones

- No instrument-specific default cost model exists yet (per-instrument
  spread/slippage values) — `BacktestConfig.spread_points` must be set
  explicitly per backtest run. A future milestone (per the roadmap's
  §7 cost-model requirement) should own a small per-instrument-class
  cost lookup.
- `max_trades_per_day` on RSI Divergence and `max_trades_per_session`
  on ORB are both simple counters with no priority/quality ranking
  between candidate signals on the same day — the first qualifying
  signal wins, not the "best" one. This is intentional (avoids
  inventing an unstated ranking rule) but worth revisiting once M8's
  FIAS ranking exists.
- No experiment/parameter registry exists yet — every backtest's exact
  config travels with its `BacktestResult`, but nothing persists that
  across runs yet. Per the roadmap discussion, this should move earlier
  than originally planned rather than waiting for a dedicated
  milestone, precisely so results stay reproducible as more strategies
  and parameter variants are tried.
- A third strategy can be added by writing one new module implementing
  `Strategy` (`name`, `params()`, `generate_signals()`) — no changes to
  `engine.py`, `metrics.py`, or `runner.py` are required, per the
  acceptance criteria.

## Addendum: two more strategies (post-v5)

Two additional strategies were added on top of this milestone, per a
direct request, confirming the "no engine/metrics/runner changes
required" claim above held for both — except one deliberate, additive
engine change:

- **`app/strategies/mean_reversion.py`** — `bb_stoch_mean_reversion`:
  Bollinger Bands + Stochastic %K/%D crossover, filtered to ranging
  (low-ADX) markets. Needed no engine changes; targets are
  `risk_reward`-derived as usual. See the module docstring for the
  exact rule.
- **`app/strategies/liquidity_sweep_fvg.py`** — `liquidity_sweep_fvg`:
  1D directional bias, 1H NY-session liquidity-sweep ("fakeout")
  detection, 5M Fair-Value-Gap/inverse-FVG pullback entry, targeting
  the previous day's high/low. This one genuinely needed something the
  engine didn't support: a fixed price target rather than an
  R-multiple. **Engine change:** `Signal` gained an optional
  `take_profit` field (`app/strategies/base.py`); when set, the engine
  uses it as the trade's take-profit as-is instead of deriving one from
  `risk_reward` (`app/backtesting/engine.py`). Backward compatible by
  construction — ORB and RSI Divergence never set it, so their behavior
  is byte-for-byte unchanged (verified by the full existing test
  suite, unmodified, still passing). Two new indicators
  (`bollinger_bands`, `stochastic_oscillator`) plus two supporting ones
  (`atr`, `adx`) and a new `detect_fvg` structural-pattern detector were
  added to `app/features/indicators.py` for these two strategies to use.
