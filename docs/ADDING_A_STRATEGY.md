# Adding a Strategy — The Research Lab Workflow

This is the whole point of M2: adding a strategy should be these 7
steps, not a rebuild of the system. Every step below names the exact
file(s) you touch and the exact command you run — if any step ever
requires touching `engine.py`, `metrics.py`, or `runner.py`, something
has gone wrong with how the strategy was written (those three files
are strategy-agnostic by design; see `docs/M2_BACKTESTING_ENGINE.md`).

## 0. Before you start

Know the rule in plain language before opening an editor. If you can't
write it as a deterministic, objective procedure — exact entry
condition, exact stop-loss method, exact take-profit method, what's
configurable vs. a fixed default and why — you're not ready for step 2.
"Vague visual logic" is explicitly rejected in this codebase for
`orb.py` and `rsi_divergence.py`, and the same standard applies to
every strategy after them.

## 1. Define rules

Write the rule down (a comment block, a short doc, whatever) before
code. Identify:
- Exact entry condition (must be checkable from `execution_df` and,
  optionally, already-aligned `market.context` — nothing else)
- Exact stop-loss method
- Every parameter that should be configurable, with a sensible default
- Whether the rule needs a "confirmed pattern" concept (a pivot, a
  range) — if so, you need a confirmation-lag rule, not just a
  detection rule (see step 2's causality checklist)

## 2. Implement strategy

```
cp app/strategies/_template.py app/strategies/<your_strategy>.py
```

Fill in the TODOs in the copied file. It already:
- satisfies the `Strategy` protocol (`app/strategies/base.py`)
- registers itself via `@register_strategy(...)` at the bottom of the
  file — this is what makes step 6 automatic instead of a separate step
- includes a causality checklist in `generate_signals`'s docstring;
  every item on it has caused a real bug in this codebase during `orb.py`
  or `rsi_divergence.py`'s own development — don't skip it

Add one import line to `scripts/lab.py` (next to the existing
`import app.strategies.orb` line) so the CLI picks it up. This is
currently the one place a new strategy still requires a manual edit —
see that file's comment for why.

## 3. Write tests

At minimum, mirror the existing pattern:
- `tests/strategies/test_<your_strategy>.py` — one test per rule
  branch (entry fires, entry doesn't fire, edge cases in your
  config's `__post_init__` validation). See `tests/strategies/test_orb.py`
  for the shape.
- **A bias test is not optional.** If your rule has any concept of a
  confirmed pattern, add a case to `tests/backtesting/test_bias.py`
  following the existing pattern: run on data up to time T, run again
  on a longer series sharing an identical prefix, assert the result up
  to T is identical. This is the single highest-value test in this
  codebase — every existing strategy has one.

Run just your new tests:
```
pytest tests/strategies/test_<your_strategy>.py -v
```

## 4. Run backtest

```
python scripts/lab.py run \
  --strategy <your_strategy> \
  --symbol US30 \
  --timeframe M5 \
  --start 2024-01-01 --end 2024-03-01 \
  --param risk_reward=2.0 \
  --spread-points 1.0 --risk-pct 1.0
```

`--param key=value` is repeatable and maps directly onto your config's
constructor kwargs — every field in your `<YourStrategy>Config` is
already runnable from the CLI with zero extra plumbing.

If this prints "INSUFFICIENT EVIDENCE: zero trades" — that's a valid,
useful result (see the roadmap's own NO TRADE / insufficient-evidence
principle), not a bug to route around. Widen the range or check your
entry condition before assuming something is broken.

## 5. Validate

A single backtest over one range is not evidence of an edge — it's one
data point. Run walk-forward before drawing any conclusion:

```python
from app.backtesting.validation import run_walk_forward
# see app/backtesting/validation.py docstring for full usage;
# this logs EVERY window as its own tagged experiment automatically
```

Then compare across windows (step 7) filtered by strategy name — a
strategy whose expectancy is only positive in one window out of five
is not validated, no matter how good that one window looks.

This minimal walk-forward split is deliberately NOT the full
robustness suite from the original roadmap brief (parameter-neighborhood
sensitivity, purging/embargo, Monte Carlo) — build those only once this
minimum is actually in regular use and proving useful.

## 6. Register strategy

Already done in step 2 (the `@register_strategy` decorator) — this step
exists in the workflow as a checkpoint, not extra work:

```
python scripts/lab.py list-strategies
```

should show your strategy with its description. If it doesn't, you
forgot the `import app.strategies.<your_strategy>` line in `scripts/lab.py`
(step 2's last instruction).

## 7. Compare results

```
python scripts/lab.py compare --sort-by expectancy
python scripts/lab.py compare --strategy <your_strategy> --sort-by expectancy
```

Every `run` and every `run_walk_forward` window appends one line to
`experiments/log.jsonl` (see `app/backtesting/experiment_log.py`) —
nothing is lost between sessions, and nothing needs a database or a
dashboard to compare. nothing here ranks strategies for you or tells
you which one to trade — it just makes the numbers comparable across
every run you've ever done, which is the actual bottleneck a research
lab (as opposed to a one-off backtest script) needs solved.

---

## What "done" looks like for a new strategy

- `app/strategies/<name>.py` exists, registered, causality-checklist-clean
- `tests/strategies/test_<name>.py` passes
- A case exists in `tests/backtesting/test_bias.py` for this strategy
- At least one walk-forward run is logged in `experiments/log.jsonl`
- `scripts/lab.py compare --strategy <name>` shows results across
  multiple windows, not just one

None of this requires touching `engine.py`, `vbt_engine.py`,
`metrics.py`, `runner.py`, or the registry mechanism itself. If a step
ever seems to require that, stop and reconsider the strategy's design
before changing shared infrastructure to accommodate one strategy.
