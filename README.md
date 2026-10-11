# Forex Intelligence

Forex Intelligence is a quantitative forex research and analysis platform.
The long-term vision includes evaluating multiple forex strategies,
analyzing market regimes, backtesting, estimating trade-outcome
probabilities, ranking strategies, a web dashboard, and paper trading.

**This project is currently a research and paper-trading platform.
Real-money automated trading is not implemented.**

## Current scope: Milestone M8

M0 built the foundation (FastAPI app, config, logging, DB connection,
Docker). M1 built the market-data foundation. M2 added a strategy-agnostic
backtesting engine plus two strategies (ORB, RSI divergence) and two
multi-timeframe/indicator-based strategies (liquidity-sweep FVG, and
Bollinger/Stochastic mean reversion). M3 turned a single backtest into a
reproducible, persisted research experiment. M4 added a deterministic,
causal Market Regime & Context engine. M5 added the first historical
outcome and probability layer. M6 added the evaluation layer that
composes M2 signals + M3 walk-forward infrastructure + M4 regime context
+ M5 probability into strategy × context evidence reports — see
`docs/M6_STRATEGY_REGIME_EVALUATION.md`. M7 added the Decision & Setup
Intelligence foundation — structured observations, a deterministic setup
state machine, and an explainable decision-state engine that consumes M6
evidence without recomputing it (see `docs/M7_DECISION_SETUP_INTELLIGENCE.md`).
M8 adds a validated ranking layer on top: real multiple-testing
correction, an optimization-leakage guard, a robustness gate, and a
transparent, component-visible composite score (FIAS) — never a
hidden number, never "the best strategy" (see
`docs/M8_RANKING_AND_FIAS.md`). In short, M8 adds:

- Exact one-sided binomial testing + Benjamini-Hochberg FDR correction
  across a ranked batch — `q_value`/`significant` on every entry
  (`app/ranking/validation.py`)
- An optimization-leakage guard and an M3-robustness disqualification
  gate — both explicit about degrading to "not checkable" absent
  provenance, never assuming a clean result
- FIAS (`app/ranking/fias.py`) — a configurable weighted composite
  over naturally-bounded `[0,1]` components (sample adequacy, win-rate
  confidence, walk-forward stability, M3 robustness), withheld below a
  minimum evidence tier, every component always visible alongside the
  score
- `rank_evaluations` (`app/ranking/ranking.py`) — batch-scoped ranking
  that never deletes an excluded entry, only labels why it was excluded

M7 adds:

- A source-agnostic `Observation` model (`app/observations`) — market
  data today, screenshot/manual sources representable for later
- Setup detection + a deterministic 7-state setup state machine
  (`app/setups`)
- `DecisionEngine` producing action states (`WAIT`/`WATCH`/
  `VALID_SETUP`/`ENTRY_CONDITION_MET`/`INVALIDATED`/
  `INSUFFICIENT_EVIDENCE`) with reasons, missing conditions and
  invalidation conditions — no confidence scores, no orders
  (`app/decision`)
- A thin stateless API: `POST /setups/detect`, `POST /setups/evaluate`

M6 adds:

- `StrategyEvaluator` (`app/evaluation/evaluator.py`) — baseline,
  combination, and walk-forward evaluation, built entirely on M5's
  `ProbabilityEstimator`/`OutcomeFilter` and M3's `walk_forward_windows`
  (no new probability math, no second walk-forward implementation)
- An explicit, configurable minimum-sample verdict (`INSUFFICIENT` /
  `LIMITED` / `ADEQUATE`) that never hides, deletes, or borrows a
  small-sample result from a broader population (`app/evaluation/models.py`)
- Descriptive temporal stability metrics across walk-forward windows —
  mean/median/range of expected R and win rate, never a single opaque
  robustness score (`app/evaluation/stability.py`)
- Non-ranking, order-preserving comparison of evaluation results, with
  an automatic multiple-testing caveat whenever more than one
  configuration is compared (`app/evaluation/comparison.py`)

M5 adds the first historical outcome and probability layer —
deterministic forward outcome labeling plus transparent, empirical
probability estimation, conditioned on strategy/instrument/regime/
session — see `docs/M5_OUTCOME_PROBABILITY.md` for the full write-up.
In short, M5 adds:

- A deterministic, direction-aware triple-barrier outcome labeler
  (`app/probability/labeling.py`) with an explicit, honest same-bar
  TP/SL collision policy (default: `unresolved`, never a guessed winner)
- `TradeOutcome` (`app/probability/models.py`) — `WIN`/`LOSS`/
  `BREAKEVEN`/`UNRESOLVED` labels, embedding M4's `RegimeContext` as-is
- A composable historical outcome dataset (`app/probability/outcomes.py`)
  with a walk-forward-safe `resolved_before(T)` causal cutoff
- Composable conditional filtering by strategy/instrument/timeframe/
  direction/regime state/session, with no function-per-combination
  (`app/probability/filters.py`)
- Transparent empirical probability estimation — sample counts, Wilson
  confidence intervals, and expected R, never fabricated precision on a
  small sample (`app/probability/estimators.py`)

M4 adds a deterministic, causal Market Regime & Context engine — a
descriptive (not predictive) read on trend/volatility/structure
conditions — see `docs/M4_REGIME_CONTEXT.md` for the full write-up. In
short, M4 adds:

- Reusable trend, volatility, and structure measurement functions
  (`app/features/trend.py`, `app/features/volatility.py`,
  `app/features/structure.py`), built on top of the existing indicators
  and swing-point detection rather than duplicating them
- Three independent regime dimensions — `TrendState`, `VolatilityState`,
  `StructureState` (`app/regimes/models.py`)
- A configurable, deterministic classifier (`app/regimes/classifier.py`,
  `app/regimes/detectors.py`, `app/regimes/config.py`) that turns those
  measurements into a `RegimeContext` per bar (`app/regimes/context.py`),
  with an explicit `UNKNOWN` state wherever data is missing or
  insufficient — never a guessed value
- Full anti-lookahead test coverage (`tests/regimes/test_bias.py`)

M3 turns a single backtest into a reproducible, persisted research
experiment — see `docs/M3_RESEARCH_EXPERIMENTS.md` for the full
write-up. In short, M3 adds:

- A canonical, deterministically-hashable experiment specification
  (`app/research/models.py`)
- A persistent experiment registry (PostgreSQL/SQLAlchemy, immutable
  once completed) (`app/research/registry.py`)
- Experiment execution (`app/research/experiment.py`), a strategy-
  agnostic parameter-sweep engine (`app/research/sweep.py`), persisted
  walk-forward experiments (`app/research/runner.py`), parameter-
  neighborhood robustness analysis (`app/research/robustness.py`), and
  experiment comparison (`app/research/comparison.py`)
- A minimal research API (`app/api/research.py`):
  `POST/GET /experiments`, `GET /experiments/{id}`,
  `GET /experiments/{id}/results`, `GET /experiments/{id}/children`,
  `GET /experiments/compare`

M1 added:

- Instrument and OHLCV candle data model (`app/data/models.py`),
  persisted via SQLAlchemy/PostgreSQL with Alembic migrations
- Supported timeframes (`1M, 5M, 15M, 30M, 1H, 4H, 1D`), extensible
  without touching existing code (`app/data/timeframes.py`)
- A vendor-agnostic market-data provider abstraction
  (`app/data/providers/`), plus a deterministic mock provider for tests
  and local development — no commercial vendor is wired in yet
- Raw provider data -> canonical candle normalization
  (`app/data/normalization.py`)
- OHLC/data-quality validation, duplicate detection, and gap detection
  that only *reports* problems — it never silently modifies data or
  manufactures candles to fill gaps (`app/data/validation.py`)
- An idempotent ingestion pipeline (provider -> normalize -> validate ->
  persist -> record provenance) (`app/data/ingestion.py`)
- Ingestion provenance / data-quality reporting
  (`IngestionRun` in `app/data/models.py`)
- A repository/query layer so other modules can ask for candles by
  instrument/timeframe/date-range without knowing about SQLAlchemy or
  PostgreSQL (`app/data/repository.py`)
- Timeframe aggregation (deriving higher timeframes from lower ones)
  (`app/data/aggregation.py`)
- Read-only HTTP endpoints for instruments and candles
  (`app/api/market_data.py`)

**Explicitly not implemented yet** (per milestone scope): strategy
ranking, the web dashboard, paper trading, machine learning, live
trading, and broker integrations.

See `docs/DEVELOPMENT_NOTES.md` for M0 decisions,
`docs/M1_DATA_FOUNDATION.md` for M1 decisions, data model, validation
rules, and known limitations, `docs/M2_BACKTESTING_ENGINE.md` for the
backtesting engine and strategies, `docs/M3_RESEARCH_EXPERIMENTS.md`
for the research/experiment layer, `docs/M4_REGIME_CONTEXT.md` for
the market regime & context engine, `docs/M5_OUTCOME_PROBABILITY.md`
for outcome labeling and probability estimation, and
`docs/M6_STRATEGY_REGIME_EVALUATION.md` for the strategy × regime
evaluation layer, `docs/M7_DECISION_SETUP_INTELLIGENCE.md` for the
setup/decision layer, and `docs/M8_RANKING_AND_FIAS.md` for the
ranking and FIAS layer.

## Technology stack

- **Backend:** Python, FastAPI
- **Database:** PostgreSQL
- **Testing:** pytest
- **Dependency management:** pyproject.toml
- **Containerization:** Docker, Docker Compose

## Project structure

```
forex-intelligence/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/                 # HTTP routers (health.py, market_data.py)
│   ├── core/                # config, logging, database (infrastructure)
│   ├── data/                # market data: models, providers, ingestion (M1)
│   ├── features/            # indicators, session windows (M2)
│   ├── regimes/             # market regime & context engine (M4)
│   ├── strategies/          # trading strategies: ORB, RSI divergence, liquidity-sweep FVG, BB/Stochastic mean reversion (M2)
│   ├── backtesting/         # backtesting engine (M2)
│   ├── research/            # experiments, sweeps, walk-forward, robustness, comparison (M3)
│   ├── probability/         # outcome labeling & probability estimation (M5)
│   ├── ranking/             # validated ranking & FIAS (M8)
│   ├── observations/        # structured market observations (M7)
│   ├── setups/              # setup model, state machine, detection (M7)
│   ├── decision/            # decision engine, M6 evidence bridge (M7)
│   ├── evaluation/          # strategy x regime evaluation & robustness (M6)
│   ├── ranking/             # strategy ranking (future milestone)
│   ├── risk/                # risk management (future milestone)
│   └── paper_trading/       # paper trading (future milestone)
├── tests/                   # pytest suite (tests/data, tests/backtesting, tests/research, tests/regimes, tests/probability, tests/evaluation, tests/observations, tests/setups, tests/decision, tests/ranking)
├── migrations/              # Alembic revisions (market-data schema from M1, experiments schema from M3)
├── scripts/                 # check_db.py and future maintenance scripts
├── docs/                    # development notes
├── config/                  # reserved for future non-secret config files
├── frontend/                # reserved for the web dashboard (future milestone)
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Environment setup

1. Install Python 3.11+.
2. Copy the environment template and adjust if needed:
   ```bash
   cp .env.example .env
   ```
   The defaults work out of the box with the Docker Compose PostgreSQL
   service below. `.env` is git-ignored — never commit real credentials.
3. Create a virtual environment and install the project with dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

## Running locally

Requires a reachable PostgreSQL instance matching `DATABASE_URL` in `.env`
(the Docker Compose `db` service works well for this even if you run the
app itself outside Docker).

```bash
uvicorn app.main:app --reload
```

Then visit `http://localhost:8000/health` — expected response:

```json
{"status": "ok"}
```

To independently verify the PostgreSQL connection:

```bash
python scripts/check_db.py
```

Apply the market-data schema migrations (requires a reachable PostgreSQL
instance matching `DATABASE_URL`):

```bash
alembic upgrade head
```

## Running with Docker

```bash
docker compose up --build
```

This starts the FastAPI app on `http://localhost:8000` and a PostgreSQL
16 instance on port 5432, using the credentials defined in
`docker-compose.yml` (local development only, not real secrets).

## Running tests

```bash
pytest
```

The `GET /health` test and all `tests/data/` market-data tests run against
an in-memory SQLite database and do not require a live PostgreSQL
connection — the repository layer is written to be portable across
backends specifically so the suite runs anywhere (see
`docs/M1_DATA_FOUNDATION.md`).

## What is NOT implemented yet

- Trade-outcome probability estimation
- Strategy ranking
- Web dashboard
- Paper trading
- Machine learning / AI-driven decision-making
- A real commercial market-data vendor integration (only the mock
  provider exists so far — see `app/data/providers/`)
- Live trading and broker integrations (and per the project objective,
  V1 as a whole will never execute real-money trades)
