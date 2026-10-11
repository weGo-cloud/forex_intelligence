# M0 Development Notes — Assumptions & Ambiguity Resolutions

This file records every place the M0 spec was ambiguous and the simplest
reasonable choice that was made, per the "do not silently make major
architectural decisions" rule.

1. **Database driver**: chose `psycopg` (v3, `psycopg[binary]`) over
   `psycopg2-binary`. It's the actively-developed driver, has first-class
   SQLAlchemy 2.0 support, and ships a binary wheel so no system-level
   `libpq-dev` build step is needed in the Dockerfile.

2. **Where DB code lives**: `app/core/database.py` rather than `app/data/`.
   `app/core/` is treated as cross-cutting infrastructure (config, logging,
   DB engine/session), while `app/data/` is reserved for the forex market
   data domain module that arrives in a later milestone. This keeps
   "infrastructure" and "domain" concerns separate.

3. **`/health` does not check the database.** It answers "is the web
   process up," which should stay reliable even if PostgreSQL is briefly
   unreachable. Database connectivity is proven independently via
   `scripts/check_db.py`, which is the artifact acceptance criterion
   "PostgreSQL connection works" maps to. This also keeps the required
   test (`GET /health` → 200) independent of having a real database
   available, so `pytest` runs the same locally and in CI without a
   database container.

4. **Default `DATABASE_URL` value**: `config.py` ships a non-secret local
   default (`postgresql+psycopg://forex:forex@localhost:5432/forex_intelligence`)
   matching `.env.example` and the `docker-compose.yml` credentials. These
   are placeholder local-development credentials, not real secrets — every
   real environment overrides `DATABASE_URL` via a real environment
   variable, and `.env` itself is git-ignored. This default exists purely
   so `pytest` and local `uvicorn` runs don't require a `.env` file to
   exist before the app can start.

5. **Empty structural directories** (`migrations/`, `scripts/`, `config/`,
   `frontend/`) are placeholders for later milestones:
   - `migrations/` — reserved for Alembic revisions once the trading data
     model is formally reviewed and introduced.
   - `config/` — reserved for future non-secret static configuration
     files (e.g. strategy parameter files), separate from environment
     variables.
   - `frontend/` — reserved for the web dashboard milestone.
   - `scripts/` currently holds only `check_db.py`.

6. **No Alembic, no migration tooling yet.** The spec explicitly says not
   to build the full trading schema yet, and there is no schema to migrate
   yet either. Adding Alembic now, with nothing for it to manage, would be
   exactly the kind of premature infrastructure the spec warns against.

7. **`app/api/health.py`** exists as its own module (rather than defining
   the route inline in `main.py`) so that adding the next router in a
   future milestone is a "create a new file + one `include_router` line"
   change, not a `main.py` rewrite — in line with the "future components
   must plug in without major rewrites" principle.
