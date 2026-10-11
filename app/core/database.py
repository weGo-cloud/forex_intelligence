"""Minimal PostgreSQL database abstraction for M0.

Scope for M0: prove that the application *can* connect to PostgreSQL using
a clean, reusable abstraction. No trading schema is defined here — the
data model will be introduced in a later milestone after formal review, at
which point the `Base` declared below becomes the parent for real ORM
models and `migrations/` starts hosting Alembic revisions.

Design notes:
- The SQLAlchemy Engine is created lazily and cached, not at import time,
  so importing this module never has side effects (important for tests
  that don't need a real database).
- `SessionLocal` is a session factory, not a global session, to avoid
  shared mutable state across requests.
"""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for future ORM models (introduced in a later milestone)."""


@lru_cache
def get_engine() -> Engine:
    """Return a cached SQLAlchemy Engine built from application settings."""
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return a cached session factory bound to the application engine."""
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and closes it after use."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:
    """Attempt a trivial round-trip query against PostgreSQL.

    Returns True if the connection and query succeed, False otherwise.
    Never raises — callers (startup checks, scripts) get a clean boolean
    instead of having to handle driver-specific exceptions.
    """
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - deliberately broad: this is a health probe
        return False
