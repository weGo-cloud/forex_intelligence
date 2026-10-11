"""Fixtures for backtesting integration tests.

Mirrors tests/data/conftest.py's approach exactly (in-memory SQLite, no
live database required) rather than importing it directly, matching the
existing convention of one small conftest per test subpackage.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base
from app.data import models as _models  # noqa: F401  (registers tables on Base.metadata)


@pytest.fixture
def db_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def utc(*args, **kwargs) -> datetime:
    return datetime(*args, tzinfo=timezone.utc, **kwargs)
