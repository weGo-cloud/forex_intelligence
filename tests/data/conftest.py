"""Fixtures for market-data tests.

Uses an in-memory SQLite database rather than PostgreSQL so the suite
runs anywhere without a live database service. The repository layer is
written to be portable across backends specifically so this works (see
app/data/repository.py for why no PostgreSQL-specific SQL is used).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base
from app.data import models as _models  # noqa: F401  (registers tables on Base.metadata)
from app.data.schemas import InstrumentCreate


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


@pytest.fixture
def eurusd() -> InstrumentCreate:
    return InstrumentCreate(
        symbol="EUR/USD",
        asset_class="forex",
        base_currency="EUR",
        quote_currency="USD",
    )


def utc(*args, **kwargs) -> datetime:
    """Shorthand for building a UTC-aware datetime in tests."""
    return datetime(*args, tzinfo=timezone.utc, **kwargs)
