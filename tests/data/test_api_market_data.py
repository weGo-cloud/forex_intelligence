"""Tests for app.api.market_data, using an in-memory SQLite DB wired in
via a FastAPI dependency override so no real PostgreSQL is required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.data import models as _models  # noqa: F401
from app.data.repository import CandleRepository, InstrumentRepository
from app.data.schemas import InstrumentCreate
from app.main import app
from tests.data.conftest import utc
from tests.data.test_repository import make_candle


@pytest.fixture
def api_client():
    # TestClient dispatches requests on a separate thread from the one
    # that builds the engine/inserts fixture data. A plain
    # "sqlite:///:memory:" engine opens a new, empty in-memory database
    # per connection/thread (each :memory: connection is its own
    # database), so the request thread would see none of the fixture
    # rows - not a data problem, a fixture engine-configuration problem.
    # StaticPool + check_same_thread=False shares the single underlying
    # connection (and its :memory: database) across both threads.
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app), factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def test_list_instruments_returns_only_active_instruments(api_client) -> None:
    client, factory = api_client
    session = factory()
    InstrumentRepository(session).get_or_create(
        InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD")
    )
    InstrumentRepository(session).get_or_create(
        InstrumentCreate(symbol="OLD/PAIR", base_currency="OLD", quote_currency="PAIR", is_active=False)
    )
    session.commit()
    session.close()

    response = client.get("/market-data/instruments")

    assert response.status_code == 200
    symbols = {item["symbol"] for item in response.json()}
    assert symbols == {"EUR/USD"}


def test_get_candles_returns_stored_candles_for_range(api_client) -> None:
    client, factory = api_client
    session = factory()
    instrument = InstrumentRepository(session).get_or_create(
        InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD")
    )
    candles = [make_candle(utc(2026, 1, 1, 0, i)) for i in range(3)]
    CandleRepository(session).bulk_insert(instrument.id, candles)
    session.commit()
    session.close()

    response = client.get(
        "/market-data/candles",
        params={
            "symbol": "EUR/USD",
            "timeframe": "1M",
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-01-01T01:00:00Z",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 3
    assert body[0]["timeframe"] == "1M"


def test_get_candles_for_unknown_instrument_returns_404(api_client) -> None:
    client, _factory = api_client

    response = client.get(
        "/market-data/candles",
        params={
            "symbol": "NOPE/USD",
            "timeframe": "1M",
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-01-01T01:00:00Z",
        },
    )

    assert response.status_code == 404
