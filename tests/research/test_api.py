"""Tests for app.api.research, following the same in-memory-SQLite +
dependency-override pattern as tests/data/test_api_market_data.py.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.strategies.orb  # noqa: F401  (registers "orb")
from app.core.database import Base, get_db
from app.data import models as _data_models  # noqa: F401
from app.data.ingestion import run_ingestion
from app.data.providers.mock import MockMarketDataProvider
from app.data.schemas import InstrumentCreate
from app.data.timeframes import Timeframe
from app.main import app
from app.research import registry as _research_models  # noqa: F401
from tests.research.conftest import orb_breakout_candles, utc


@pytest.fixture
def api_client():
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

    session = factory()
    start = utc(2024, 1, 2, 0, 0)
    candles = orb_breakout_candles(start, count=40 * 6 * 3)
    end = start + timedelta(minutes=5 * len(candles))
    provider = MockMarketDataProvider(fixed_candles=candles)
    run_ingestion(
        session,
        provider,
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        start=start,
        end=end,
        instrument_defaults=InstrumentCreate(symbol="EURUSD", asset_class="forex", base_currency="EUR", quote_currency="USD"),
    )
    session.commit()
    session.close()

    try:
        yield TestClient(app), start, end
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def _create_payload(start, end, **overrides):
    payload = {
        "strategy_name": "orb",
        "strategy_params": {
            "instrument_symbol": "EURUSD",
            "opening_range_minutes": 15,
            "execution_timeframe_minutes": 5,
            "risk_reward": 2.0,
        },
        "instrument_symbol": "EURUSD",
        "timeframe": "5M",
        "test_period_start": start.isoformat(),
        "test_period_end": end.isoformat(),
    }
    payload.update(overrides)
    return payload


def test_create_experiment_runs_and_returns_completed(api_client) -> None:
    client, start, end = api_client
    response = client.post("/experiments", json=_create_payload(start, end))
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["experiment_id"].startswith("exp_")
    assert body["result_summary"]["trade_count"] >= 0


def test_get_experiment_by_id(api_client) -> None:
    client, start, end = api_client
    created = client.post("/experiments", json=_create_payload(start, end)).json()

    response = client.get(f"/experiments/{created['experiment_id']}")
    assert response.status_code == 200
    assert response.json()["experiment_id"] == created["experiment_id"]


def test_get_unknown_experiment_returns_404(api_client) -> None:
    client, _start, _end = api_client
    response = client.get("/experiments/exp_999999")
    assert response.status_code == 404


def test_list_experiments(api_client) -> None:
    client, start, end = api_client
    client.post("/experiments", json=_create_payload(start, end))
    client.post("/experiments", json=_create_payload(start, end))

    response = client.get("/experiments", params={"strategy_name": "orb"})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_results(api_client) -> None:
    client, start, end = api_client
    created = client.post("/experiments", json=_create_payload(start, end)).json()

    response = client.get(f"/experiments/{created['experiment_id']}/results")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_compare_experiments(api_client) -> None:
    client, start, end = api_client
    a = client.post("/experiments", json=_create_payload(start, end)).json()
    b = client.post(
        "/experiments",
        json=_create_payload(
            start,
            end,
            strategy_params={
                "instrument_symbol": "EURUSD",
                "opening_range_minutes": 15,
                "execution_timeframe_minutes": 5,
                "risk_reward": 3.0,
            },
        ),
    ).json()

    response = client.get("/experiments/compare", params=[("id", a["experiment_id"]), ("id", b["experiment_id"])])
    assert response.status_code == 200
    rows = response.json()
    assert [r["experiment_id"] for r in rows] == [a["experiment_id"], b["experiment_id"]]


def test_create_experiment_with_invalid_period_returns_422(api_client) -> None:
    client, start, end = api_client
    response = client.post("/experiments", json=_create_payload(end, start))  # end before start
    assert response.status_code == 422
