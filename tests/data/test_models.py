"""Tests for app.data.models: DB-level constraints as a safety net.

These complement the repository tests: the repository already prevents
duplicates at the application level, but the DB constraint must also
hold on its own (e.g. against a concurrent writer bypassing the
repository).
"""

from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.data.models import Candle
from app.data.repository import InstrumentRepository
from app.data.schemas import InstrumentCreate
from tests.data.conftest import utc


def _candle_kwargs(instrument_id: int, timestamp) -> dict:
    return dict(
        instrument_id=instrument_id,
        timeframe="1M",
        timestamp=timestamp,
        open=Decimal("1.1"),
        high=Decimal("1.2"),
        low=Decimal("1.0"),
        close=Decimal("1.15"),
        volume=Decimal("10"),
    )


def test_duplicate_candle_violates_unique_constraint(db_session: Session, eurusd: InstrumentCreate) -> None:
    instrument = InstrumentRepository(db_session).get_or_create(eurusd)
    timestamp = utc(2026, 1, 1, 12, 0)

    db_session.add(Candle(**_candle_kwargs(instrument.id, timestamp)))
    db_session.commit()

    db_session.add(Candle(**_candle_kwargs(instrument.id, timestamp)))
    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_naive_timestamp_is_rejected_before_reaching_the_database(
    db_session: Session, eurusd: InstrumentCreate
) -> None:
    from datetime import datetime

    from sqlalchemy.exc import StatementError

    instrument = InstrumentRepository(db_session).get_or_create(eurusd)
    kwargs = _candle_kwargs(instrument.id, datetime(2026, 1, 1, 12, 0))  # naive, no tzinfo

    db_session.add(Candle(**kwargs))
    # SQLAlchemy wraps the type decorator's ValueError in a StatementError
    # before it reaches the DBAPI - assert on the wrapper and its cause
    # rather than the bare ValueError, so this test tracks what actually
    # gets raised instead of an implementation detail one layer too deep.
    with pytest.raises(StatementError) as exc_info:
        db_session.flush()
    assert isinstance(exc_info.value.orig, ValueError)

    db_session.rollback()
