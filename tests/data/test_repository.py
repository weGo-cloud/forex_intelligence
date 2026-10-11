"""Tests for app.data.repository, exercised against an in-memory SQLite DB."""

from decimal import Decimal

from sqlalchemy.orm import Session

from app.data.repository import CandleRepository, InstrumentRepository
from app.data.schemas import CanonicalCandle, InstrumentCreate
from app.data.timeframes import Timeframe
from tests.data.conftest import utc


def make_candle(timestamp, **overrides) -> CanonicalCandle:
    defaults = dict(
        instrument_symbol="EUR/USD",
        timeframe=Timeframe.M1,
        timestamp=timestamp,
        open=Decimal("1.1"),
        high=Decimal("1.101"),
        low=Decimal("1.099"),
        close=Decimal("1.1005"),
        volume=Decimal("100"),
        source="mock",
    )
    defaults.update(overrides)
    return CanonicalCandle(**defaults)


def test_get_or_create_instrument_creates_once_and_reuses_after(db_session: Session, eurusd: InstrumentCreate) -> None:
    repo = InstrumentRepository(db_session)

    first = repo.get_or_create(eurusd)
    second = repo.get_or_create(eurusd)

    assert first.id == second.id
    assert repo.get_by_symbol("EUR/USD") is not None


def test_list_active_excludes_inactive_instruments(db_session: Session) -> None:
    repo = InstrumentRepository(db_session)
    repo.get_or_create(InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD"))
    repo.get_or_create(
        InstrumentCreate(symbol="OLD/PAIR", base_currency="OLD", quote_currency="PAIR", is_active=False)
    )

    active_symbols = {i.symbol for i in repo.list_active()}

    assert active_symbols == {"EUR/USD"}


def test_bulk_insert_persists_new_candles(db_session: Session, eurusd: InstrumentCreate) -> None:
    instrument = InstrumentRepository(db_session).get_or_create(eurusd)
    candle_repo = CandleRepository(db_session)
    candles = [make_candle(utc(2026, 1, 1, 12, i)) for i in range(3)]

    inserted, duplicates = candle_repo.bulk_insert(instrument.id, candles)

    assert inserted == 3
    assert duplicates == 0
    stored = candle_repo.query(instrument.id, Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 2, 0, 0))
    assert len(stored) == 3


def test_bulk_insert_is_idempotent_on_rerun(db_session: Session, eurusd: InstrumentCreate) -> None:
    instrument = InstrumentRepository(db_session).get_or_create(eurusd)
    candle_repo = CandleRepository(db_session)
    candles = [make_candle(utc(2026, 1, 1, 12, i)) for i in range(3)]

    candle_repo.bulk_insert(instrument.id, candles)
    inserted_again, duplicates_again = candle_repo.bulk_insert(instrument.id, candles)

    assert inserted_again == 0
    assert duplicates_again == 3
    stored = candle_repo.query(instrument.id, Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 2, 0, 0))
    assert len(stored) == 3  # no duplicate rows despite re-running


def test_bulk_insert_partial_overlap_only_inserts_new_rows(db_session: Session, eurusd: InstrumentCreate) -> None:
    instrument = InstrumentRepository(db_session).get_or_create(eurusd)
    candle_repo = CandleRepository(db_session)

    first_batch = [make_candle(utc(2026, 1, 1, 12, i)) for i in range(3)]
    candle_repo.bulk_insert(instrument.id, first_batch)

    overlapping_batch = [make_candle(utc(2026, 1, 1, 12, i)) for i in range(2, 5)]
    inserted, duplicates = candle_repo.bulk_insert(instrument.id, overlapping_batch)

    assert inserted == 2  # minutes 3 and 4 are new
    assert duplicates == 1  # minute 2 already existed
    stored = candle_repo.query(instrument.id, Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 2, 0, 0))
    assert len(stored) == 5


def test_query_filters_by_date_range(db_session: Session, eurusd: InstrumentCreate) -> None:
    instrument = InstrumentRepository(db_session).get_or_create(eurusd)
    candle_repo = CandleRepository(db_session)
    candles = [make_candle(utc(2026, 1, 1, 12, i)) for i in range(10)]
    candle_repo.bulk_insert(instrument.id, candles)

    stored = candle_repo.query(
        instrument.id,
        Timeframe.M1,
        utc(2026, 1, 1, 12, 3),
        utc(2026, 1, 1, 12, 6),
    )

    assert [c.timestamp for c in stored] == [utc(2026, 1, 1, 12, i) for i in (3, 4, 5)]


def test_query_returns_results_ordered_by_timestamp(db_session: Session, eurusd: InstrumentCreate) -> None:
    instrument = InstrumentRepository(db_session).get_or_create(eurusd)
    candle_repo = CandleRepository(db_session)
    # Insert out of order.
    candles = [make_candle(utc(2026, 1, 1, 12, i)) for i in (2, 0, 1)]
    candle_repo.bulk_insert(instrument.id, candles)

    stored = candle_repo.query(instrument.id, Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 2, 0, 0))

    assert [c.timestamp for c in stored] == [utc(2026, 1, 1, 12, i) for i in (0, 1, 2)]


def test_stored_timestamps_round_trip_as_utc_aware(db_session: Session, eurusd: InstrumentCreate) -> None:
    instrument = InstrumentRepository(db_session).get_or_create(eurusd)
    candle_repo = CandleRepository(db_session)
    candle_repo.bulk_insert(instrument.id, [make_candle(utc(2026, 1, 1, 12, 0))])

    stored = candle_repo.query(instrument.id, Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 2, 0, 0))

    assert stored[0].timestamp.tzinfo is not None
    assert stored[0].timestamp.utcoffset().total_seconds() == 0
