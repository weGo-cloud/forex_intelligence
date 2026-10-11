"""End-to-end tests for app.data.ingestion, using the mock provider
against an in-memory SQLite database.
"""

from datetime import timedelta

from sqlalchemy.orm import Session

from app.data.ingestion import run_ingestion
from app.data.providers.base import ProviderCandle
from app.data.providers.mock import MockMarketDataProvider
from app.data.repository import CandleRepository, InstrumentRepository
from app.data.schemas import InstrumentCreate
from app.data.timeframes import Timeframe
from tests.data.conftest import utc


def test_ingestion_creates_instrument_and_persists_candles(db_session: Session) -> None:
    provider = MockMarketDataProvider()
    start = utc(2026, 1, 1, 0, 0)
    end = utc(2026, 1, 1, 1, 0)

    result = run_ingestion(
        db_session,
        provider,
        symbol="EUR/USD",
        timeframe=Timeframe.M1,
        start=start,
        end=end,
        instrument_defaults=InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD"),
    )

    assert result.received == 60
    assert result.accepted == 60
    assert result.rejected == 0
    assert result.duplicates == 0
    assert result.status == "success"
    assert result.run_id is not None

    instrument = InstrumentRepository(db_session).get_by_symbol("EUR/USD")
    assert instrument is not None
    stored = CandleRepository(db_session).query(instrument.id, Timeframe.M1, start, end)
    assert len(stored) == 60


def test_ingestion_is_idempotent_on_rerun(db_session: Session) -> None:
    provider = MockMarketDataProvider()
    start = utc(2026, 1, 1, 0, 0)
    end = utc(2026, 1, 1, 0, 10)
    defaults = InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD")

    first = run_ingestion(db_session, provider, "EUR/USD", Timeframe.M1, start, end, instrument_defaults=defaults)
    second = run_ingestion(db_session, provider, "EUR/USD", Timeframe.M1, start, end, instrument_defaults=defaults)

    assert first.accepted == 10
    assert second.accepted == 0
    assert second.duplicates == 10

    instrument = InstrumentRepository(db_session).get_by_symbol("EUR/USD")
    stored = CandleRepository(db_session).query(instrument.id, Timeframe.M1, start, end)
    assert len(stored) == 10  # rerun did not create duplicate rows


def test_ingestion_rejects_malformed_records_without_aborting_the_batch(db_session: Session) -> None:
    good_timestamp = utc(2026, 1, 1, 12, 0)
    bad_ohlc_timestamp = utc(2026, 1, 1, 12, 1)
    unparseable_timestamp_record = ProviderCandle(
        raw_timestamp="not-a-timestamp", raw_open=1.1, raw_high=1.2, raw_low=1.0, raw_close=1.15
    )
    bad_ohlc_record = ProviderCandle(
        raw_timestamp=bad_ohlc_timestamp, raw_open=1.1, raw_high=1.0, raw_low=1.2, raw_close=1.15  # high < low
    )
    good_record = ProviderCandle(
        raw_timestamp=good_timestamp, raw_open=1.1, raw_high=1.2, raw_low=1.0, raw_close=1.15
    )

    provider = MockMarketDataProvider(
        fixed_candles=[unparseable_timestamp_record, bad_ohlc_record, good_record]
    )
    defaults = InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD")

    result = run_ingestion(
        db_session,
        provider,
        "EUR/USD",
        Timeframe.M1,
        utc(2026, 1, 1, 0, 0),
        utc(2026, 1, 2, 0, 0),
        instrument_defaults=defaults,
    )

    assert result.received == 3
    assert result.accepted == 1
    assert result.rejected == 2
    assert result.status == "partial"
    codes = {issue.code for issue in result.rejection_reasons}
    assert "normalization_error" in codes
    assert "high_below_low" in codes


def test_ingestion_sorts_out_of_order_records_before_persisting(db_session: Session) -> None:
    t0 = utc(2026, 1, 1, 12, 0)
    t1 = t0 + timedelta(minutes=1)
    t2 = t0 + timedelta(minutes=2)
    out_of_order = [
        ProviderCandle(raw_timestamp=t2, raw_open=1.1, raw_high=1.2, raw_low=1.0, raw_close=1.15),
        ProviderCandle(raw_timestamp=t0, raw_open=1.1, raw_high=1.2, raw_low=1.0, raw_close=1.15),
        ProviderCandle(raw_timestamp=t1, raw_open=1.1, raw_high=1.2, raw_low=1.0, raw_close=1.15),
    ]
    provider = MockMarketDataProvider(fixed_candles=out_of_order)
    defaults = InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD")

    run_ingestion(
        db_session, provider, "EUR/USD", Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 2, 0, 0),
        instrument_defaults=defaults,
    )

    instrument = InstrumentRepository(db_session).get_by_symbol("EUR/USD")
    stored = CandleRepository(db_session).query(instrument.id, Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 2, 0, 0))
    assert [c.timestamp for c in stored] == [t0, t1, t2]


def test_ingestion_reports_gaps_without_rejecting_surrounding_data(db_session: Session) -> None:
    t0 = utc(2026, 1, 1, 12, 0)
    t_far = t0 + timedelta(minutes=30)
    records = [
        ProviderCandle(raw_timestamp=t0, raw_open=1.1, raw_high=1.2, raw_low=1.0, raw_close=1.15),
        ProviderCandle(raw_timestamp=t_far, raw_open=1.1, raw_high=1.2, raw_low=1.0, raw_close=1.15),
    ]
    provider = MockMarketDataProvider(fixed_candles=records)
    defaults = InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD")

    result = run_ingestion(
        db_session, provider, "EUR/USD", Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 2, 0, 0),
        instrument_defaults=defaults,
    )

    assert result.accepted == 2
    assert result.rejected == 0
    assert len(result.gaps) == 1


def test_ingestion_records_provenance(db_session: Session) -> None:
    from app.data.models import IngestionRun

    provider = MockMarketDataProvider()
    defaults = InstrumentCreate(symbol="EUR/USD", base_currency="EUR", quote_currency="USD")

    result = run_ingestion(
        db_session, provider, "EUR/USD", Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 1, 0, 5),
        instrument_defaults=defaults,
    )

    run = db_session.get(IngestionRun, result.run_id)
    assert run is not None
    assert run.provider_name == "mock"
    assert run.received_count == 5
    assert run.accepted_count == 5
    assert run.status == "success"


def test_ingestion_raises_for_unknown_instrument_without_defaults(db_session: Session) -> None:
    import pytest

    provider = MockMarketDataProvider()
    with pytest.raises(ValueError):
        run_ingestion(
            db_session, provider, "GBP/JPY", Timeframe.M1, utc(2026, 1, 1, 0, 0), utc(2026, 1, 1, 0, 5)
        )
