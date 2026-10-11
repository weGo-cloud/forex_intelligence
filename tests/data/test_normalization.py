"""Tests for app.data.normalization."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.data.exceptions import NormalizationError
from app.data.normalization import normalize_candle
from app.data.providers.base import ProviderCandle
from app.data.timeframes import Timeframe


def test_normalizes_datetime_timestamp_and_numeric_prices() -> None:
    raw = ProviderCandle(
        raw_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        raw_open=1.1000,
        raw_high=1.1010,
        raw_low=1.0990,
        raw_close=1.1005,
        raw_volume=1000,
    )

    canonical = normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")

    assert canonical.instrument_symbol == "EUR/USD"
    assert canonical.timeframe == Timeframe.M1
    assert canonical.timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert canonical.open == Decimal("1.1")
    assert canonical.volume == Decimal("1000")
    assert canonical.source == "mock"


def test_normalizes_epoch_seconds_timestamp() -> None:
    epoch_seconds = 1_735_732_800  # 2025-01-01T12:00:00Z
    raw = ProviderCandle(
        raw_timestamp=epoch_seconds,
        raw_open="1.1",
        raw_high="1.11",
        raw_low="1.09",
        raw_close="1.105",
    )

    canonical = normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")

    assert canonical.timestamp.tzinfo == timezone.utc
    assert canonical.timestamp == datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)


def test_normalizes_epoch_milliseconds_timestamp() -> None:
    epoch_ms = 1_735_732_800_000  # same instant as above, in ms
    raw = ProviderCandle(
        raw_timestamp=epoch_ms,
        raw_open="1.1",
        raw_high="1.11",
        raw_low="1.09",
        raw_close="1.105",
    )

    canonical = normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")

    assert canonical.timestamp == datetime.fromtimestamp(1_735_732_800, tz=timezone.utc)


def test_normalizes_iso_string_timestamp_with_z_suffix() -> None:
    raw = ProviderCandle(
        raw_timestamp="2026-01-01T12:00:00Z",
        raw_open="1.1",
        raw_high="1.11",
        raw_low="1.09",
        raw_close="1.105",
    )

    canonical = normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")

    assert canonical.timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_naive_datetime_timestamp_is_rejected() -> None:
    raw = ProviderCandle(
        raw_timestamp=datetime(2026, 1, 1, 12, 0),  # no tzinfo
        raw_open="1.1",
        raw_high="1.11",
        raw_low="1.09",
        raw_close="1.105",
    )

    with pytest.raises(NormalizationError):
        normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")


def test_unparseable_timestamp_string_is_rejected() -> None:
    raw = ProviderCandle(
        raw_timestamp="not-a-timestamp",
        raw_open="1.1",
        raw_high="1.11",
        raw_low="1.09",
        raw_close="1.105",
    )

    with pytest.raises(NormalizationError):
        normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")


def test_missing_price_field_is_rejected() -> None:
    raw = ProviderCandle(
        raw_timestamp="2026-01-01T12:00:00Z",
        raw_open=None,
        raw_high="1.11",
        raw_low="1.09",
        raw_close="1.105",
    )

    with pytest.raises(NormalizationError):
        normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")


def test_non_numeric_price_field_is_rejected() -> None:
    raw = ProviderCandle(
        raw_timestamp="2026-01-01T12:00:00Z",
        raw_open="not-a-number",
        raw_high="1.11",
        raw_low="1.09",
        raw_close="1.105",
    )

    with pytest.raises(NormalizationError):
        normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")


def test_missing_volume_normalizes_to_none() -> None:
    raw = ProviderCandle(
        raw_timestamp="2026-01-01T12:00:00Z",
        raw_open="1.1",
        raw_high="1.11",
        raw_low="1.09",
        raw_close="1.105",
        raw_volume=None,
    )

    canonical = normalize_candle(raw, symbol="EUR/USD", timeframe=Timeframe.M1, source="mock")

    assert canonical.volume is None
