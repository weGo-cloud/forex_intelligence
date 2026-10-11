"""Tests for app.data.providers.mock."""

from app.data.providers.base import ProviderCandle
from app.data.providers.mock import MockMarketDataProvider
from app.data.timeframes import Timeframe
from tests.data.conftest import utc


def test_generated_mode_produces_expected_candle_count() -> None:
    provider = MockMarketDataProvider()
    start = utc(2026, 1, 1, 0, 0)
    end = utc(2026, 1, 1, 1, 0)

    candles = list(provider.get_historical_candles("EUR/USD", Timeframe.M1, start, end))

    assert len(candles) == 60
    assert candles[0].raw_timestamp == start


def test_generated_mode_is_deterministic_across_calls() -> None:
    provider = MockMarketDataProvider()
    start = utc(2026, 1, 1, 0, 0)
    end = utc(2026, 1, 1, 0, 10)

    first_run = list(provider.get_historical_candles("EUR/USD", Timeframe.M1, start, end))
    second_run = list(provider.get_historical_candles("EUR/USD", Timeframe.M1, start, end))

    assert first_run == second_run


def test_fixed_mode_returns_exactly_the_supplied_candles() -> None:
    fixed = [
        ProviderCandle(raw_timestamp=utc(2026, 1, 1), raw_open=1.1, raw_high=1.2, raw_low=1.0, raw_close=1.15),
    ]
    provider = MockMarketDataProvider(fixed_candles=fixed)

    result = list(provider.get_historical_candles("EUR/USD", Timeframe.H1, utc(2026, 1, 1), utc(2026, 1, 2)))

    assert result == fixed


def test_provider_name_is_mock() -> None:
    assert MockMarketDataProvider().name == "mock"
