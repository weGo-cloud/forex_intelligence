"""Deterministic fake market-data provider.

Used to exercise the ingestion pipeline without depending on a real
vendor or network access — both in the automated test suite and for
manual local development.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional

from app.data.providers.base import MarketDataProvider, ProviderCandle
from app.data.timeframes import Timeframe


class MockMarketDataProvider(MarketDataProvider):
    """Fake provider with two modes:

    1. Fixed mode: constructed with an explicit list of `ProviderCandle`
       objects (including deliberately malformed ones), which it
       returns verbatim. Used by tests that need to exercise specific
       validation/normalization failure paths.
    2. Generated mode: constructed without fixed candles, it produces a
       deterministic, seed-free synthetic OHLCV sequence at regular
       intervals for the requested range. Used by tests and local
       development that just need "some plausible candles".
    """

    def __init__(
        self,
        fixed_candles: Optional[list[ProviderCandle]] = None,
        base_price: float = 1.10000,
    ) -> None:
        self._fixed_candles = fixed_candles
        self._base_price = base_price

    @property
    def name(self) -> str:
        return "mock"

    def get_historical_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> Iterable[ProviderCandle]:
        if self._fixed_candles is not None:
            return list(self._fixed_candles)
        return list(self._generate(timeframe, start, end))

    def _generate(self, timeframe: Timeframe, start: datetime, end: datetime) -> Iterable[ProviderCandle]:
        step = timedelta(minutes=timeframe.minutes)
        current = start
        price = self._base_price
        index = 0
        while current < end:
            # Small deterministic oscillation - no randomness, so test
            # assertions on exact values are stable across runs.
            delta = 0.0005 * ((index % 5) - 2)
            open_price = price
            close_price = round(price + delta, 5)
            high_price = round(max(open_price, close_price) + 0.0002, 5)
            low_price = round(min(open_price, close_price) - 0.0002, 5)
            volume = 100 + index

            yield ProviderCandle(
                raw_timestamp=current,
                raw_open=open_price,
                raw_high=high_price,
                raw_low=low_price,
                raw_close=close_price,
                raw_volume=volume,
            )

            price = close_price
            current += step
            index += 1
