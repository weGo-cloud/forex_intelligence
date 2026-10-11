"""Vendor-agnostic market-data provider abstraction.

The rest of the application (ingestion pipeline, and eventually
strategy/backtesting code) depends only on `MarketDataProvider`, never
on a specific commercial vendor's SDK or response format. Each
concrete provider is responsible for translating its vendor's response
shape into a list of `ProviderCandle` objects; everything downstream of
that (normalization, validation, persistence) is vendor-independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from app.data.timeframes import Timeframe


@dataclass(frozen=True)
class ProviderCandle:
    """A single candle exactly as returned by a market-data vendor.

    Fields are deliberately loosely typed (`Any`): vendors disagree on
    whether timestamps are epoch seconds, epoch milliseconds, or ISO
    strings, and whether prices are floats or strings. Converting this
    into a well-typed `CanonicalCandle` is normalization's job
    (see app/data/normalization.py), not this dataclass's.
    """

    raw_timestamp: Any
    raw_open: Any
    raw_high: Any
    raw_low: Any
    raw_close: Any
    raw_volume: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(ABC):
    """Interface every market-data vendor integration must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider, recorded in provenance data."""
        raise NotImplementedError

    @abstractmethod
    def get_historical_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> Iterable[ProviderCandle]:
        """Return raw historical candles for `symbol`/`timeframe` in `[start, end)`.

        `start` and `end` are UTC-aware datetimes. Implementations
        should not raise on an empty result — an empty iterable is a
        valid, non-error response (e.g. no data for that range yet).
        """
        raise NotImplementedError
