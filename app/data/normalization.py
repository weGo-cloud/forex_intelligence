"""Provider-specific raw data -> canonical candle normalization.

This is the only place that should ever parse a provider's raw
timestamp/price representation. Everything downstream (validation,
persistence, future strategy/backtesting code) works exclusively with
`CanonicalCandle`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from app.data.exceptions import NormalizationError
from app.data.providers.base import ProviderCandle
from app.data.schemas import CanonicalCandle
from app.data.timeframes import Timeframe

# Heuristic cutoff for distinguishing epoch seconds from epoch
# milliseconds: seconds since epoch won't exceed this many digits until
# the year 2286, and milliseconds since epoch already exceed it today.
_EPOCH_MS_THRESHOLD = 10_000_000_000


def _to_decimal(value: Any, field_name: str) -> Decimal:
    if value is None:
        raise NormalizationError(f"{field_name} is missing")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise NormalizationError(f"{field_name} is not a valid number: {value!r}") from exc


def _to_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        seconds = value / 1000 if value > _EPOCH_MS_THRESHOLD else value
        parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise NormalizationError(f"timestamp is not a parseable ISO-8601 string: {value!r}") from exc
    else:
        raise NormalizationError(f"Unsupported timestamp type: {type(value).__name__}")

    if parsed.tzinfo is None:
        # Refuse to guess. A provider that sends naive timestamps must
        # be fixed at the provider layer to attach an explicit offset -
        # silently assuming UTC here is exactly the kind of "hard-coded
        # timezone assumption" this milestone is required to avoid.
        raise NormalizationError("timestamp has no timezone information; refusing to assume one")
    return parsed.astimezone(timezone.utc)


def normalize_candle(
    raw: ProviderCandle,
    symbol: str,
    timeframe: Timeframe,
    source: str,
) -> CanonicalCandle:
    """Convert one raw provider candle into canonical form.

    Raises `NormalizationError` for any raw record that cannot be
    converted (bad timestamp, non-numeric price, missing field). This
    is expected to happen for genuinely malformed vendor data; callers
    (the ingestion pipeline) catch it and record a rejected record
    rather than letting it abort the whole batch.
    """
    timestamp = _to_utc_datetime(raw.raw_timestamp)
    open_price = _to_decimal(raw.raw_open, "open")
    high_price = _to_decimal(raw.raw_high, "high")
    low_price = _to_decimal(raw.raw_low, "low")
    close_price = _to_decimal(raw.raw_close, "close")
    volume = None if raw.raw_volume is None else _to_decimal(raw.raw_volume, "volume")

    return CanonicalCandle(
        instrument_symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        source=source,
    )
