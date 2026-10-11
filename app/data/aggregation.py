"""Derive higher-timeframe candles from lower-timeframe canonical candles.

Kept intentionally simple for M1: pure in-memory aggregation over a
list of `CanonicalCandle`, with no scheduling, caching, or persistence
concerns attached. Callers decide what to do with the result (store
it, hand it to a future feature/backtesting module, etc).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from app.data.schemas import CanonicalCandle
from app.data.timeframes import Timeframe


def aggregate_candles(candles: list[CanonicalCandle], to_timeframe: Timeframe) -> list[CanonicalCandle]:
    """Aggregate same-timeframe candles into `to_timeframe` candles.

    Buckets are aligned to UTC epoch boundaries (e.g. hourly buckets
    always start on the hour, UTC), so the result is correct regardless
    of what time the input range happens to start at, and never depends
    on a session/timezone assumption.

    Only complete buckets are emitted: if a bucket is missing one or
    more of its expected sub-candles (e.g. the input ends mid-hour, or
    there's a gap), it is silently dropped rather than emitted as a
    partial/misleading aggregate. This mirrors the same principle as
    gap detection: never manufacture data to paper over what's missing.
    """
    if not candles:
        return []

    from_timeframe = candles[0].timeframe
    if any(c.timeframe != from_timeframe for c in candles):
        raise ValueError("All input candles must share the same source timeframe")

    from_minutes = from_timeframe.minutes
    to_minutes = to_timeframe.minutes
    if to_minutes <= from_minutes or to_minutes % from_minutes != 0:
        raise ValueError(
            f"Cannot aggregate {from_timeframe.value} into {to_timeframe.value}: "
            "target timeframe must be a whole multiple of the source timeframe"
        )

    expected_members_per_bucket = to_minutes // from_minutes
    bucket_seconds = to_minutes * 60

    buckets: dict[datetime, list[CanonicalCandle]] = defaultdict(list)
    for candle in candles:
        epoch_seconds = int(candle.timestamp.timestamp())
        bucket_epoch = (epoch_seconds // bucket_seconds) * bucket_seconds
        bucket_start = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)
        buckets[bucket_start].append(candle)

    aggregated: list[CanonicalCandle] = []
    for bucket_start in sorted(buckets):
        members = sorted(buckets[bucket_start], key=lambda c: c.timestamp)
        if len(members) != expected_members_per_bucket:
            continue  # incomplete bucket - do not manufacture or guess missing data

        volumes = [m.volume for m in members if m.volume is not None]
        aggregated.append(
            CanonicalCandle(
                instrument_symbol=members[0].instrument_symbol,
                timeframe=to_timeframe,
                timestamp=bucket_start,
                open=members[0].open,
                high=max(m.high for m in members),
                low=min(m.low for m in members),
                close=members[-1].close,
                volume=sum(volumes) if volumes else None,
                source=members[0].source,
            )
        )

    return aggregated
