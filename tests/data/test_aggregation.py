"""Tests for app.data.aggregation."""

from decimal import Decimal

import pytest

from app.data.aggregation import aggregate_candles
from app.data.schemas import CanonicalCandle
from app.data.timeframes import Timeframe
from tests.data.conftest import utc


def make_m1_candle(minute: int, open_: str, high: str, low: str, close: str, volume: str = "10") -> CanonicalCandle:
    return CanonicalCandle(
        instrument_symbol="EUR/USD",
        timeframe=Timeframe.M1,
        timestamp=utc(2026, 1, 1, 0, minute),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        source="mock",
    )


def test_aggregates_five_1m_candles_into_one_5m_candle() -> None:
    candles = [
        make_m1_candle(0, "1.10", "1.12", "1.09", "1.11"),
        make_m1_candle(1, "1.11", "1.13", "1.10", "1.115"),
        make_m1_candle(2, "1.115", "1.116", "1.105", "1.108"),
        make_m1_candle(3, "1.108", "1.14", "1.10", "1.135"),
        make_m1_candle(4, "1.135", "1.136", "1.12", "1.125"),
    ]

    result = aggregate_candles(candles, Timeframe.M5)

    assert len(result) == 1
    bucket = result[0]
    assert bucket.timestamp == utc(2026, 1, 1, 0, 0)
    assert bucket.open == Decimal("1.10")  # first candle's open
    assert bucket.close == Decimal("1.125")  # last candle's close
    assert bucket.high == Decimal("1.14")  # max high across the bucket
    assert bucket.low == Decimal("1.09")  # min low across the bucket
    assert bucket.volume == Decimal("50")  # sum of volumes


def test_incomplete_trailing_bucket_is_dropped_not_manufactured() -> None:
    # Only 3 of the 5 expected 1M candles for the 00:00-00:05 bucket.
    candles = [make_m1_candle(i, "1.10", "1.11", "1.09", "1.10") for i in range(3)]

    result = aggregate_candles(candles, Timeframe.M5)

    assert result == []


def test_aggregation_across_multiple_complete_buckets() -> None:
    candles = [make_m1_candle(i, "1.10", "1.11", "1.09", "1.10") for i in range(10)]

    result = aggregate_candles(candles, Timeframe.M5)

    assert len(result) == 2
    assert result[0].timestamp == utc(2026, 1, 1, 0, 0)
    assert result[1].timestamp == utc(2026, 1, 1, 0, 5)


def test_bucket_boundaries_are_aligned_to_utc_epoch_not_input_start() -> None:
    # Input starts at minute 3, not on a 5-minute boundary. Only the
    # candles that complete an aligned 00:00-00:05 or 00:05-00:10 bucket
    # should produce output.
    candles = [make_m1_candle(minute, "1.10", "1.11", "1.09", "1.10") for minute in range(3, 10)]

    result = aggregate_candles(candles, Timeframe.M5)

    # minutes 3,4 alone don't complete the 00:00 bucket (missing 0,1,2) -> dropped
    # minutes 5,6,7,8,9 complete the 00:05 bucket -> emitted
    assert len(result) == 1
    assert result[0].timestamp == utc(2026, 1, 1, 0, 5)


def test_rejects_non_multiple_target_timeframe() -> None:
    candles = [make_m1_candle(0, "1.10", "1.11", "1.09", "1.10")]
    with pytest.raises(ValueError):
        aggregate_candles(candles, Timeframe.M1)  # same timeframe, not a valid target


def test_rejects_mixed_source_timeframes() -> None:
    m1 = make_m1_candle(0, "1.10", "1.11", "1.09", "1.10")
    h1 = CanonicalCandle(
        instrument_symbol="EUR/USD",
        timeframe=Timeframe.H1,
        timestamp=utc(2026, 1, 1, 1, 0),
        open=Decimal("1.10"),
        high=Decimal("1.11"),
        low=Decimal("1.09"),
        close=Decimal("1.10"),
    )

    with pytest.raises(ValueError):
        aggregate_candles([m1, h1], Timeframe.H1)


def test_empty_input_returns_empty_output() -> None:
    assert aggregate_candles([], Timeframe.M5) == []
