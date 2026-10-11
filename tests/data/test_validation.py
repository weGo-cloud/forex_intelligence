"""Tests for app.data.validation."""

from datetime import timedelta
from decimal import Decimal

from app.data.schemas import CanonicalCandle
from app.data.timeframes import Timeframe
from app.data.validation import (
    detect_gaps,
    find_duplicates_in_batch,
    is_sorted_by_timestamp,
    validate_candle,
)
from tests.data.conftest import utc


def make_candle(**overrides) -> CanonicalCandle:
    defaults = dict(
        instrument_symbol="EUR/USD",
        timeframe=Timeframe.M1,
        timestamp=utc(2026, 1, 1, 12, 0),
        open=Decimal("1.1000"),
        high=Decimal("1.1010"),
        low=Decimal("1.0990"),
        close=Decimal("1.1005"),
        volume=Decimal("1000"),
        source="mock",
    )
    defaults.update(overrides)
    return CanonicalCandle(**defaults)


def test_valid_candle_has_no_issues() -> None:
    assert validate_candle(make_candle()) == []


def test_high_below_low_is_rejected() -> None:
    issues = validate_candle(make_candle(high=Decimal("1.0"), low=Decimal("1.1")))
    codes = {issue.code for issue in issues}
    assert "high_below_low" in codes


def test_high_not_covering_open_or_close_is_rejected() -> None:
    issues = validate_candle(make_candle(high=Decimal("1.0995"), open=Decimal("1.1000")))
    codes = {issue.code for issue in issues}
    assert "high_not_extreme" in codes


def test_low_not_covering_open_or_close_is_rejected() -> None:
    issues = validate_candle(make_candle(low=Decimal("1.1002"), close=Decimal("1.1000")))
    codes = {issue.code for issue in issues}
    assert "low_not_extreme" in codes


def test_non_positive_price_is_rejected() -> None:
    issues = validate_candle(make_candle(open=Decimal("0")))
    codes = {issue.code for issue in issues}
    assert "non_positive_price" in codes


def test_negative_volume_is_rejected() -> None:
    issues = validate_candle(make_candle(volume=Decimal("-1")))
    codes = {issue.code for issue in issues}
    assert "negative_volume" in codes


def test_none_volume_is_allowed() -> None:
    # Spot forex has no centralized volume; None must not be flagged.
    assert validate_candle(make_candle(volume=None)) == []


def test_find_duplicates_in_batch_keeps_first_and_counts_rest() -> None:
    t = utc(2026, 1, 1, 12, 0)
    candles = [make_candle(timestamp=t), make_candle(timestamp=t), make_candle(timestamp=t)]

    unique, duplicate_count = find_duplicates_in_batch(candles)

    assert len(unique) == 1
    assert duplicate_count == 2


def test_is_sorted_by_timestamp() -> None:
    ascending = [make_candle(timestamp=utc(2026, 1, 1, 12, i)) for i in range(3)]
    assert is_sorted_by_timestamp(ascending) is True

    descending = list(reversed(ascending))
    assert is_sorted_by_timestamp(descending) is False


def test_detect_gaps_flags_interval_larger_than_threshold() -> None:
    base = utc(2026, 1, 1, 12, 0)
    candles = [
        make_candle(timestamp=base),
        make_candle(timestamp=base + timedelta(minutes=1)),
        # Large jump: several missing 1-minute candles.
        make_candle(timestamp=base + timedelta(minutes=30)),
    ]

    gaps = detect_gaps(candles, Timeframe.M1)

    assert len(gaps) == 1
    assert gaps[0].after == base + timedelta(minutes=1)
    assert gaps[0].before == base + timedelta(minutes=30)
    assert gaps[0].missing_periods == 28


def test_detect_gaps_does_not_flag_normal_consecutive_candles() -> None:
    base = utc(2026, 1, 1, 12, 0)
    candles = [make_candle(timestamp=base + timedelta(minutes=i)) for i in range(5)]

    assert detect_gaps(candles, Timeframe.M1) == []


def test_detect_gaps_on_daily_timeframe_flags_weekend_style_jump_but_does_not_reject() -> None:
    # A ~3 day jump on a 1D timeframe is a legitimate weekend-style gap.
    # detect_gaps only reports it - it never rejects or drops candles.
    friday = utc(2026, 1, 2, 0, 0)
    monday = friday + timedelta(days=3)
    candles = [make_candle(timeframe=Timeframe.D1, timestamp=friday), make_candle(timeframe=Timeframe.D1, timestamp=monday)]

    gaps = detect_gaps(candles, Timeframe.D1)

    assert len(gaps) == 1
    # Reporting-only: the input candle list itself is never mutated,
    # filled in, or reduced because of a detected gap.
    assert len(candles) == 2
