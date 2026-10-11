"""Data-quality checks applied to canonical candles before persistence.

Three distinct concerns live here, deliberately kept separate:

- `validate_candle`: per-candle structural correctness (impossible OHLC
  relationships, non-positive prices). A candle that fails this is
  rejected — it never becomes trusted canonical data.
- `find_duplicates_in_batch` / out-of-order detection: batch-level
  bookkeeping needed for idempotent ingestion.
- `detect_gaps`: REPORTING only. A gap is never a reason to reject or
  manufacture data — legitimate market closures (weekends, holidays)
  produce gaps routinely. This exists purely so gaps can be
  investigated later.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from app.data.schemas import CanonicalCandle, GapIssue, ValidationIssue
from app.data.timeframes import Timeframe

# A gap is flagged once it exceeds this multiple of the timeframe's
# nominal duration. > 1.0 so that ordinary provider timing jitter (a
# few seconds either side of the exact boundary) never triggers a
# false positive.
_DEFAULT_GAP_MULTIPLE_THRESHOLD = 1.5


def validate_candle(candle: CanonicalCandle) -> list[ValidationIssue]:
    """Return structural problems with a single candle (empty = valid).

    Checks OHLC relationships and price/volume sign only. Does not
    check for duplicates or gaps — those are batch-level concerns
    handled separately.
    """
    issues: list[ValidationIssue] = []

    for field_name in ("open", "high", "low", "close"):
        value: Decimal = getattr(candle, field_name)
        if value <= 0:
            issues.append(
                ValidationIssue(
                    code="non_positive_price",
                    message=f"{field_name} must be positive, got {value}",
                    candle_timestamp=candle.timestamp,
                )
            )

    if candle.volume is not None and candle.volume < 0:
        issues.append(
            ValidationIssue(
                code="negative_volume",
                message=f"volume must not be negative, got {candle.volume}",
                candle_timestamp=candle.timestamp,
            )
        )

    if candle.high < candle.low:
        issues.append(
            ValidationIssue(
                code="high_below_low",
                message=f"high ({candle.high}) is below low ({candle.low})",
                candle_timestamp=candle.timestamp,
            )
        )

    if candle.high < candle.open or candle.high < candle.close:
        issues.append(
            ValidationIssue(
                code="high_not_extreme",
                message=f"high ({candle.high}) must be >= open ({candle.open}) and >= close ({candle.close})",
                candle_timestamp=candle.timestamp,
            )
        )

    if candle.low > candle.open or candle.low > candle.close:
        issues.append(
            ValidationIssue(
                code="low_not_extreme",
                message=f"low ({candle.low}) must be <= open ({candle.open}) and <= close ({candle.close})",
                candle_timestamp=candle.timestamp,
            )
        )

    return issues


def find_duplicates_in_batch(candles: list[CanonicalCandle]) -> tuple[list[CanonicalCandle], int]:
    """Deduplicate a batch by (instrument, timeframe, timestamp).

    Returns (deduplicated candles, duplicate count). The first
    occurrence of a given key is kept; later ones are counted as
    duplicates. This only handles duplicates *within* the incoming
    batch — duplicates against already-stored data are handled by the
    repository layer at persistence time.
    """
    seen: set[tuple[str, Timeframe, object]] = set()
    unique: list[CanonicalCandle] = []
    duplicate_count = 0
    for candle in candles:
        key = (candle.instrument_symbol, candle.timeframe, candle.timestamp)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(candle)
    return unique, duplicate_count


def is_sorted_by_timestamp(candles: list[CanonicalCandle]) -> bool:
    """Return True if `candles` is already sorted ascending by timestamp."""
    return all(candles[i].timestamp <= candles[i + 1].timestamp for i in range(len(candles) - 1))


def detect_gaps(
    sorted_candles: list[CanonicalCandle],
    timeframe: Timeframe,
    gap_multiple_threshold: float = _DEFAULT_GAP_MULTIPLE_THRESHOLD,
) -> list[GapIssue]:
    """Flag intervals between consecutive candles larger than expected.

    Input must already be sorted ascending by timestamp. This never
    rejects or fills data — it only produces a report. A gap over a
    weekend or holiday is expected and legitimate; distinguishing that
    from a genuinely missing/corrupt range requires session-calendar
    knowledge this milestone deliberately does not implement (see
    docs/M1_DATA_FOUNDATION.md).
    """
    issues: list[GapIssue] = []
    expected = timedelta(minutes=timeframe.minutes)
    threshold = expected * gap_multiple_threshold

    for previous, current in zip(sorted_candles, sorted_candles[1:]):
        delta = current.timestamp - previous.timestamp
        if delta <= timedelta(0):
            continue  # non-positive deltas are duplicate/out-of-order concerns, not gaps
        if delta > threshold:
            missing_periods = max(int(delta / expected) - 1, 1)
            issues.append(
                GapIssue(
                    after=previous.timestamp,
                    before=current.timestamp,
                    expected_interval_minutes=timeframe.minutes,
                    missing_periods=missing_periods,
                )
            )
    return issues
