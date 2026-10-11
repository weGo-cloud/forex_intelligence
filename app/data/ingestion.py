"""Historical data ingestion pipeline.

Orchestrates: provider -> retrieve -> normalize -> validate -> persist
valid data -> record rejected/problematic records -> produce an
ingestion/data-quality result. This is the only module that wires the
provider, normalization, validation, and repository layers together —
each of those stays independently testable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.data.exceptions import NormalizationError
from app.data.models import IngestionStatus
from app.data.normalization import normalize_candle
from app.data.providers.base import MarketDataProvider
from app.data.repository import CandleRepository, IngestionRunRepository, InstrumentRepository
from app.data.schemas import CanonicalCandle, IngestionResult, InstrumentCreate, ValidationIssue
from app.data.timeframes import Timeframe
from app.data.validation import detect_gaps, find_duplicates_in_batch, is_sorted_by_timestamp, validate_candle

logger = get_logger(__name__)

# Provenance records store rejection reasons for investigation, but an
# unbounded ingestion failure (e.g. a vendor sending years of garbage)
# should not grow a single JSON column without limit.
_MAX_STORED_ISSUES = 200


def run_ingestion(
    session: Session,
    provider: MarketDataProvider,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    instrument_defaults: Optional[InstrumentCreate] = None,
) -> IngestionResult:
    """Run one provider -> normalize -> validate -> persist ingestion pass.

    Idempotent: re-running the same (instrument, timeframe, range)
    inserts no new rows for candles already stored - see
    `CandleRepository.bulk_insert`. Commits the session on success;
    callers running this inside a larger transaction should use their
    own session/scope.

    Raises `ValueError` if the instrument is unknown and
    `instrument_defaults` was not supplied to create it.
    """
    instrument_repo = InstrumentRepository(session)
    instrument = instrument_repo.get_by_symbol(symbol)
    if instrument is None:
        if instrument_defaults is None:
            raise ValueError(f"Unknown instrument {symbol!r} and no instrument_defaults provided to create it")
        instrument = instrument_repo.get_or_create(instrument_defaults)

    raw_candles = list(provider.get_historical_candles(symbol, timeframe, start, end))
    received = len(raw_candles)

    valid_candles: list[CanonicalCandle] = []
    rejection_reasons: list[ValidationIssue] = []
    rejected_records = 0

    for raw in raw_candles:
        try:
            canonical = normalize_candle(raw, symbol=symbol, timeframe=timeframe, source=provider.name)
        except NormalizationError as exc:
            rejection_reasons.append(ValidationIssue(code="normalization_error", message=str(exc)))
            rejected_records += 1
            continue

        # A single candle can fail more than one structural check at
        # once (e.g. high below low AND high not covering close). Every
        # issue is kept for the provenance report, but the record is
        # only counted once as rejected - see the invariant note below.
        candle_issues = validate_candle(canonical)
        if candle_issues:
            rejection_reasons.extend(candle_issues)
            rejected_records += 1
            continue

        valid_candles.append(canonical)

    if valid_candles and not is_sorted_by_timestamp(valid_candles):
        logger.info(
            "Ingestion for %s/%s received out-of-order candles; sorting before persistence.",
            symbol,
            timeframe.value,
        )
    valid_candles.sort(key=lambda c: c.timestamp)

    deduped_candles, in_batch_duplicates = find_duplicates_in_batch(valid_candles)

    candle_repo = CandleRepository(session)
    inserted, db_duplicates = candle_repo.bulk_insert(instrument.id, deduped_candles)
    total_duplicates = in_batch_duplicates + db_duplicates

    gaps = detect_gaps(deduped_candles, timeframe)

    accepted = inserted
    rejected = rejected_records
    # By construction: received == accepted + rejected + total_duplicates
    # (rejection_reasons may hold MORE entries than `rejected` records,
    # since one rejected candle can carry several distinct issues - the
    # count above is over records, the list below is over issues.)

    # Duplicates are treated as a normal, successful outcome of an
    # idempotent rerun - not a quality problem - so they never affect
    # status on their own. Only actual rejected records do.
    if received == 0:
        status = IngestionStatus.SUCCESS.value
    elif rejected == received:
        status = IngestionStatus.FAILED.value
    elif rejected > 0:
        status = IngestionStatus.PARTIAL.value
    else:
        status = IngestionStatus.SUCCESS.value

    stored_issues = [issue.model_dump(mode="json") for issue in rejection_reasons[:_MAX_STORED_ISSUES]]
    stored_issues.extend({"code": "gap", **gap.model_dump(mode="json")} for gap in gaps)

    run_repo = IngestionRunRepository(session)
    run = run_repo.create(
        provider_name=provider.name,
        instrument_id=instrument.id,
        timeframe=timeframe.value,
        requested_start=start,
        requested_end=end,
        received_count=received,
        accepted_count=accepted,
        rejected_count=rejected,
        duplicate_count=total_duplicates,
        status=status,
        issues=stored_issues,
    )
    session.commit()

    return IngestionResult(
        provider_name=provider.name,
        instrument_symbol=symbol,
        timeframe=timeframe,
        requested_start=start,
        requested_end=end,
        received=received,
        accepted=accepted,
        rejected=rejected,
        duplicates=total_duplicates,
        gaps=gaps,
        rejection_reasons=rejection_reasons,
        status=status,
        run_id=run.id,
    )
