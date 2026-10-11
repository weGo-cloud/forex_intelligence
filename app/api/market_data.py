"""Read-only market-data HTTP endpoints.

Deliberately query-only for M1: ingestion is triggered by application
code (see app/data/ingestion.py) or a future scheduled job, not by an
HTTP request, so there is no POST endpoint here that could kick off a
provider fetch on demand.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.data.repository import CandleRepository, InstrumentRepository
from app.data.schemas import CandleRead, InstrumentRead
from app.data.timeframes import Timeframe

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/instruments", response_model=list[InstrumentRead])
def list_instruments(db: Session = Depends(get_db)) -> list[InstrumentRead]:
    """List currently active instruments."""
    repo = InstrumentRepository(db)
    return [InstrumentRead.model_validate(instrument) for instrument in repo.list_active()]


@router.get("/candles", response_model=list[CandleRead])
def get_candles(
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    db: Session = Depends(get_db),
) -> list[CandleRead]:
    """Return stored candles for an instrument/timeframe/date range."""
    instrument_repo = InstrumentRepository(db)
    instrument = instrument_repo.get_by_symbol(symbol)
    if instrument is None:
        raise HTTPException(status_code=404, detail=f"Unknown instrument: {symbol}")

    candle_repo = CandleRepository(db)
    candles = candle_repo.query(instrument.id, timeframe, start, end)
    return [CandleRead.model_validate(candle) for candle in candles]
