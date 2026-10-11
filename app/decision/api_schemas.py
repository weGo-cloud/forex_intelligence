"""Pydantic request/response schemas for the setups API, kept separate
from domain models (same convention as `app.research.api_schemas`)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ObservationPayload(BaseModel):
    observation_type: str
    instrument_symbol: str
    timeframe: str
    timestamp: datetime
    source: str
    direction: Optional[str] = None
    price: Optional[float] = None
    zone_low: Optional[float] = None
    zone_high: Optional[float] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectRequest(BaseModel):
    instrument_symbol: str
    timeframe: str
    observations: list[ObservationPayload]
    as_of: Optional[datetime] = None


class EvaluateRequest(DetectRequest):
    at: datetime
    evidence: Optional[dict[str, Any]] = None
