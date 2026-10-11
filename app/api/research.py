"""Experiment management HTTP endpoints.

Thin wrapper over `app.research.experiment`/`registry`/`comparison`:
no business logic lives here, only request validation and translating
domain exceptions to HTTP status codes. Only SINGLE_BACKTEST experiments
can be created directly via POST /experiments - sweep/walk-forward/
robustness runs create many linked experiments and are orchestrated
functions, not (yet) a single HTTP call (out of scope for M3 V1's
"minimal API layer"; the underlying functions are available to a
future endpoint or script without any change).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.research.api_schemas import ComparisonRowResponse, ExperimentCreateRequest, ExperimentResponse
from app.research.comparison import IncompatibleExperimentsError, compare_experiments
from app.research.experiment import execute_experiment
from app.research.models import ExperimentDefinition, ExperimentStatus, ExperimentType
from app.research.registry import ExperimentRegistry

router = APIRouter(prefix="/experiments", tags=["research"])


def _definition_from_request(request: ExperimentCreateRequest) -> ExperimentDefinition:
    return ExperimentDefinition(
        strategy_name=request.strategy_name,
        strategy_params=request.strategy_params,
        instrument_symbol=request.instrument_symbol,
        timeframe=request.timeframe,
        test_period_start=request.test_period_start,
        test_period_end=request.test_period_end,
        initial_equity=request.initial_equity,
        risk_per_trade_pct=request.risk_per_trade_pct,
        spread_points=request.spread_points,
        commission_per_trade=request.commission_per_trade,
        same_bar_sl_tp_policy=request.same_bar_sl_tp_policy,
        max_holding_bars=request.max_holding_bars,
        one_trade_at_a_time=request.one_trade_at_a_time,
        allow_short=request.allow_short,
        context_timeframes=request.context_timeframes,
        experiment_name=request.experiment_name,
        tags=request.tags,
    )


@router.post("", response_model=ExperimentResponse, status_code=201)
def create_experiment(request: ExperimentCreateRequest, db: Session = Depends(get_db)) -> ExperimentResponse:
    """Create and immediately run a SINGLE_BACKTEST experiment."""
    try:
        definition = _definition_from_request(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    registry = ExperimentRegistry(db)
    outcome = execute_experiment(db, registry, definition)
    db.commit()
    return ExperimentResponse.model_validate(outcome.record)


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    strategy_name: str | None = None,
    instrument_symbol: str | None = None,
    experiment_type: ExperimentType | None = None,
    status: ExperimentStatus | None = None,
    parent_experiment_id: str | None = None,
    limit: int = Query(default=100, le=1000),
    db: Session = Depends(get_db),
) -> list[ExperimentResponse]:
    registry = ExperimentRegistry(db)
    records = registry.list(
        strategy_name=strategy_name,
        instrument_symbol=instrument_symbol,
        experiment_type=experiment_type,
        status=status,
        parent_experiment_id=parent_experiment_id,
        limit=limit,
    )
    return [ExperimentResponse.model_validate(r) for r in records]


@router.get("/compare", response_model=list[ComparisonRowResponse])
def compare(ids: list[str] = Query(..., alias="id"), db: Session = Depends(get_db)) -> list[ComparisonRowResponse]:
    """Compare experiments by ID: `?id=exp_000001&id=exp_000002`."""
    registry = ExperimentRegistry(db)
    try:
        rows = compare_experiments(registry, ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IncompatibleExperimentsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return [ComparisonRowResponse(**row.__dict__) for row in rows]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)) -> ExperimentResponse:
    registry = ExperimentRegistry(db)
    try:
        record = registry.get_or_raise(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ExperimentResponse.model_validate(record)


@router.get("/{experiment_id}/results")
def get_results(experiment_id: str, db: Session = Depends(get_db)) -> dict:
    registry = ExperimentRegistry(db)
    try:
        record = registry.get_or_raise(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "experiment_id": record.experiment_id,
        "status": record.status,
        "result_summary": record.result_summary,
        "failure_reason": record.failure_reason,
    }


@router.get("/{experiment_id}/children", response_model=list[ExperimentResponse])
def get_children(experiment_id: str, db: Session = Depends(get_db)) -> list[ExperimentResponse]:
    registry = ExperimentRegistry(db)
    try:
        registry.get_or_raise(experiment_id)  # 404 if the parent itself doesn't exist
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [ExperimentResponse.model_validate(r) for r in registry.children(experiment_id)]
