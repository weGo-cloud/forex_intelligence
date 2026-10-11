"""Persistent experiment registry.

M2's `experiment_log.py` (JSONL file) established that logging every run
is worth doing but is not a durable, queryable, first-class research
entity. This module is that upgrade: an `experiments` table in the same
PostgreSQL database M1 already uses, via the same SQLAlchemy/Base
convention as `app.data.models`.

Market candles remain owned by M1 - this table never stores OHLCV data,
only the experiment's configuration, hash, status, and result summary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from app.core.database import Base
from app.data.models import UTCDateTime  # reuse M1's UTC-enforcing column type - no reason to redefine it
from app.research.models import ExperimentDefinition, ExperimentStatus, ExperimentType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentImmutableError(Exception):
    """Raised when code attempts to modify a COMPLETED experiment's
    definition or config hash. Completed experiments are immutable by
    design (M3 spec section 18) - a changed configuration is a new
    experiment, optionally linked via `parent_experiment_id`."""


class ExperimentRecord(Base):
    """One persisted research experiment.

    `config_hash` is not unique-constrained: legitimately re-running the
    exact same configuration (e.g. to confirm reproducibility) is
    allowed and produces a second row with the same hash - `config_hash`
    identifies the CONFIGURATION, `experiment_id` identifies the RUN.
    """

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    experiment_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=ExperimentStatus.PENDING.value)

    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    instrument_symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    experiment_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tags: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)

    parent_experiment_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("experiments.experiment_id"), nullable=True, index=True
    )
    children: Mapped[list["ExperimentRecord"]] = relationship(
        "ExperimentRecord",
        backref="parent",
        remote_side=[experiment_id],
        foreign_keys=[parent_experiment_id],
    )

    result_summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow, onupdate=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)

    __table_args__ = (
        Index("ix_experiments_strategy_instrument", "strategy_name", "instrument_symbol"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"ExperimentRecord(experiment_id={self.experiment_id!r}, status={self.status!r})"


def _next_experiment_id(session: Session) -> str:
    """Sequential, human-referenceable IDs (exp_000001, ...) - same
    reasoning as M2's experiment_log.py: a researcher needs to say
    "look at exp_000042" out loud, a UUID makes that miserable. Uses a
    COUNT rather than an autoincrement-derived string so the ID format
    is stable regardless of the underlying `id` primary key sequence
    (e.g. after row deletion in a test database)."""
    count = session.query(ExperimentRecord).count()
    return f"exp_{count + 1:06d}"


class ExperimentRegistry:
    """Repository for creating, retrieving, and updating experiments.

    All mutation methods enforce immutability once an experiment is
    COMPLETED (M3 spec section 18): its `config`/`config_hash` can never
    change after completion. Re-running with different settings must go
    through `ExperimentDefinition` + a new `create()` call, optionally
    with `parent_experiment_id` set to link it.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, definition: ExperimentDefinition) -> ExperimentRecord:
        record = ExperimentRecord(
            experiment_id=_next_experiment_id(self.session),
            experiment_type=definition.experiment_type.value,
            status=ExperimentStatus.PENDING.value,
            strategy_name=definition.strategy_name,
            instrument_symbol=definition.instrument_symbol,
            config=definition.canonical_dict(),
            config_hash=definition.config_hash(),
            experiment_name=definition.experiment_name,
            tags=dict(definition.tags),
            parent_experiment_id=definition.parent_experiment_id,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        return self.session.query(ExperimentRecord).filter_by(experiment_id=experiment_id).one_or_none()

    def get_or_raise(self, experiment_id: str) -> ExperimentRecord:
        record = self.get(experiment_id)
        if record is None:
            raise KeyError(f"No experiment with id {experiment_id!r}")
        return record

    def list(
        self,
        strategy_name: Optional[str] = None,
        instrument_symbol: Optional[str] = None,
        experiment_type: Optional[ExperimentType] = None,
        status: Optional[ExperimentStatus] = None,
        parent_experiment_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[ExperimentRecord]:
        query = self.session.query(ExperimentRecord)
        if strategy_name is not None:
            query = query.filter(ExperimentRecord.strategy_name == strategy_name)
        if instrument_symbol is not None:
            query = query.filter(ExperimentRecord.instrument_symbol == instrument_symbol)
        if experiment_type is not None:
            query = query.filter(ExperimentRecord.experiment_type == experiment_type.value)
        if status is not None:
            query = query.filter(ExperimentRecord.status == status.value)
        if parent_experiment_id is not None:
            query = query.filter(ExperimentRecord.parent_experiment_id == parent_experiment_id)
        # Deterministic ordering: newest sequential ID last-created first is
        # useful for humans, but "deterministic" matters more than
        # "recency-first" for reproducible API responses - order by the
        # sequential ID itself, which is monotonic with creation order.
        return query.order_by(ExperimentRecord.experiment_id.asc()).limit(limit).all()

    def children(self, experiment_id: str) -> list[ExperimentRecord]:
        return (
            self.session.query(ExperimentRecord)
            .filter(ExperimentRecord.parent_experiment_id == experiment_id)
            .order_by(ExperimentRecord.experiment_id.asc())
            .all()
        )

    def mark_running(self, experiment_id: str) -> ExperimentRecord:
        record = self.get_or_raise(experiment_id)
        self._require_mutable(record)
        record.status = ExperimentStatus.RUNNING.value
        self.session.flush()
        return record

    def mark_completed(self, experiment_id: str, result_summary: dict[str, Any]) -> ExperimentRecord:
        record = self.get_or_raise(experiment_id)
        self._require_mutable(record)
        record.status = ExperimentStatus.COMPLETED.value
        record.result_summary = result_summary
        record.completed_at = _utcnow()
        self.session.flush()
        return record

    def mark_failed(self, experiment_id: str, reason: str) -> ExperimentRecord:
        """Persist useful failure information without leaving the
        experiment looking like a zero-metric success (M3 spec section
        15). `reason` should be a short, non-sensitive description -
        this is stored and may be returned over the public API, so raw
        stack traces / internal exception details should not be passed
        directly."""
        record = self.get_or_raise(experiment_id)
        self._require_mutable(record)
        record.status = ExperimentStatus.FAILED.value
        record.failure_reason = reason
        record.completed_at = _utcnow()
        self.session.flush()
        return record

    def _require_mutable(self, record: ExperimentRecord) -> None:
        if record.status == ExperimentStatus.COMPLETED.value:
            raise ExperimentImmutableError(
                f"Experiment {record.experiment_id!r} is COMPLETED and immutable; "
                "create a new experiment (optionally with parent_experiment_id set) instead."
            )
