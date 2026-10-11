"""Canonical experiment specification.

An `ExperimentDefinition` captures everything needed to reproduce a
research run: strategy identity/parameters, instrument, timeframe, test
period, cost/execution assumptions, and experiment type. It is
deliberately NOT coupled to any concrete strategy (no ORB/RSI-specific
fields) - strategy identity is a name + a plain parameter dict, exactly
like `app.strategies.registry` already treats strategies.

Reproducibility rests on two properties, both enforced here:

1. Deterministic serialization: `canonical_json()` always produces the
   same bytes for equivalent configurations, regardless of dict
   insertion order (achieved via `json.dumps(..., sort_keys=True)` on a
   fully-normalized structure - see `_normalize`).
2. Deterministic hashing: `config_hash()` is a SHA-256 digest of that
   canonical JSON. Equivalent configurations hash identically;
   materially different ones (almost) never collide.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


class ExperimentType(str, enum.Enum):
    """The kind of research run an experiment represents.

    All experiment types ultimately produce a structured
    `ExperimentResultSummary` (see results in registry.py) - there is no
    separate incompatible result shape per type, so comparison and the
    future ranking engine can treat them uniformly.
    """

    SINGLE_BACKTEST = "single_backtest"
    WALK_FORWARD = "walk_forward"
    PARAMETER_SWEEP = "parameter_sweep"
    ROBUSTNESS_ANALYSIS = "robustness_analysis"


class ExperimentStatus(str, enum.Enum):
    """Explicit experiment lifecycle. A failed experiment must never be
    indistinguishable from a successful one with zero metrics."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC); naive datetimes are rejected")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ExperimentDefinition:
    """A canonical, reproducible specification of one research run.

    `parent_experiment_id` links a child (e.g. one parameter-sweep
    combination, or one walk-forward window) back to the experiment
    that spawned it, forming the provenance chain required by section
    12 of the M3 spec: Experiment -> Configuration -> Data selection ->
    Strategy -> Execution engine -> Metrics.
    """

    strategy_name: str
    strategy_params: dict[str, Any]
    instrument_symbol: str
    timeframe: str  # Timeframe.value, e.g. "5M" - kept as plain str so this
    # module never imports app.data.timeframes (no coupling to M1 internals)
    test_period_start: datetime
    test_period_end: datetime

    # --- Backtest execution assumptions (mirrors BacktestConfig's
    # reproducibility-relevant fields; kept as plain values here rather
    # than importing BacktestConfig so this module has zero dependency
    # on app.backtesting) ---
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    spread_points: float = 0.0
    commission_per_trade: float = 0.0
    same_bar_sl_tp_policy: str = "stop_loss_first"
    max_holding_bars: Optional[int] = None
    one_trade_at_a_time: bool = True
    allow_short: bool = True

    data_source: str = "m1_postgres"
    execution_engine: str = "custom_bar_simulator"
    experiment_type: ExperimentType = ExperimentType.SINGLE_BACKTEST

    context_timeframes: dict[str, str] = field(default_factory=dict)  # e.g. {"1H": "1H"}

    experiment_name: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)
    parent_experiment_id: Optional[str] = None

    created_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        object.__setattr__(self, "test_period_start", _require_utc_aware(self.test_period_start, "test_period_start"))
        object.__setattr__(self, "test_period_end", _require_utc_aware(self.test_period_end, "test_period_end"))
        object.__setattr__(self, "created_at", _require_utc_aware(self.created_at, "created_at"))
        if self.test_period_end <= self.test_period_start:
            raise ValueError("test_period_end must be after test_period_start")
        if not isinstance(self.experiment_type, ExperimentType):
            object.__setattr__(self, "experiment_type", ExperimentType(self.experiment_type))

    def with_overrides(self, **overrides: Any) -> "ExperimentDefinition":
        """Return a new definition with fields overridden - used by the
        sweep/walk-forward/robustness engines to derive child
        definitions without mutating the parent (definitions are frozen,
        so this is the only supported way to derive a variant)."""
        data = asdict(self)
        data["experiment_type"] = self.experiment_type  # asdict() would stringify the enum; keep it typed
        data.update(overrides)
        return ExperimentDefinition(**data)

    def canonical_dict(self) -> dict[str, Any]:
        """A fully JSON-safe, deterministically-ordered representation
        used for both hashing and storage. `created_at` and
        `parent_experiment_id` are deliberately EXCLUDED: two runs of
        the identical configuration submitted at different times (or
        as children of different parents in a re-run) must hash
        identically, since config_hash answers "is this the same
        experiment configuration", not "is this the same database row".
        `experiment_name`/`tags` are excluded for the same reason -
        they are labels, not configuration."""
        return {
            "strategy_name": self.strategy_name,
            "strategy_params": _normalize(self.strategy_params),
            "instrument_symbol": self.instrument_symbol,
            "timeframe": self.timeframe,
            "test_period_start": self.test_period_start.isoformat(),
            "test_period_end": self.test_period_end.isoformat(),
            "initial_equity": self.initial_equity,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "spread_points": self.spread_points,
            "commission_per_trade": self.commission_per_trade,
            "same_bar_sl_tp_policy": self.same_bar_sl_tp_policy,
            "max_holding_bars": self.max_holding_bars,
            "one_trade_at_a_time": self.one_trade_at_a_time,
            "allow_short": self.allow_short,
            "data_source": self.data_source,
            "execution_engine": self.execution_engine,
            "experiment_type": self.experiment_type.value,
            "context_timeframes": _normalize(self.context_timeframes),
        }

    def canonical_json(self) -> str:
        """Deterministic JSON: sorted keys, no whitespace variance."""
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        """SHA-256 hex digest of `canonical_json()`. Stable across
        process restarts and dict insertion order; changes whenever any
        materially-relevant field changes."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _normalize(value: Any) -> Any:
    """Recursively normalize a structure so it serializes identically
    regardless of input ordering or non-JSON-native types (e.g. a stray
    enum member in strategy_params). Dict keys are sorted at every
    level (on top of `json.dumps(sort_keys=True)`, which alone would be
    sufficient - this belt-and-braces pass also stringifies values
    `json.dumps` cannot natively handle, e.g. enum members, so an odd
    parameter type never crashes hashing)."""
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
