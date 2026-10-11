from __future__ import annotations

from datetime import datetime, timezone

from app.probability.models import ExitReason, OutcomeLabel, TradeOutcome
from app.regimes.context import RegimeContext
from app.regimes.models import StructureState, TrendState, VolatilityState


def ts(offset: int, hour: int = 0) -> datetime:
    """A deterministic timestamp `offset` days after 2024-01-01 (safely
    handles offsets beyond a single month, unlike constructing
    `datetime(2024, 1, offset)` directly)."""
    from datetime import timedelta

    return datetime(2024, 1, 1, hour, tzinfo=timezone.utc) + timedelta(days=offset)


def make_context(
    trend=TrendState.UPTREND, vol=VolatilityState.NORMAL, struct=StructureState.TRENDING, sessions=("london",),
    timestamp=None,
) -> RegimeContext:
    return RegimeContext(
        timestamp=timestamp or ts(1),
        instrument_symbol="EURUSD",
        timeframe="5M",
        trend_state=trend,
        volatility_state=vol,
        structure_state=struct,
        sessions=sessions,
    )


_UNSET = object()


def make_outcome(
    signal_ts: datetime,
    label: OutcomeLabel = OutcomeLabel.WIN,
    r: float | None = 1.0,
    strategy: str = "orb",
    instrument: str = "EURUSD",
    timeframe: str = "5M",
    direction: str = "long",
    regime_context=_UNSET,
    exit_ts: datetime | None = None,
) -> TradeOutcome:
    reason = {
        OutcomeLabel.WIN: ExitReason.TAKE_PROFIT,
        OutcomeLabel.LOSS: ExitReason.STOP_LOSS,
        OutcomeLabel.BREAKEVEN: ExitReason.TIME_EXPIRY,
        OutcomeLabel.UNRESOLVED: ExitReason.SAME_BAR_UNRESOLVED,
    }[label]
    resolved_context = make_context(timestamp=signal_ts) if regime_context is _UNSET else regime_context
    return TradeOutcome(
        strategy_name=strategy,
        instrument_symbol=instrument,
        timeframe=timeframe,
        direction=direction,
        signal_timestamp=signal_ts,
        entry_timestamp=signal_ts,
        exit_timestamp=exit_ts if exit_ts is not None else signal_ts,
        entry_price=100.0,
        exit_price=(100.0 + r) if r is not None else None,
        stop_loss=98.0,
        take_profit=104.0,
        r_multiple=r,
        outcome_label=label,
        exit_reason=reason,
        evaluation_horizon_bars=20,
        bars_to_exit=5,
        regime_context=resolved_context,
    )
