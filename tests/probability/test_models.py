from __future__ import annotations

import json
from datetime import datetime, timezone

from app.probability.models import ExitReason, OutcomeLabel, TradeOutcome


def _outcome(**overrides) -> TradeOutcome:
    kwargs = dict(
        strategy_name="orb",
        instrument_symbol="EURUSD",
        timeframe="5M",
        direction="long",
        signal_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        entry_timestamp=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
        exit_timestamp=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        entry_price=100.0,
        exit_price=104.0,
        stop_loss=98.0,
        take_profit=104.0,
        r_multiple=2.0,
        outcome_label=OutcomeLabel.WIN,
        exit_reason=ExitReason.TAKE_PROFIT,
        evaluation_horizon_bars=20,
        bars_to_exit=5,
    )
    kwargs.update(overrides)
    return TradeOutcome(**kwargs)


class TestSerialization:
    def test_to_dict_is_json_safe(self) -> None:
        outcome = _outcome()
        payload = outcome.to_dict()
        json.dumps(payload)  # must not raise
        assert payload["outcome_label"] == "win"
        assert payload["exit_reason"] == "take_profit"

    def test_none_entry_serializes_to_none(self) -> None:
        outcome = _outcome(entry_timestamp=None, entry_price=None, exit_price=None, r_multiple=None,
                            outcome_label=OutcomeLabel.UNRESOLVED, exit_reason=ExitReason.INSUFFICIENT_DATA,
                            bars_to_exit=None)
        payload = outcome.to_dict()
        assert payload["entry_timestamp"] is None
        assert payload["entry_price"] is None

    def test_regime_context_none_serializes_to_none(self) -> None:
        outcome = _outcome(regime_context=None)
        assert outcome.to_dict()["regime_context"] is None

    def test_is_frozen(self) -> None:
        outcome = _outcome()
        try:
            outcome.entry_price = 999.0  # type: ignore[misc]
            assert False, "should have raised"
        except Exception:
            pass
