from __future__ import annotations

from datetime import datetime, timezone

from app.probability.filters import OutcomeFilter
from app.probability.models import ExitReason, OutcomeLabel, TradeOutcome
from app.probability.outcomes import OutcomeDataset
from app.regimes.context import RegimeContext
from app.regimes.models import StructureState, TrendState, VolatilityState

TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _ctx(trend=TrendState.UPTREND, vol=VolatilityState.HIGH, struct=StructureState.TRENDING, sessions=("london",)):
    return RegimeContext(
        timestamp=TS, instrument_symbol="EURUSD", timeframe="5M",
        trend_state=trend, volatility_state=vol, structure_state=struct, sessions=sessions,
    )


def _outcome(**overrides) -> TradeOutcome:
    kwargs = dict(
        strategy_name="orb", instrument_symbol="EURUSD", timeframe="5M", direction="long",
        signal_timestamp=TS, entry_timestamp=TS, exit_timestamp=TS,
        entry_price=100.0, exit_price=104.0, stop_loss=98.0, take_profit=104.0,
        r_multiple=2.0, outcome_label=OutcomeLabel.WIN, exit_reason=ExitReason.TAKE_PROFIT,
        evaluation_horizon_bars=20, bars_to_exit=5, regime_context=_ctx(),
    )
    kwargs.update(overrides)
    return TradeOutcome(**kwargs)


class TestBasicFields:
    def test_matches_strategy_name(self) -> None:
        outcome = _outcome(strategy_name="orb")
        assert OutcomeFilter(strategy_name="orb").matches(outcome)
        assert not OutcomeFilter(strategy_name="rsi_divergence").matches(outcome)

    def test_matches_instrument_and_timeframe(self) -> None:
        outcome = _outcome(instrument_symbol="EURUSD", timeframe="5M")
        assert OutcomeFilter(instrument_symbol="EURUSD", timeframe="5M").matches(outcome)
        assert not OutcomeFilter(instrument_symbol="GBPUSD").matches(outcome)

    def test_matches_direction(self) -> None:
        outcome = _outcome(direction="long")
        assert OutcomeFilter(direction="long").matches(outcome)
        assert not OutcomeFilter(direction="short").matches(outcome)

    def test_empty_filter_matches_everything(self) -> None:
        assert OutcomeFilter().matches(_outcome())


class TestRegimeFields:
    def test_matches_trend_state(self) -> None:
        outcome = _outcome(regime_context=_ctx(trend=TrendState.UPTREND))
        assert OutcomeFilter(trend_state=TrendState.UPTREND).matches(outcome)
        assert not OutcomeFilter(trend_state=TrendState.DOWNTREND).matches(outcome)

    def test_matches_volatility_and_structure(self) -> None:
        outcome = _outcome(regime_context=_ctx(vol=VolatilityState.HIGH, struct=StructureState.TRENDING))
        assert OutcomeFilter(volatility_state=VolatilityState.HIGH, structure_state=StructureState.TRENDING).matches(outcome)

    def test_matches_session_by_membership(self) -> None:
        outcome = _outcome(regime_context=_ctx(sessions=("london", "new_york")))
        assert OutcomeFilter(session="london").matches(outcome)
        assert OutcomeFilter(session="new_york").matches(outcome)
        assert not OutcomeFilter(session="tokyo").matches(outcome)

    def test_missing_regime_context_fails_regime_filter(self) -> None:
        outcome = _outcome(regime_context=None)
        assert not OutcomeFilter(trend_state=TrendState.UPTREND).matches(outcome)

    def test_missing_regime_context_passes_non_regime_filter(self) -> None:
        outcome = _outcome(regime_context=None, strategy_name="orb")
        assert OutcomeFilter(strategy_name="orb").matches(outcome)


class TestCompositeFilter:
    def test_all_conditions_must_match(self) -> None:
        outcome = _outcome(strategy_name="orb", instrument_symbol="EURUSD", regime_context=_ctx(trend=TrendState.UPTREND))
        f = OutcomeFilter(strategy_name="orb", instrument_symbol="EURUSD", trend_state=TrendState.UPTREND)
        assert f.matches(outcome)
        f2 = OutcomeFilter(strategy_name="orb", instrument_symbol="EURUSD", trend_state=TrendState.DOWNTREND)
        assert not f2.matches(outcome)


class TestApply:
    def test_apply_returns_filtered_dataset(self) -> None:
        dataset = OutcomeDataset.from_list([
            _outcome(strategy_name="orb"),
            _outcome(strategy_name="rsi_divergence"),
        ])
        result = OutcomeFilter(strategy_name="orb").apply(dataset)
        assert len(result) == 1
        assert result[0].strategy_name == "orb"
