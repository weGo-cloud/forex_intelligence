from __future__ import annotations

from app.evaluation.evaluator import StrategyEvaluator
from app.probability.filters import OutcomeFilter
from app.probability.outcomes import OutcomeDataset
from app.regimes.models import StructureState, TrendState, VolatilityState
from tests.evaluation.conftest import make_context, make_outcome, ts


class TestTrendFiltering:
    def test_filters_by_trend_state(self) -> None:
        up = [make_outcome(ts(i), regime_context=make_context(trend=TrendState.UPTREND)) for i in range(10)]
        down = [make_outcome(ts(20 + i), regime_context=make_context(trend=TrendState.DOWNTREND)) for i in range(5)]
        dataset = OutcomeDataset.from_list(up + down)

        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(trend_state=TrendState.UPTREND))
        assert result.probability.counts.total_observations == 10


class TestVolatilityFiltering:
    def test_filters_by_volatility_state(self) -> None:
        high = [make_outcome(ts(i), regime_context=make_context(vol=VolatilityState.HIGH)) for i in range(7)]
        low = [make_outcome(ts(20 + i), regime_context=make_context(vol=VolatilityState.LOW)) for i in range(3)]
        dataset = OutcomeDataset.from_list(high + low)

        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(volatility_state=VolatilityState.HIGH))
        assert result.probability.counts.total_observations == 7


class TestStructureFiltering:
    def test_filters_by_structure_state(self) -> None:
        trending = [make_outcome(ts(i), regime_context=make_context(struct=StructureState.TRENDING)) for i in range(4)]
        ranging = [make_outcome(ts(20 + i), regime_context=make_context(struct=StructureState.RANGING)) for i in range(6)]
        dataset = OutcomeDataset.from_list(trending + ranging)

        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(structure_state=StructureState.RANGING))
        assert result.probability.counts.total_observations == 6


class TestSessionFiltering:
    def test_filters_by_session_membership(self) -> None:
        london = [make_outcome(ts(i), regime_context=make_context(sessions=("london",))) for i in range(5)]
        tokyo = [make_outcome(ts(20 + i), regime_context=make_context(sessions=("tokyo",))) for i in range(5)]
        dataset = OutcomeDataset.from_list(london + tokyo)

        result = StrategyEvaluator().evaluate(dataset, OutcomeFilter(session="london"))
        assert result.probability.counts.total_observations == 5


class TestCombinedFilters:
    def test_combines_strategy_instrument_and_regime_dimensions(self) -> None:
        matching = [
            make_outcome(
                ts(i), strategy="orb", instrument="EURUSD",
                regime_context=make_context(trend=TrendState.UPTREND, vol=VolatilityState.HIGH, sessions=("london",)),
            )
            for i in range(8)
        ]
        non_matching = [
            make_outcome(
                ts(20 + i), strategy="orb", instrument="EURUSD",
                regime_context=make_context(trend=TrendState.DOWNTREND, vol=VolatilityState.HIGH, sessions=("london",)),
            )
            for i in range(4)
        ]
        dataset = OutcomeDataset.from_list(matching + non_matching)

        f = OutcomeFilter(
            strategy_name="orb", instrument_symbol="EURUSD",
            trend_state=TrendState.UPTREND, volatility_state=VolatilityState.HIGH, session="london",
        )
        result = StrategyEvaluator().evaluate(dataset, f)
        assert result.probability.counts.total_observations == 8
