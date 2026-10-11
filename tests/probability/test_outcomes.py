from __future__ import annotations

from datetime import datetime, timezone

from app.probability.models import ExitReason, OutcomeLabel, TradeOutcome
from app.probability.outcomes import OutcomeDataset


def _outcome(exit_ts: datetime, label: OutcomeLabel = OutcomeLabel.WIN, strategy: str = "orb") -> TradeOutcome:
    return TradeOutcome(
        strategy_name=strategy,
        instrument_symbol="EURUSD",
        timeframe="5M",
        direction="long",
        signal_timestamp=exit_ts,
        entry_timestamp=exit_ts,
        exit_timestamp=exit_ts,
        entry_price=100.0,
        exit_price=104.0,
        stop_loss=98.0,
        take_profit=104.0,
        r_multiple=2.0,
        outcome_label=label,
        exit_reason=ExitReason.TAKE_PROFIT,
        evaluation_horizon_bars=20,
        bars_to_exit=5,
    )


def _ts(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


class TestBasics:
    def test_len_and_iter(self) -> None:
        outcomes = [_outcome(_ts(1)), _outcome(_ts(2))]
        dataset = OutcomeDataset.from_list(outcomes)
        assert len(dataset) == 2
        assert list(dataset) == outcomes

    def test_getitem(self) -> None:
        outcomes = [_outcome(_ts(1)), _outcome(_ts(2))]
        dataset = OutcomeDataset.from_list(outcomes)
        assert dataset[0] == outcomes[0]


class TestResolvedBefore:
    def test_excludes_outcomes_resolved_after_cutoff(self) -> None:
        dataset = OutcomeDataset.from_list([_outcome(_ts(1)), _outcome(_ts(5)), _outcome(_ts(10))])
        result = dataset.resolved_before(_ts(5))
        assert len(result) == 2
        assert all(o.exit_timestamp <= _ts(5) for o in result)

    def test_empty_dataset_stays_empty(self) -> None:
        dataset = OutcomeDataset.from_list([])
        assert len(dataset.resolved_before(_ts(1))) == 0


class TestGroupBy:
    def test_groups_by_key(self) -> None:
        dataset = OutcomeDataset.from_list(
            [_outcome(_ts(1), strategy="orb"), _outcome(_ts(2), strategy="orb"), _outcome(_ts(3), strategy="rsi")]
        )
        groups = dataset.group_by(lambda o: o.strategy_name)
        assert set(groups.keys()) == {"orb", "rsi"}
        assert len(groups["orb"]) == 2
        assert len(groups["rsi"]) == 1


class TestSerialization:
    def test_to_list_of_dicts(self) -> None:
        dataset = OutcomeDataset.from_list([_outcome(_ts(1))])
        result = dataset.to_list_of_dicts()
        assert len(result) == 1
        assert result[0]["outcome_label"] == "win"

    def test_to_dataframe(self) -> None:
        dataset = OutcomeDataset.from_list([_outcome(_ts(1)), _outcome(_ts(2))])
        df = dataset.to_dataframe()
        assert len(df) == 2
        assert "outcome_label" in df.columns
