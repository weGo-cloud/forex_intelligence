"""Tests for app.data.timeframes."""

from app.data.timeframes import Timeframe


def test_all_required_timeframes_are_supported() -> None:
    required_values = {"1M", "5M", "15M", "30M", "1H", "4H", "1D"}
    actual_values = {tf.value for tf in Timeframe}
    assert actual_values == required_values


def test_minutes_are_correct_for_each_timeframe() -> None:
    assert Timeframe.M1.minutes == 1
    assert Timeframe.M5.minutes == 5
    assert Timeframe.M15.minutes == 15
    assert Timeframe.M30.minutes == 30
    assert Timeframe.H1.minutes == 60
    assert Timeframe.H4.minutes == 240
    assert Timeframe.D1.minutes == 1440


def test_timeframe_is_constructible_from_its_string_value() -> None:
    assert Timeframe("1H") is Timeframe.H1
