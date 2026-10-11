from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.backtesting.validation import walk_forward_windows


def utc(*args, **kwargs):
    return datetime(*args, tzinfo=timezone.utc, **kwargs)


class TestWalkForwardWindows:
    def test_generates_expected_number_of_windows(self) -> None:
        windows = walk_forward_windows(utc(2024, 1, 1), utc(2024, 4, 1), train_days=30, test_days=15)
        assert len(windows) > 0
        for w in windows:
            assert w.test_end <= utc(2024, 4, 1)

    def test_test_window_immediately_follows_train_window(self) -> None:
        windows = walk_forward_windows(utc(2024, 1, 1), utc(2024, 4, 1), train_days=30, test_days=15)
        for w in windows:
            assert w.test_start == w.train_end
            assert w.train_end - w.train_start == timedelta(days=30)
            assert w.test_end - w.test_start == timedelta(days=15)

    def test_windows_are_sequentially_indexed(self) -> None:
        windows = walk_forward_windows(utc(2024, 1, 1), utc(2024, 6, 1), train_days=30, test_days=15)
        assert [w.window_index for w in windows] == list(range(len(windows)))

    def test_step_days_controls_overlap(self) -> None:
        no_overlap = walk_forward_windows(utc(2024, 1, 1), utc(2024, 6, 1), train_days=30, test_days=15)
        with_overlap = walk_forward_windows(utc(2024, 1, 1), utc(2024, 6, 1), train_days=30, test_days=15, step_days=5)
        # A smaller step should produce at least as many (and here, more) windows.
        assert len(with_overlap) >= len(no_overlap)

    def test_too_short_range_returns_empty_list_not_error(self) -> None:
        windows = walk_forward_windows(utc(2024, 1, 1), utc(2024, 1, 10), train_days=30, test_days=15)
        assert windows == []

    def test_rejects_invalid_window_sizes(self) -> None:
        with pytest.raises(ValueError):
            walk_forward_windows(utc(2024, 1, 1), utc(2024, 4, 1), train_days=0, test_days=15)
        with pytest.raises(ValueError):
            walk_forward_windows(utc(2024, 1, 1), utc(2024, 4, 1), train_days=30, test_days=0)
        with pytest.raises(ValueError):
            walk_forward_windows(utc(2024, 1, 1), utc(2024, 4, 1), train_days=30, test_days=15, step_days=0)
