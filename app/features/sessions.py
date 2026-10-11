"""Session-window helpers used by strategies that need "the New York
session" or similar concepts.

Per the roadmap's timezone rule (and M1's own stance in
docs/M1_DATA_FOUNDATION.md: "No session/exchange timezone offsets are
hard-coded anywhere"), session boundaries are computed from an IANA zone
name (e.g. "America/New_York") converted to UTC per calendar day, not
from a fixed UTC-offset constant. This makes DST transitions correct
automatically instead of silently wrong twice a year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class SessionWindow:
    """A daily session window, e.g. 09:30-16:00 America/New_York."""

    tz_name: str
    start: time
    end: time

    def session_bounds_utc(self, for_dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Return (start_utc, end_utc) for each distinct local calendar
        date present in `for_dates` (a UTC-aware DatetimeIndex),
        converted through the session's IANA zone."""
        tz = ZoneInfo(self.tz_name)
        local_dates = sorted({ts.tz_convert(tz).date() for ts in for_dates})
        bounds = []
        for d in local_dates:
            start_local = pd.Timestamp.combine(d, self.start).tz_localize(tz)
            end_local = pd.Timestamp.combine(d, self.end).tz_localize(tz)
            bounds.append((start_local.tz_convert("UTC"), end_local.tz_convert("UTC")))
        return bounds

    def session_date(self, ts: pd.Timestamp) -> object:
        """The local calendar date (in this session's zone) that `ts`
        falls on — used to bucket signals per trading day for
        max-trades-per-session limits."""
        return ts.tz_convert(ZoneInfo(self.tz_name)).date()
