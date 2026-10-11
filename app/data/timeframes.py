"""Supported candle timeframes.

Timeframes are represented as a string enum so they serialize cleanly
(API responses, JSON provenance records, DB columns) while still giving
application code a single typed place to reason about "how many minutes
is this timeframe".

Adding a new timeframe is a two-line change: add the enum member and its
entry in `_MINUTES_BY_TIMEFRAME`. No other module hard-codes the list of
supported timeframes.
"""

from __future__ import annotations

import enum


class Timeframe(str, enum.Enum):
    """A supported candle timeframe.

    Values match the wire/display format used throughout the system
    (API query params, provenance records, DB storage).
    """

    M1 = "1M"
    M5 = "5M"
    M15 = "15M"
    M30 = "30M"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"

    @property
    def minutes(self) -> int:
        """Nominal duration of one candle of this timeframe, in minutes.

        This is a heuristic used for gap-detection thresholds and
        aggregation bucket sizing, NOT a claim about session boundaries.
        In particular, a "1D" candle does not necessarily span exactly
        1440 minutes of active trading (weekends/holidays exist) — see
        app/data/validation.py for how gap detection treats this as a
        reporting signal rather than a rejection rule.
        """
        return _MINUTES_BY_TIMEFRAME[self]


_MINUTES_BY_TIMEFRAME: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}
