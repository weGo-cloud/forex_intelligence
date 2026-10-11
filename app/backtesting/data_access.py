"""The only module that bridges M1's persistence layer into M2's pandas
world. Every other M2 module (strategies, engine, metrics) operates on
plain pandas DataFrames and never imports SQLAlchemy or the repository
classes directly — this keeps the research code testable without a
database and keeps this one seam as the sole place that would need to
change if M1's storage layer changes.

Reuses, unmodified: `Timeframe`, `CandleRepository`, `InstrumentRepository`
(app/data/repository.py, app/data/timeframes.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    # Only needed for type hints. Deferred so that the pure-pandas
    # functions in this module (candles_to_dataframe, align_higher_timeframe)
    # can be imported and unit-tested without SQLAlchemy installed/available -
    # they never touch the database. `load_ohlcv` imports the real
    # repository/session types lazily, at call time, since it's the only
    # function that actually needs them.
    from sqlalchemy.orm import Session

    from app.data.timeframes import Timeframe


class UnknownInstrumentError(Exception):
    """Raised when `load_ohlcv` is asked for a symbol M1 has never seen."""


def candles_to_dataframe(candles) -> pd.DataFrame:
    """Convert a sequence of M1 `Candle` ORM rows (or CanonicalCandle) into
    a pandas DataFrame indexed by UTC timestamp with float OHLCV columns.

    M1 stores prices as `Decimal` deliberately, to keep stored data free
    of float rounding error (see docs/M1_DATA_FOUNDATION.md). This
    function is the single, explicit point where that precision is
    converted to `float` for vectorized research math — acceptable here
    because backtest P&L is an estimate, not a system of record, but the
    conversion boundary is kept in exactly one place rather than
    scattered through strategy code.
    """
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(float)

    rows = {
        "timestamp": [c.timestamp for c in candles],
        "open": [float(c.open) for c in candles],
        "high": [float(c.high) for c in candles],
        "low": [float(c.low) for c in candles],
        "close": [float(c.close) for c in candles],
        "volume": [float(c.volume) if c.volume is not None else float("nan") for c in candles],
    }
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps in candle set; M1's unique constraint should prevent this")
    return df


def load_ohlcv(
    session: "Session",
    symbol: str,
    timeframe: "Timeframe",
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load stored candles for `symbol`/`timeframe` in `[start, end)` as a
    DataFrame. Raises `UnknownInstrumentError` if the symbol was never
    registered in M1 — deliberately not returning an empty frame, since
    "no such instrument" and "instrument exists but has no data in this
    range" are different failure modes a researcher needs to tell apart.
    """
    from app.data.repository import CandleRepository, InstrumentRepository  # deferred - see module docstring

    instrument = InstrumentRepository(session).get_by_symbol(symbol)
    if instrument is None:
        raise UnknownInstrumentError(f"No instrument registered for symbol {symbol!r}")

    candles = CandleRepository(session).query(instrument.id, timeframe, start, end)
    return candles_to_dataframe(candles)


def align_higher_timeframe(base_index: pd.DatetimeIndex, htf_df: pd.DataFrame, htf_minutes: int) -> pd.DataFrame:
    """Reindex a higher-timeframe DataFrame onto `base_index`, exposing
    each HTF bar's data only from the moment that bar has actually
    CLOSED — never from its open timestamp.

    M1 candle timestamps mark bar OPEN time (confirmed in
    app/data/aggregation.py: buckets are keyed by bucket start). A 1H
    candle timestamped 09:00 covers [09:00, 10:00) and is not fully
    known until 10:00. Naively using `reindex(method="ffill")` on the
    raw HTF index would make that candle's high/low/close visible to a
    5-minute bar at 09:05 — three intrabar values that haven't happened
    yet. This function shifts the HTF frame's effective visibility
    timestamp forward by its own bar duration before the as-of join, so
    the 09:00 HTF candle only becomes visible to base bars at or after
    10:00.

    Returns a DataFrame aligned to `base_index`, columns suffixed
    `_htf`, containing NaN wherever no HTF candle has closed yet.
    """
    if htf_df.empty:
        return pd.DataFrame(index=base_index)

    visible_from = htf_df.index + pd.Timedelta(minutes=htf_minutes)
    shifted = htf_df.copy()
    shifted.index = visible_from
    shifted = shifted.add_suffix("_htf")

    base_sorted = pd.DataFrame(index=base_index.sort_values())
    aligned = pd.merge_asof(
        base_sorted.reset_index().rename(columns={"index": "timestamp"}),
        shifted.reset_index().rename(columns={"index": "timestamp"}),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    return aligned.reindex(base_index)
