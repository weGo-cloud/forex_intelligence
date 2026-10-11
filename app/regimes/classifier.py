"""`RegimeClassifier`: the M4 engine's public entry point.

```
features (app.regimes.features)
    -> RegimeClassifier
    -> RegimeContext (app.regimes.context)
```

No hidden global state (config and session definitions are both
constructor arguments, immutable dataclasses/plain dicts); deterministic
(identical `df` + identical `config` always produces identical output -
see `tests/regimes/test_bias.py::test_determinism`); every value read
out of a feature series at bar i, never i+1 or later.

MULTI-TIMEFRAME READINESS (M4 mission section 14): this classifier
takes one OHLC frame and produces one context per bar of THAT frame -
it does not assume "one symbol = one timeframe = one context" at the
API level. A caller wanting `execution timeframe = 15M, context
timeframe = 1H` calls this twice (once per frame) and combines the two
`RegimeContext` sequences however it needs, OR - for a single execution
frame that needs 1H-aligned regime context alongside its own 5M
context - runs the classifier on the 1H frame first and aligns the
resulting per-1H-bar contexts onto the 5M index using
`app.backtesting.data_access.align_higher_timeframe` (M2/M3's existing
causal alignment utility), rather than this module inventing a second
one. Nothing here special-cases "the" execution timeframe.
"""

from __future__ import annotations

from datetime import time

import pandas as pd

from app.features.sessions import SessionWindow
from app.regimes.config import RegimeClassifierConfig
from app.regimes.context import RegimeContext
from app.regimes.detectors import classify_structure, classify_trend, classify_volatility
from app.regimes.features import RegimeFeatureSet, compute_feature_set

# Approximate, commonly-used major-session local trading hours. These
# are a reasonable DEFAULT, not an authoritative source (real exchange/
# liquidity-session hours vary by convention and don't have one
# universally agreed definition) - pass a custom `sessions` dict to
# `RegimeClassifier` to override. None of these wrap midnight (kept
# consistent with `SessionWindow`'s own same-day start/end model).
DEFAULT_SESSIONS: dict[str, SessionWindow] = {
    "sydney": SessionWindow("Australia/Sydney", time(7, 0), time(16, 0)),
    "tokyo": SessionWindow("Asia/Tokyo", time(9, 0), time(18, 0)),
    "london": SessionWindow("Europe/London", time(8, 0), time(17, 0)),
    "new_york": SessionWindow("America/New_York", time(8, 0), time(17, 0)),
}


def session_labels_at(ts: pd.Timestamp, sessions: dict[str, SessionWindow]) -> tuple[str, ...]:
    """Names of every session in `sessions` whose local window covers
    `ts` - zero, one, or more than one (sessions legitimately overlap,
    e.g. London/New York). Reuses `SessionWindow` as-is (no second
    timezone system, per the M4 mission section 13); `sessions={}`
    (or `None` passed through `RegimeClassifier`) makes this always
    return `()`, the explicit "session unavailable" representation.
    """
    active = []
    for name, window in sessions.items():
        local_time = ts.tz_convert(window.tz_name).time()
        if window.start <= local_time < window.end:
            active.append(name)
    return tuple(active)


class RegimeClassifier:
    def __init__(
        self,
        config: RegimeClassifierConfig | None = None,
        sessions: dict[str, SessionWindow] | None = DEFAULT_SESSIONS,
    ) -> None:
        self.config = config or RegimeClassifierConfig()
        self.sessions = sessions or {}

    def compute_feature_set(self, df: pd.DataFrame) -> RegimeFeatureSet:
        return compute_feature_set(df, self.config)

    def classify_at(
        self,
        feature_set: RegimeFeatureSet,
        position: int,
        instrument_symbol: str,
        timeframe: str,
    ) -> RegimeContext:
        """Classify a single already-computed bar position. Splitting
        this from `classify_series` lets a caller (e.g. a live/streaming
        consumer, or M3 research code that already has a feature set for
        other reasons) classify one bar at a time without recomputing
        the whole feature set."""
        ts = feature_set.close.index[position]

        trend = classify_trend(
            close=feature_set.close.iloc[position],
            slow_ma=feature_set.slow_ma.iloc[position],
            slow_ma_slope=feature_set.slow_ma_slope.iloc[position],
            plus_di=feature_set.plus_di.iloc[position],
            minus_di=feature_set.minus_di.iloc[position],
            adx=feature_set.trend_adx.iloc[position],
            config=self.config,
        )
        volatility = classify_volatility(feature_set.volatility_percentile.iloc[position], self.config)
        structure = classify_structure(
            feature_set.structural_trend_strength.iloc[position],
            feature_set.range_compression_ratio.iloc[position],
            self.config,
        )

        return RegimeContext(
            timestamp=ts,
            instrument_symbol=instrument_symbol,
            timeframe=timeframe,
            trend_state=trend,
            volatility_state=volatility,
            structure_state=structure,
            sessions=session_labels_at(ts, self.sessions),
            feature_values=feature_set.feature_values_at(position),
        )

    def classify_series(self, df: pd.DataFrame, instrument_symbol: str, timeframe: str) -> list[RegimeContext]:
        """Compute the feature set once, then classify every bar in
        `df`. Convenience wrapper over `compute_feature_set` +
        `classify_at`, not a separate code path (so there is exactly
        one place any of this logic lives)."""
        feature_set = self.compute_feature_set(df)
        return [
            self.classify_at(feature_set, position, instrument_symbol, timeframe)
            for position in range(len(df))
        ]
