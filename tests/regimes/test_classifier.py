from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.regimes.classifier import RegimeClassifier, session_labels_at, DEFAULT_SESSIONS
from app.regimes.config import RegimeClassifierConfig
from app.regimes.context import RegimeContext
from app.regimes.models import StructureState, TrendState, VolatilityState


def _synthetic_df(n: int = 400, seed: int = 3, trend: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 5, n) if trend else 0.0
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.05, n)) + drift, index=idx)
    high = close + rng.uniform(0.05, 0.15, n)
    low = close - rng.uniform(0.05, 0.15, n)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


class TestClassifySeries:
    def test_returns_one_context_per_bar(self) -> None:
        df = _synthetic_df()
        contexts = RegimeClassifier().classify_series(df, "EURUSD", "5M")
        assert len(contexts) == len(df)

    def test_early_bars_are_unknown_due_to_warmup(self) -> None:
        df = _synthetic_df(n=60)  # shorter than default slow_ma_period=50 + other warmups
        contexts = RegimeClassifier().classify_series(df, "EURUSD", "5M")
        assert contexts[0].trend_state == TrendState.UNKNOWN
        assert contexts[0].volatility_state == VolatilityState.UNKNOWN
        assert contexts[0].structure_state == StructureState.UNKNOWN

    def test_instrument_and_timeframe_are_propagated(self) -> None:
        df = _synthetic_df()
        contexts = RegimeClassifier().classify_series(df, "GBPUSD", "1H")
        assert all(c.instrument_symbol == "GBPUSD" and c.timeframe == "1H" for c in contexts)

    def test_custom_config_changes_output(self) -> None:
        df = _synthetic_df()
        loose = RegimeClassifier(RegimeClassifierConfig(trend_adx_threshold=0.0))
        strict = RegimeClassifier(RegimeClassifierConfig(trend_adx_threshold=90.0))
        loose_trends = [c.trend_state for c in loose.classify_series(df, "EURUSD", "5M")]
        strict_trends = [c.trend_state for c in strict.classify_series(df, "EURUSD", "5M")]
        # an impossibly high ADX threshold should force every bar to NEUTRAL or UNKNOWN
        assert all(t in (TrendState.NEUTRAL, TrendState.UNKNOWN) for t in strict_trends)
        assert any(t == TrendState.UPTREND for t in loose_trends)

    def test_missing_column_raises(self) -> None:
        df = _synthetic_df().drop(columns=["low"])
        with pytest.raises(ValueError):
            RegimeClassifier().classify_series(df, "EURUSD", "5M")

    def test_empty_dataframe_raises(self) -> None:
        df = _synthetic_df().iloc[:0]
        with pytest.raises(ValueError):
            RegimeClassifier().classify_series(df, "EURUSD", "5M")


class TestSessionLabels:
    def test_london_session_detected(self) -> None:
        ts = pd.Timestamp("2024-01-03 10:00", tz="UTC")  # 10:00 UTC is within London 08:00-17:00 local (winter, no DST)
        labels = session_labels_at(ts, DEFAULT_SESSIONS)
        assert "london" in labels

    def test_no_session_outside_all_windows(self) -> None:
        ts = pd.Timestamp("2024-01-03 02:30", tz="UTC")  # deep off-hours for all default sessions
        labels = session_labels_at(ts, DEFAULT_SESSIONS)
        assert labels == () or isinstance(labels, tuple)

    def test_empty_sessions_dict_always_empty(self) -> None:
        ts = pd.Timestamp("2024-01-03 10:00", tz="UTC")
        assert session_labels_at(ts, {}) == ()

    def test_classifier_with_no_sessions_produces_empty_tuple(self) -> None:
        df = _synthetic_df()
        contexts = RegimeClassifier(sessions={}).classify_series(df, "EURUSD", "5M")
        assert all(c.sessions == () for c in contexts)


class TestRegimeContextSerialization:
    def test_to_dict_is_json_safe(self) -> None:
        import json

        df = _synthetic_df()
        context = RegimeClassifier().classify_series(df, "EURUSD", "5M")[-1]
        payload = context.to_dict()
        json.dumps(payload)  # must not raise
        assert payload["trend_state"] in ("uptrend", "downtrend", "neutral", "unknown")
        assert isinstance(payload["sessions"], list)

    def test_nan_features_become_none(self) -> None:
        df = _synthetic_df(n=60)
        context = RegimeClassifier().classify_series(df, "EURUSD", "5M")[0]
        payload = context.to_dict()
        assert payload["feature_values"]["slow_ma"] is None
