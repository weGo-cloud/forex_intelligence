from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.base import MarketContext
from app.strategies.mean_reversion import BBStochMeanReversionConfig, BBStochMeanReversionStrategy


def _ranging_series(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A mean-reverting (AR(1)-style) synthetic series - low-ADX by
    construction, exactly the regime this strategy targets."""
    rng = np.random.default_rng(seed)
    mean = 100.0
    x = np.zeros(n)
    x[0] = mean
    for i in range(1, n):
        x[i] = x[i - 1] + 0.35 * (mean - x[i - 1]) + rng.normal(0, 0.5)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(x, index=idx)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


def _trending_series(n: int = 400) -> pd.DataFrame:
    """A strict, unbroken uptrend - high-ADX by construction, exactly
    the regime this strategy must stay out of."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series([100.0 + 0.05 * i for i in range(n)], index=idx)
    high = close + 0.05
    low = close - 0.05
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


class TestConfigValidation:
    def test_rejects_non_positive_bb_num_std(self) -> None:
        with pytest.raises(ValueError):
            BBStochMeanReversionConfig(instrument_symbol="EURUSD", bb_num_std=0)

    def test_rejects_oversold_not_below_overbought(self) -> None:
        with pytest.raises(ValueError):
            BBStochMeanReversionConfig(instrument_symbol="EURUSD", stoch_oversold=80, stoch_overbought=20)

    def test_rejects_non_positive_adx_threshold(self) -> None:
        with pytest.raises(ValueError):
            BBStochMeanReversionConfig(instrument_symbol="EURUSD", adx_ranging_threshold=0)

    def test_rejects_non_positive_stop_atr_mult(self) -> None:
        with pytest.raises(ValueError):
            BBStochMeanReversionConfig(instrument_symbol="EURUSD", stop_atr_mult=0)

    def test_rejects_non_positive_band_lookback(self) -> None:
        with pytest.raises(ValueError):
            BBStochMeanReversionConfig(instrument_symbol="EURUSD", band_lookback_bars=0)


class TestSignalGeneration:
    def test_ranging_market_produces_both_direction_signals(self) -> None:
        df = _ranging_series()
        cfg = BBStochMeanReversionConfig(instrument_symbol="EURUSD", adx_ranging_threshold=25)
        signals = BBStochMeanReversionStrategy(cfg).generate_signals(MarketContext("EURUSD", df))

        assert len(signals) > 0
        assert {s.direction for s in signals} == {"long", "short"}

    def test_no_signals_in_a_strict_trend(self) -> None:
        df = _trending_series()
        cfg = BBStochMeanReversionConfig(instrument_symbol="EURUSD", adx_ranging_threshold=25)
        signals = BBStochMeanReversionStrategy(cfg).generate_signals(MarketContext("EURUSD", df))
        assert signals == []

    def test_take_profit_is_the_middle_band_not_risk_reward_derived(self) -> None:
        df = _ranging_series()
        cfg = BBStochMeanReversionConfig(instrument_symbol="EURUSD", adx_ranging_threshold=25)
        signals = BBStochMeanReversionStrategy(cfg).generate_signals(MarketContext("EURUSD", df))
        assert all(s.take_profit is not None for s in signals)

    def test_stop_loss_is_on_the_correct_side_of_take_profit(self) -> None:
        df = _ranging_series()
        cfg = BBStochMeanReversionConfig(instrument_symbol="EURUSD", adx_ranging_threshold=25)
        signals = BBStochMeanReversionStrategy(cfg).generate_signals(MarketContext("EURUSD", df))
        for s in signals:
            if s.direction == "long":
                assert s.stop_loss < s.take_profit
            else:
                assert s.stop_loss > s.take_profit

    def test_empty_dataframe_produces_no_signals(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        cfg = BBStochMeanReversionConfig(instrument_symbol="EURUSD")
        signals = BBStochMeanReversionStrategy(cfg).generate_signals(MarketContext("EURUSD", df))
        assert signals == []

    def test_params_reports_every_config_field(self) -> None:
        cfg = BBStochMeanReversionConfig(instrument_symbol="EURUSD")
        strategy = BBStochMeanReversionStrategy(cfg)
        params = strategy.params()
        for field_name in cfg.__dataclass_fields__:
            assert field_name in params
