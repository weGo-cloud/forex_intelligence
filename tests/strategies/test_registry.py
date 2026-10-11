from __future__ import annotations

import pytest

from app.strategies.registry import (
    _reset_registry_for_tests,
    get_strategy,
    list_strategies,
    register_strategy,
)


class DummyConfig:
    def __init__(self, instrument_symbol: str, multiplier: int = 1) -> None:
        self.instrument_symbol = instrument_symbol
        self.multiplier = multiplier


class DummyStrategy:
    def __init__(self, config: DummyConfig) -> None:
        self.config = config
        self.name = "dummy"

    def params(self):
        return {"multiplier": self.config.multiplier}

    def generate_signals(self, market):
        return []


@pytest.fixture(autouse=True)
def clean_registry():
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


class TestRegistry:
    def test_register_and_get(self) -> None:
        @register_strategy("dummy", DummyConfig, "A test strategy")
        def build(**kwargs):
            return DummyStrategy(DummyConfig(**kwargs))

        s = get_strategy("dummy", instrument_symbol="US30", multiplier=3)
        assert s.name == "dummy"
        assert s.params()["multiplier"] == 3

    def test_unknown_name_raises_with_available_list(self) -> None:
        @register_strategy("dummy", DummyConfig)
        def build(**kwargs):
            return DummyStrategy(DummyConfig(**kwargs))

        with pytest.raises(KeyError, match="dummy"):
            get_strategy("nope")

    def test_duplicate_registration_rejected(self) -> None:
        @register_strategy("dummy", DummyConfig)
        def build_a(**kwargs):
            return DummyStrategy(DummyConfig(**kwargs))

        with pytest.raises(ValueError):

            @register_strategy("dummy", DummyConfig)
            def build_b(**kwargs):
                return DummyStrategy(DummyConfig(**kwargs))

    def test_list_strategies_is_sorted(self) -> None:
        @register_strategy("zzz", DummyConfig)
        def build_z(**kwargs):
            return DummyStrategy(DummyConfig(**kwargs))

        @register_strategy("aaa", DummyConfig)
        def build_a(**kwargs):
            return DummyStrategy(DummyConfig(**kwargs))

        names = [e.name for e in list_strategies()]
        assert names == ["aaa", "zzz"]
