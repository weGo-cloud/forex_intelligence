"""Strategy registry: the single place a new strategy becomes
discoverable by name instead of requiring a hand-edited import
somewhere else in the codebase.

This is what turns "register strategy" (step 6 of the research
workflow) into an actual action instead of an implicit side effect of
writing an import statement. A strategy that isn't registered here
can't be run from the CLI or compared against others, on purpose —
registration is the boundary between "code that exists" and "code the
lab can experiment with."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.strategies.base import Strategy


@dataclass(frozen=True)
class StrategyEntry:
    name: str
    factory: Callable[..., Strategy]  # takes the strategy's Config kwargs, returns a Strategy
    config_cls: type
    description: str


_REGISTRY: dict[str, StrategyEntry] = {}


def register_strategy(name: str, config_cls: type, description: str = "") -> Callable[[Callable[..., Strategy]], Callable[..., Strategy]]:
    """Decorator for a strategy's factory function.

    Usage (see app/strategies/orb.py bottom for the real registration):

        @register_strategy("orb", ORBConfig, "Opening Range Breakout")
        def build_orb(**kwargs) -> ORBStrategy:
            return ORBStrategy(ORBConfig(**kwargs))
    """

    def decorator(factory: Callable[..., Strategy]) -> Callable[..., Strategy]:
        if name in _REGISTRY:
            raise ValueError(f"Strategy {name!r} is already registered (by {_REGISTRY[name].factory!r})")
        _REGISTRY[name] = StrategyEntry(name=name, factory=factory, config_cls=config_cls, description=description)
        return factory

    return decorator


def get_strategy(name: str, **config_kwargs) -> Strategy:
    """Build a registered strategy by name. Raises KeyError with the
    list of valid names if `name` isn't registered — deliberately not a
    generic ImportError, since the failure mode here is almost always
    "forgot to register it" or "typo'd the name", not a Python error."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"No strategy registered as {name!r}. Available: {available}")
    return _REGISTRY[name].factory(**config_kwargs)


def list_strategies() -> list[StrategyEntry]:
    return sorted(_REGISTRY.values(), key=lambda e: e.name)


def _reset_registry_for_tests() -> None:
    """Test-only escape hatch - not for production use. Registration is
    deliberately global and permanent at runtime (importing a strategy
    module registers it once); tests that need a clean registry use
    this instead of module-reload gymnastics."""
    _REGISTRY.clear()
