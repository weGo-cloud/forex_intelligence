"""The deterministic setup state machine (mission section 8) - the ONE
place a `Setup.status` change is decided. No LLM, detector, or API
handler sets `status` directly; every status change goes through
`transition()`, which either returns a new `Setup` (frozen dataclasses
are never mutated in place) or raises `InvalidTransitionError`.

Ordering, forward-only, no skipping:

    DETECTED -> WATCH -> CONFIRMING -> VALID -> ENTRY_CONDITION_MET

From any NON-terminal state, moving to `INVALIDATED` or `EXPIRED` is
always allowed (an invalidation or expiry can happen at any stage).
`INVALIDATED` and `EXPIRED` are terminal - nothing transitions out of
them (mission's own examples: `INVALIDATED -> VALID` and
`EXPIRED -> ENTRY_MET` are both invalid).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Optional

from app.setups.models import Setup, SetupStatus

_ORDER: list[SetupStatus] = [
    SetupStatus.DETECTED,
    SetupStatus.WATCH,
    SetupStatus.CONFIRMING,
    SetupStatus.VALID,
    SetupStatus.ENTRY_CONDITION_MET,
]
_TERMINAL: frozenset[SetupStatus] = frozenset({SetupStatus.INVALIDATED, SetupStatus.EXPIRED})


class InvalidTransitionError(ValueError):
    """Raised by `transition()` for any state change not permitted by
    the table above - including transitioning out of a terminal state,
    skipping a step in the forward order, or moving backward."""


def is_valid_transition(current: SetupStatus, target: SetupStatus) -> bool:
    if current == target:
        return False  # a "transition" to the same state is not a transition
    if current in _TERMINAL:
        return False
    if target in _TERMINAL:
        return True
    if current not in _ORDER or target not in _ORDER:
        return False
    return _ORDER.index(target) == _ORDER.index(current) + 1


def transition(setup: Setup, target: SetupStatus, at: datetime, reason: Optional[str] = None) -> Setup:
    """Returns a NEW `Setup` with `status=target` and `updated_at=at` -
    `setup` itself is never mutated (it's frozen). `at` is an explicit
    caller-supplied timestamp (never `datetime.now()`) so every
    transition stays deterministic and testable, matching this
    project's established causality discipline throughout M1-M6."""
    if not is_valid_transition(setup.status, target):
        raise InvalidTransitionError(f"Cannot transition setup {setup.id!r} from {setup.status} to {target}")

    metadata = dict(setup.metadata)
    if reason is not None:
        metadata["last_transition_reason"] = reason

    return dataclasses.replace(setup, status=target, updated_at=at, metadata=metadata)
