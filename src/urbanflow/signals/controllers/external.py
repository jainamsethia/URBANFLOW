"""Externally commanded control: the API, manual holds and RL (plan H.4).

``request_phase(p)`` queues ``p``; ``decide`` returns the queued request (the runtime
enforces min-green, yellow and all-red) and forgets it once the switch starts: at the
first decision where the runtime will switch (``green_elapsed >= min_green - 1e-9``) or
where ``p`` is already green. The last request wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from pydantic import JsonValue

from urbanflow.core.constants import TIME_EPS
from urbanflow.signals.controllers.base import (
    ControllerBase,
    ControllerContext,
    ControllerSetup,
    register_controller,
)

__all__ = ["External"]


@register_controller("external")
class External(ControllerBase):
    """Phases on request (``accepts_requests = True``)."""

    accepts_requests: ClassVar[bool] = True

    def __init__(self) -> None:
        self._pending: int | None = None

    def reset(self, setup: ControllerSetup) -> None:  # noqa: ARG002
        """Forget any queued request."""
        self._pending = None

    def request_phase(self, phase: int) -> None:
        """Queue ``phase`` (replacing an earlier request)."""
        self._pending = int(phase)

    @property
    def pending(self) -> int | None:
        """The queued request, if any."""
        return self._pending

    def decide(self, ctx: ControllerContext) -> int | None:
        """The queued request; forgotten once the runtime switches to it."""
        q = self._pending
        if q is not None and (q == ctx.phase or ctx.green_elapsed >= ctx.min_green - TIME_EPS):
            self._pending = None
        return q

    def state_dict(self) -> dict[str, JsonValue]:
        """``{"pending": phase or None}``."""
        return {"pending": self._pending}

    def load_state_dict(self, d: Mapping[str, JsonValue]) -> None:
        """Restore :meth:`state_dict` output."""
        pending = d.get("pending")
        self._pending = pending if isinstance(pending, int) else None
