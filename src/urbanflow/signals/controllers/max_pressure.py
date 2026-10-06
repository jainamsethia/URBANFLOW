"""``max_pressure``: Varaiya's max-pressure control (plan H.4).

Every ``decision_interval`` seconds of green (and only after min-green) the phase with the
largest pressure ``P_phi = sum over green connectors c of (x_in(c) - x_out(c))`` is chosen,
``x`` being vehicle counts on the connector's from- and to-lane. Ties keep the current
phase, then the lowest index. At max-green the best *other* phase with waiting vehicles
is taken, so a phase cannot hold the green forever against demand.

Reference: P. Varaiya, "Max pressure control of a network of signalized intersections",
Transportation Research Part C 36 (2013).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from urbanflow.core.constants import MAX_PRESSURE_DECISION_INTERVAL, TIME_EPS
from urbanflow.core.types import BoolArray, IntArray
from urbanflow.signals.controllers.base import (
    ControllerBase,
    ControllerContext,
    ControllerSetup,
    register_controller,
)

__all__ = ["MaxPressure", "MaxPressureParams", "max_pressure_choice"]


class MaxPressureParams(BaseModel):
    """``max_pressure`` parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_interval: float = Field(default=MAX_PRESSURE_DECISION_INTERVAL, gt=0)
    """Seconds between decisions."""


def max_pressure_choice(
    x_in: IntArray, x_out: IntArray, phase_conn: BoolArray, current: int, *, exclude: int = -1
) -> int:
    """Phase with the largest pressure; ties go to ``current``, then the lowest index.

    ``phase_conn[p, c]`` marks the connectors green in phase ``p``; ``exclude`` removes one
    phase from the choice (used at max-green).
    """
    pressure = phase_conn.astype(np.float64) @ (x_in - x_out).astype(np.float64)
    if 0 <= exclude < pressure.size and pressure.size > 1:
        pressure[exclude] = -np.inf
    best = pressure.max()
    if current != exclude and pressure[current] == best:
        return current
    return int(np.flatnonzero(pressure == best)[0])


@register_controller("max_pressure")
class MaxPressure(ControllerBase):
    """Greedy max-pressure phase selection."""

    Params: ClassVar[type[BaseModel]] = MaxPressureParams

    def __init__(self) -> None:
        self._interval = MAX_PRESSURE_DECISION_INTERVAL
        self._next = 0.0

    def reset(self, setup: ControllerSetup) -> None:
        """Read the decision interval; the first decision comes after min-green."""
        self._interval = cast(MaxPressureParams, setup.params).decision_interval
        self._next = 0.0

    def decide(self, ctx: ControllerContext) -> int | None:
        """Every interval after min-green: the max-pressure phase (None keeps the green)."""
        if ctx.green_elapsed < ctx.min_green - TIME_EPS or ctx.time < self._next - TIME_EPS:
            return None
        self._next = ctx.time + self._interval
        maxed = ctx.green_elapsed >= ctx.max_green - TIME_EPS
        x_in = ctx.conn_in_count
        q = max_pressure_choice(
            x_in, ctx.conn_out_count, ctx.phase_conn, ctx.phase, exclude=ctx.phase if maxed else -1
        )
        if q == ctx.phase or (maxed and not x_in[ctx.phase_conn[q]].any()):
            return None  # keep the green (at max-green only for a phase with demand)
        return q

    def state_dict(self) -> dict[str, JsonValue]:
        """``{"next": time of the next decision}``."""
        return {"next": self._next}

    def load_state_dict(self, d: Mapping[str, JsonValue]) -> None:
        """Restore :meth:`state_dict`."""
        self._next = float(cast(float, d["next"]))
