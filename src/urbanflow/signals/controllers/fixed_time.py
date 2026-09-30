"""Fixed-time control with a coordination offset (plan H.4).

``decide`` requests ``(p + 1) mod P`` once ``green_elapsed >= max(duration[p],
min_green[p]) - 1e-9``. With ``d_p = max(duration_p, min_green_p)`` and ``n(x) =
ceil(x/dt - 1e-9)`` the realised (quantised) cycle is

    C_q = sum_p [ n(d_p) + [losing_p != {}] (n(yellow) + n(all_red)) ] dt

where ``losing_p`` is the losing set of ``trans(p -> p+1)``. At reset the cycle position
``u = (-offset) mod C_q`` (the cycle's first green, the program's ``initial_phase``, starts
at ``t = offset``) is mapped to a stage, phase, target and elapsed time, including
positions inside yellow or all-red, so coordinated intersections keep their offsets
exactly. A mid-run install (``set_controller``, the end of a manual hold) continues from
the current phase instead, so offsets may drift after a hold.
"""

from __future__ import annotations

import math
from typing import ClassVar, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from urbanflow.core.constants import TIME_EPS
from urbanflow.core.types import FloatArray, Stage
from urbanflow.signals.controllers.base import (
    ControllerBase,
    ControllerContext,
    ControllerSetup,
    register_controller,
)
from urbanflow.signals.program import SignalProgram

__all__ = [
    "CyclePosition",
    "FixedTime",
    "FixedTimeParams",
    "cycle_position",
    "realised_cycle",
    "stage_steps",
]

CyclePosition = tuple[Stage, int, int, float, float]
"""``(stage, phase, target, stage_elapsed, green_elapsed)``; target -1 in green."""


class FixedTimeParams(BaseModel):
    """``fixed_time`` parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    offset: float = Field(default=0.0, allow_inf_nan=False)
    """Time at which the cycle's first green (``initial_phase``) starts, s (mod the cycle)."""


def stage_steps(seconds: float, dt: float) -> int:
    """Steps a stage of ``seconds`` is applied: ``ceil(seconds/dt - 1e-9)`` (H.2)."""
    return max(0, math.ceil(seconds / dt - TIME_EPS))


def _blocks(program: SignalProgram, dt: float) -> list[tuple[int, int, int, int, int]]:
    """Per phase in cycle order from ``initial_phase``: ``(p, next, green, yellow, all_red)``
    in steps (no intergreen when nothing loses green)."""
    green = np.maximum(program.duration, program.min_green)
    n = program.n_phases
    out = []
    for k in range(n):
        p = (program.initial_phase + k) % n
        q = (p + 1) % n
        inter = not program.transition(p, q).immediate
        y = stage_steps(program.yellow, dt) if inter else 0
        ar = stage_steps(program.all_red, dt) if inter else 0
        out.append((p, q, stage_steps(float(green[p]), dt), y, ar))
    return out


def realised_cycle(program: SignalProgram, dt: float) -> float:
    """The realised fixed-time cycle ``C_q``, s."""
    return sum(g + y + ar for _, _, g, y, ar in _blocks(program, dt)) * dt


def cycle_position(program: SignalProgram, dt: float, offset: float) -> CyclePosition:
    """The runtime state at ``t = 0`` of a fixed-time program with ``offset``.

    The position ``u = (-offset) mod C_q`` is taken at the step it falls in
    (``floor(u/dt + 1e-9)``), so the first green starts at the first step with
    ``t >= offset`` (mod ``C_q``).
    """
    blocks = _blocks(program, dt)
    total = sum(g + y + ar for _, _, g, y, ar in blocks)
    u = (-offset) % (total * dt)
    s = math.floor(u / dt + TIME_EPS) % total
    for p, q, g, y, ar in blocks:
        if s < g:
            return Stage.green, p, -1, s * dt, s * dt
        if s < g + y:
            return Stage.yellow, p, q, (s - g) * dt, s * dt
        if s < g + y + ar:
            return Stage.all_red, p, q, (s - g - y) * dt, s * dt
        s -= g + y + ar
    raise AssertionError("unreachable: the position is inside the cycle")  # pragma: no cover


@register_controller("fixed_time")
class FixedTime(ControllerBase):
    """Cycle through the phases with their programmed durations (H.4)."""

    Params: ClassVar[type[BaseModel]] = FixedTimeParams

    def __init__(self) -> None:
        self._green: FloatArray = np.zeros(0)

    def reset(self, setup: ControllerSetup) -> None:
        """At a simulation reset, place the runtime at the offset's cycle position."""
        prog = setup.program
        self._green = np.maximum(prog.duration, prog.min_green)
        if setup.initial:
            offset = cast(FixedTimeParams, setup.params).offset
            stage, phase, target, elapsed, green = cycle_position(prog, setup.dt, offset)
            setup.set_initial(
                phase, stage=stage, target=target, stage_elapsed=elapsed, green_elapsed=green
            )

    def decide(self, ctx: ControllerContext) -> int | None:
        """The next phase once the current one has had ``max(duration, min_green)``."""
        p = ctx.phase
        if ctx.green_elapsed >= self._green[p] - TIME_EPS:
            return (p + 1) % ctx.n_phases
        return None
