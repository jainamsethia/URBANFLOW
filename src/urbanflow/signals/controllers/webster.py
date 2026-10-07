"""Webster's optimal fixed-time plan from the scenario demand (plan H.4).

Each movement's design flow ``q_m`` (veh/h) is split evenly over its source lanes, so a
lane shared by several movements carries the sum of their shares, ``q_l``. With saturation
flow ``s`` per lane the critical flow ratio of phase ``p`` is ``y_p = max q_l / s`` over the
lanes of the movements green in ``p``, and ``Y = sum_p y_p``. The lost time ``L`` is
``lost_time_per_phase * P`` or, by default, ``yellow + all_red`` of every transition
``p -> p+1`` with a losing movement.
Webster (1958):

    C_0 = (1.5 L + 5) / (1 - Y),  clipped to cycle_bounds;  C = C_max when Y >= 0.95
    g_p = max(min_green_p, (C - L) y_p / Y)

and with no demand (``Y = 0``) ``C = C_min`` split equally. The controller then runs as
``fixed_time`` (offset included) with the greens ``g_p``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import ClassVar, Self, cast

import numpy as np
from pydantic import BaseModel, Field, model_validator

from urbanflow.core import constants as C
from urbanflow.core.types import FloatArray
from urbanflow.signals.controllers.base import ControllerSetup, register_controller
from urbanflow.signals.controllers.fixed_time import FixedTime, FixedTimeParams
from urbanflow.signals.program import SignalProgram

__all__ = ["Webster", "WebsterParams", "webster_timing"]

log = logging.getLogger(__name__)


class WebsterParams(FixedTimeParams):
    """``webster`` parameters (plus ``fixed_time``'s ``offset``)."""

    saturation_flow: float = Field(default=C.WEBSTER_SATURATION_FLOW, gt=0, allow_inf_nan=False)
    """Saturation flow per lane, veh/h."""
    cycle_bounds: tuple[float, float] = (C.WEBSTER_CYCLE_MIN, C.WEBSTER_CYCLE_MAX)
    """Minimum and maximum cycle, s."""
    lost_time_per_phase: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    """Lost time per phase, s; null = yellow + all-red of each intergreen."""

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        lo, hi = self.cycle_bounds
        if not 0 < lo <= hi:
            raise ValueError("cycle_bounds must satisfy 0 < min <= max")
        return self


def webster_timing(
    ratios: FloatArray, lost: float, min_green: FloatArray, bounds: tuple[float, float]
) -> tuple[float, FloatArray]:
    """``(C, g)``: Webster's cycle and greens for critical flow ratios ``y_p``."""
    lo, hi = bounds
    total = float(ratios.sum())
    if total <= 0:
        cycle = lo
        green = np.full(len(ratios), (cycle - lost) / len(ratios))
    else:
        if total >= C.WEBSTER_Y_MAX:
            cycle = hi
        else:
            c0 = (C.WEBSTER_LOST_TIME_FACTOR * lost + C.WEBSTER_CYCLE_CONSTANT) / (1.0 - total)
            cycle = min(max(c0, lo), hi)
        green = (cycle - lost) * ratios / total
    return cycle, np.maximum(green, min_green)


@register_controller("webster")
class Webster(FixedTime):
    """Fixed-time control with Webster's cycle and green splits for the design demand.

    ponytail: timing from the scenario's flows only; no ``update_interval`` re-timing from
    measured arrivals. Add it when demand drifts within a run matters.
    """

    Params: ClassVar[type[BaseModel]] = WebsterParams

    cycle: float = 0.0
    """Webster's design cycle ``C``, s (the realised cycle adds dt quantisation and
    min-green floors)."""

    def timed_program(self, setup: ControllerSetup) -> SignalProgram:
        """The program with Webster's greens as durations."""
        prog, p = setup.program, cast(WebsterParams, setup.params)
        n = prog.n_phases
        flows, lanes = setup.movement_flows, setup.movement_lanes
        if len(flows) != len(prog.movements) or len(lanes) != len(prog.movements):
            flows, lanes = np.zeros(len(prog.movements)), ((),) * len(prog.movements)
        lane_flow: dict[int, float] = {}
        for q, src in zip(flows.tolist(), lanes, strict=True):
            for lane in src:
                lane_flow[lane] = lane_flow.get(lane, 0.0) + q / len(src)
        y = (
            np.array(
                [
                    max(
                        (lane_flow[lane] for m in np.flatnonzero(row >= 2) for lane in lanes[m]),
                        default=0.0,
                    )
                    for row in prog.phase_state  # G or g
                ]
            )
            / p.saturation_flow
        )
        if p.lost_time_per_phase is not None:
            lost = p.lost_time_per_phase * n
        else:
            changes = sum(not prog.transition(q, (q + 1) % n).immediate for q in range(n))
            lost = changes * (prog.yellow + prog.all_red)
        self.cycle, greens = webster_timing(y, lost, prog.min_green, p.cycle_bounds)
        log.debug(
            "webster intersection %d: Y=%.3f C=%.1f s greens=%s",
            prog.intersection,
            y.sum(),
            self.cycle,
            np.round(greens, 1).tolist(),
        )
        return replace(prog, duration=greens)
