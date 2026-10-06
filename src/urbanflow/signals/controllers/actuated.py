"""``actuated``: gap-out / max-out actuated control (plan H.4).

In green phase ``p`` after min-green the controller ends the green when every detector on
the lanes served by ``p`` has been empty for longer than ``gap`` seconds (gap-out) or when
``green_elapsed >= max_green`` (max-out). The next phase is the first one after ``p`` in
cycle order with demand (a served detector occupied or a queue); with no demand anywhere
the green rests in ``p``.

ponytail: detectors are the network's fixed ``DETECTOR_LENGTH`` zones at the stop line;
a per-controller ``detector_length`` needs per-intersection detector geometry.
"""

from __future__ import annotations

from typing import ClassVar, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from urbanflow.core.constants import ACTUATED_GAP, TIME_EPS
from urbanflow.core.types import BoolArray
from urbanflow.signals.controllers.base import (
    ControllerBase,
    ControllerContext,
    ControllerSetup,
    register_controller,
)

__all__ = ["Actuated", "ActuatedParams", "served_lanes"]


class ActuatedParams(BaseModel):
    """``actuated`` parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gap: float = Field(default=ACTUATED_GAP, gt=0)
    """Gap-out threshold: seconds without a detection on every served lane."""


def served_lanes(ctx: ControllerContext) -> BoolArray:
    """``[P, n_in]``: incoming lane ``i`` has a connector that is green in phase ``p``."""
    lane_pos = {lane: i for i, lane in enumerate(ctx.in_lanes.tolist())}
    cols = np.array([lane_pos[x] for x in ctx.conn_from_lane.tolist()], dtype=np.intp)
    out = np.zeros((ctx.n_phases, ctx.in_lanes.size), dtype=bool)
    for p in range(ctx.n_phases):
        out[p, cols[ctx.phase_conn[p]]] = True
    return out


@register_controller("actuated")
class Actuated(ControllerBase):
    """Gap-out / max-out with skipping of phases without demand."""

    Params: ClassVar[type[BaseModel]] = ActuatedParams

    def __init__(self) -> None:
        self._gap = ACTUATED_GAP

    def reset(self, setup: ControllerSetup) -> None:
        """Read the gap threshold."""
        self._gap = cast(ActuatedParams, setup.params).gap

    def decide(self, ctx: ControllerContext) -> int | None:
        """Next phase with demand on gap-out / max-out, else None."""
        if ctx.green_elapsed < ctx.min_green - TIME_EPS:
            return None
        served = served_lanes(ctx)
        p = ctx.phase
        gapped = bool(np.all(ctx.detector_last_seen[served[p]] > self._gap))
        maxed = ctx.green_elapsed >= ctx.max_green - TIME_EPS
        if not (gapped or maxed):
            return None
        demand = (ctx.detector_occupied | (ctx.lane_queue > 0))[None, :] & served
        for k in range(1, ctx.n_phases):
            q = (p + k) % ctx.n_phases
            if demand[q].any():
                return q
        return None  # no demand elsewhere: rest in green
