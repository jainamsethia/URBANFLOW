"""Emergency-vehicle signal preemption around any inner controller (plan H.4).

While an emergency vehicle is on one of the intersection's connectors, or within
``detection_distance`` of the stop line on an incoming lane with a planned movement, the
controller requests the phase in which the nearest such vehicle's movement is protected
(G), else permitted (g), and holds it; the current phase wins ties, so a phase that already
serves the vehicle is kept. Without such a vehicle it delegates to ``inner``, which then
continues from the current phase. Min-green, yellow and all-red still apply (the runtime
owns them); emergency vehicles obey the signal like any other vehicle.

Other vehicles move over by lane change (``engine.lane_changes``). ponytail: no min-green
truncation; add a ``truncate_min_green`` parameter when response times matter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from urbanflow.core import constants as C
from urbanflow.core.errors import SimulationError, UrbanFlowError
from urbanflow.core.types import SignalState
from urbanflow.scenario.schema import ControllerSpec
from urbanflow.signals.controllers.base import (
    ControllerBase,
    ControllerContext,
    ControllerSetup,
    SignalController,
    create_controller,
    register_controller,
)

__all__ = ["Preemption", "PreemptionParams"]


class PreemptionParams(BaseModel):
    """``preemption`` parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inner: ControllerSpec = ControllerSpec()
    """The controller in charge without an emergency vehicle (default ``fixed_time``)."""
    detection_distance: float = Field(
        default=C.PREEMPTION_DETECTION_DISTANCE, gt=0, allow_inf_nan=False
    )
    """Distance from the stop line at which an approaching emergency vehicle is served, m."""

    @field_validator("inner", mode="before")
    @classmethod
    def _by_name(cls, value: object) -> object:
        return {"type": value} if isinstance(value, str) else value

    @field_validator("inner")
    @classmethod
    def _inner(cls, spec: ControllerSpec) -> ControllerSpec:
        if spec.type == "preemption":
            raise ValueError("preemption cannot wrap preemption")
        try:
            create_controller(spec)
        except UrbanFlowError as exc:
            raise ValueError(str(exc)) from None
        return spec


@register_controller("preemption")
class Preemption(ControllerBase):
    """Serve approaching emergency vehicles, otherwise run the inner controller."""

    Params: ClassVar[type[BaseModel]] = PreemptionParams

    def __init__(self) -> None:
        self.inner: SignalController | None = None
        self.active = False
        """An emergency vehicle is being served (set by :meth:`decide`)."""
        self._distance = C.PREEMPTION_DETECTION_DISTANCE

    def reset(self, setup: ControllerSetup) -> None:
        """Create and reset the inner controller with the same program and timing."""
        p = cast(PreemptionParams, setup.params)
        self.inner, inner_params = create_controller(p.inner)
        self.inner.reset(replace(setup, params=inner_params))
        self._distance = p.detection_distance
        self.active = False

    def target(self, ctx: ControllerContext) -> int | None:
        """The phase serving the nearest emergency vehicle in range, or None."""
        state = ctx.program.phase_state
        for _lane, movement, distance in ctx.emergency_approach:
            if movement < 0 or distance > self._distance:
                continue
            column = state[:, movement]
            best = int(column.max())
            if best < SignalState.g.code:
                continue  # never green: nothing to preempt for
            return ctx.phase if column[ctx.phase] == best else int(np.argmax(column))
        return None

    def decide(self, ctx: ControllerContext) -> int | None:
        """Hold or request the emergency phase; otherwise ask the inner controller."""
        if self.inner is None:
            raise SimulationError("preemption: decide() before reset()")
        phase = self.target(ctx)
        self.active = phase is not None
        if phase is None:
            return self.inner.decide(ctx)
        return None if phase == ctx.phase else phase

    def state_dict(self) -> dict[str, JsonValue]:
        """The inner controller's state and the preemption flag."""
        inner = self.inner.state_dict() if self.inner is not None else {}
        return {"active": self.active, "inner": inner}

    def load_state_dict(self, d: Mapping[str, JsonValue]) -> None:
        """Restore :meth:`state_dict` output."""
        self.active = bool(d.get("active", False))
        inner = d.get("inner")
        if self.inner is not None and isinstance(inner, Mapping):
            self.inner.load_state_dict(inner)
