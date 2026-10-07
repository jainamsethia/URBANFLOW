"""``emergency``: a base scenario plus emergency-vehicle trips and signal preemption (E.10).

The base generator's scenario gets trips ``ems.{k}`` of the built-in ``emergency`` type at
``times`` along the longest boundary-to-boundary route. With ``preemption`` every signal
controller ``c`` becomes ``{"type": "preemption", "params": {"inner": c}}``.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from urbanflow.core import constants as C
from urbanflow.scenario.builder import ScenarioBuilder
from urbanflow.scenario.generators import generate as run_generator
from urbanflow.scenario.generators import register_generator
from urbanflow.scenario.generators._common import longest_boundary_path
from urbanflow.scenario.scenario import Scenario

__all__ = ["EmergencyParams", "generate"]


class EmergencyParams(BaseModel):
    """Parameters of :func:`generate`."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    base: str = "corridor"
    """Generator of the underlying scenario."""
    base_params: dict[str, Any] = Field(default_factory=dict)
    """Its parameters (``duration`` defaults to this generator's)."""
    times: tuple[float, ...] = Field(default=C.EMERGENCY_TIMES, min_length=1)
    """Departure times of the emergency vehicles, s."""
    preemption: bool = True
    """Wrap every signal controller in ``preemption``."""
    duration: float = Field(default=C.DURATION, gt=0)
    """Simulated duration, s."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.base == "emergency":
            raise PydanticCustomError("uf_params", "base cannot be emergency", {})
        if any(not 0 <= t < self.duration for t in self.times):
            raise PydanticCustomError("uf_params", "times must lie in [0, duration)", {})
        return self


@register_generator("emergency", params=EmergencyParams)
def generate(p: EmergencyParams) -> Scenario:
    """Emergency vehicles crossing a base scenario, with signal preemption."""
    base = run_generator(p.base, {"duration": p.duration, **p.base_params})
    b = ScenarioBuilder.from_scenario(base)
    route = longest_boundary_path(base.spec)
    for k, t in enumerate(p.times):
        b.trip(f"ems.{k}", t, route=route, vehicle_type="emergency")
    if p.preemption:
        declared = {ix.id: ix for ix in base.spec.network.intersections}
        for ix in base.resolved.network.intersections:
            if ix.signal is None:
                continue
            sig = declared[ix.id].signal
            data = sig.model_dump(mode="json", exclude_unset=True) if sig is not None else {}
            inner = ix.signal.controller.model_dump(mode="json")
            data["controller"] = {"type": "preemption", "params": {"inner": inner}}
            b.update("intersection", ix.id, signal=data)
    return b.build()
