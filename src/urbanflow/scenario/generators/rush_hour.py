"""``rush_hour``: a base scenario's demand shaped into a directional peak (E.10).

Every flow ``f`` of the base scenario is split into slices ``"{f}:{k}"`` with windows
``[k slice, (k+1) slice)`` running at ``r_k = r_f m(t_k) d_f`` (``t_k`` = the slice midpoint):

    m(t) = f_off + (1 - f_off) max(0, 1 - |t - t_peak| / w)
    d_f  = 1 + beta cos(theta_f - phi)

where ``theta_f`` is the bearing of the flow's origin-to-destination vector (degrees
counter-clockwise from east) and ``phi = peak_direction_deg``; ``peak_direction_deg=None``
gives ``d_f = 1``. Flows with ``count`` stay whole.
"""

from __future__ import annotations

import math
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from urbanflow.core import constants as C
from urbanflow.scenario.generators import generate as run_generator
from urbanflow.scenario.generators import register_generator
from urbanflow.scenario.generators._common import split_flows_by_profile
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import FlowSpec, ScenarioSpec

__all__ = ["RushHourParams", "generate"]


class RushHourParams(BaseModel):
    """Parameters of :func:`generate`."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    base: str = "corridor"
    """Generator of the underlying scenario."""
    base_params: dict[str, Any] = Field(default_factory=dict)
    """Its parameters (``duration`` comes from this generator)."""
    duration: float = Field(default=C.RUSH_HOUR_DURATION, gt=0)
    """Simulated duration, s."""
    peak_time: float = Field(default=C.RUSH_HOUR_PEAK_TIME, ge=0)
    """Time of the demand peak, s."""
    peak_width: float = Field(default=C.RUSH_HOUR_PEAK_WIDTH, gt=0)
    """Half-width of the triangular peak, s."""
    off_peak_factor: float = Field(default=C.RUSH_HOUR_OFF_PEAK_FACTOR, ge=0, le=1)
    """Demand outside the peak as a share of the base rate."""
    slice: float = Field(default=C.RUSH_HOUR_SLICE, gt=0)
    """Length of the constant-rate slices, s."""
    peak_direction_deg: float | None = 0.0
    """Bearing of the peak direction (0 = eastbound, 90 = northbound); null = none."""
    directional_strength: float = Field(default=C.RUSH_HOUR_DIRECTIONAL_STRENGTH, ge=0, le=1)
    """beta: flows along the peak direction get ``1 + beta``, against it ``1 - beta``."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.base in {"rush_hour", "emergency"}:
            raise PydanticCustomError("uf_params", "base cannot be {base}", {"base": self.base})
        if self.slice > self.duration:
            raise PydanticCustomError("uf_params", "slice is longer than the duration", {})
        return self


def _bearings(spec: ScenarioSpec) -> dict[str, float | None]:
    """Origin-to-destination bearing of every flow, radians (None for a closed loop)."""
    point = {ix.id: ix.point for ix in spec.network.intersections}
    roads = {r.id: (r.from_, r.to) for r in spec.network.roads}
    out: dict[str, float | None] = {}
    for f in spec.demand.flows:
        path = f.route or (f.routes[0].roads if f.routes else (f.origin, f.destination))
        (x0, y0), (x1, y1) = point[roads[str(path[0])][0]], point[roads[str(path[-1])][1]]
        out[f.id] = math.atan2(y1 - y0, x1 - x0) if (x1, y1) != (x0, y0) else None
    return out


@register_generator("rush_hour", params=RushHourParams)
def generate(p: RushHourParams) -> Scenario:
    """A base scenario with a directional, time-varying demand peak."""
    base = run_generator(p.base, {"duration": p.duration, **p.base_params})
    spec = base.spec
    bearing = _bearings(spec)
    phi = None if p.peak_direction_deg is None else math.radians(p.peak_direction_deg)

    def profile(flow: FlowSpec, t: float) -> float:
        peak = max(0.0, 1.0 - abs(t - p.peak_time) / p.peak_width)
        m = p.off_peak_factor + (1.0 - p.off_peak_factor) * peak
        theta = bearing[flow.id]
        d = (
            1.0
            if phi is None or theta is None
            else 1.0 + p.directional_strength * math.cos(theta - phi)
        )
        return m * d

    shaped = split_flows_by_profile(spec, profile, p.slice)
    meta = shaped.meta.model_copy(
        update={
            "name": f"rush_hour_{shaped.meta.name}",
            "description": f"{shaped.meta.description} Rush-hour demand peaking at "
            f"{p.peak_time:g} s.".strip(),
        }
    )
    out = shaped.model_copy(update={"meta": meta})  # copies skip validation: re-validate
    return Scenario.from_dict(out.model_dump(mode="json", by_alias=True, exclude_unset=True))
