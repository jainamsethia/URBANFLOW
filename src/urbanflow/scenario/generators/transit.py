"""``transit``: bus lines with far-side stops across a base scenario (plan E.10).

The base generator's scenario keeps its mixed traffic. Line ``bus_fwd`` runs along the
longest boundary-to-boundary route (the one with the fewest turns among equally long ones,
since a turn leaves the bus away from the curb) and, with ``bidirectional``, ``bus_rev``
along the same streets the other way (when every road has an opposite road). Each road of
a line that starts at a junction gets a far-side stop ``stop_position`` metres after it, on
the curb lane, if the road is long enough to stop there and still have
``STOP_NEAR_LINE_DISTANCE`` before its stop line (so the bus can reach the lane it needs
afterwards).
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from urbanflow.core import constants as C
from urbanflow.core.types import IntersectionKind, TurnKind
from urbanflow.scenario.builder import ScenarioBuilder
from urbanflow.scenario.derive import RoadGraph, polyline_length, topology
from urbanflow.scenario.generators import generate as run_generator
from urbanflow.scenario.generators import register_generator
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import DEFAULT_VEHICLE_TYPES, ScenarioSpec

__all__ = ["TransitParams", "generate"]


class TransitParams(BaseModel):
    """Parameters of :func:`generate`."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    base: str = "corridor"
    """Generator of the underlying scenario."""
    base_params: dict[str, Any] = Field(default_factory=dict)
    """Its parameters (``duration`` defaults to this generator's)."""
    headway: float = Field(default=C.TRANSIT_HEADWAY, gt=0)
    """Seconds between buses of a line."""
    stop_position: float = Field(default=C.TRANSIT_STOP_POSITION, gt=0)
    """Where the bus front stops after each junction, m (at least the bus length)."""
    dwell: float = Field(default=C.STOP_DWELL, ge=0)
    """Dwell time at every stop, s."""
    bidirectional: bool = True
    """Also run a line the opposite way."""
    duration: float = Field(default=C.DURATION, gt=0)
    """Simulated duration, s."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.base == "transit":
            raise PydanticCustomError("uf_params", "base cannot be transit", {})
        bus = DEFAULT_VEHICLE_TYPES["bus"].length
        if self.stop_position < bus:
            raise PydanticCustomError(
                "uf_params", "stop_position must be at least the bus length ({n:g} m)", {"n": bus}
            )
        return self


def bus_route(resolved: ScenarioSpec) -> list[str]:
    """The longest shortest path from a boundary entry to a boundary exit of a *resolved*
    spec, with the fewest turns among equally long ones (then by entry and exit ids)."""
    topo = topology(resolved)
    graph = RoadGraph(resolved)
    turn = {
        (m.from_road, m.to_road): m.turn
        for ix in resolved.network.intersections
        for m in ix.movements or ()
    }
    boundary = {j for j, kind in topo.kinds.items() if kind is IntersectionKind.boundary}
    exits = sorted(r for r, road in topo.roads.items() if road.to in boundary)
    best: tuple[float, int, list[str]] = (-math.inf, 0, [])
    for entry in sorted(r for r, road in topo.roads.items() if road.from_ in boundary):
        lengths, prev = graph.tree(entry)
        for exit_road in exits:
            if exit_road not in lengths:
                continue
            path = graph.unwind(prev, entry, exit_road)
            turns = sum(turn.get(pair) is not TurnKind.straight for pair in pairwise(path))
            key = (round(lengths[exit_road], 6), -turns)
            if key > best[:2]:
                best = (*key, path)
    return best[2]


@register_generator("transit", params=TransitParams)
def generate(p: TransitParams) -> Scenario:
    """Bus lines with far-side stops across a base scenario."""
    base = run_generator(p.base, {"duration": p.duration, **p.base_params})
    topo = topology(base.resolved)
    ends = {r: (road.from_, road.to) for r, road in topo.roads.items()}
    opposite = {(a, z): r for r, (a, z) in ends.items()}
    forward = bus_route(base.resolved)
    lines = [("bus_fwd", forward)]
    backward = [opposite.get(ends[r][::-1]) for r in reversed(forward)]
    if p.bidirectional and all(backward):
        lines.append(("bus_rev", [str(r) for r in backward]))

    def lane_length(road: str) -> float:
        a, z = ends[road]
        trim = sum(
            float(topo.radii[j]) for j in (a, z) if topo.kinds[j] is not IntersectionKind.boundary
        )
        return polyline_length(topo.points[road]) - trim

    b = ScenarioBuilder.from_scenario(base)
    for line_id, route in lines:
        stops = [
            (road, p.stop_position)
            for road in route[1:]
            if lane_length(road) >= p.stop_position + C.STOP_NEAR_LINE_DISTANCE
        ]
        if stops:
            b.transit_line(line_id, route, stops, headway=p.headway, dwell=p.dwell)
    return b.build()
