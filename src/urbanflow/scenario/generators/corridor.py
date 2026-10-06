"""``corridor``: an arterial with side streets and optional green-wave coordination (E.10).

Junctions ``J0 .. J{n-1}`` lie on the x axis ``spacing`` metres apart. The main street runs
from boundary ``W`` to boundary ``E``; each junction ``Ji`` has a side street from ``Ni``
(north) to ``Si`` (south). Every junction is signalised with two fixed-time phases: the
main street gets ``g_m = (C - 2 (yellow + all_red)) * main_share`` and the side street the
rest of the cycle. With ``coordination="green_wave"`` the offsets are
``o_i = round(i * spacing / v_main) mod C``, so a platoon leaving ``W`` at the main speed
meets green lights eastbound.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from urbanflow.core import constants as C
from urbanflow.scenario.builder import ScenarioBuilder
from urbanflow.scenario.generators import register_generator
from urbanflow.scenario.generators._common import TurnRatios, od_flows
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import Arrival

__all__ = ["CorridorParams", "generate"]


def _default_ratios() -> TurnRatios:
    far, straight, near = C.GRID_TURN_RATIOS
    return TurnRatios(far=far, straight=straight, near=near)


class CorridorParams(BaseModel):
    """Parameters of :func:`generate`."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    intersections: int = Field(default=5, ge=1, le=C.CORRIDOR_MAX_JUNCTIONS)
    """Signalised junctions along the main street."""
    spacing: float = Field(default=C.CORRIDOR_SPACING, gt=0, le=C.MAX_COORDINATE)
    """Distance between neighbouring junctions, m."""
    main_lanes: int = Field(default=2, ge=1, le=C.MAX_LANES_PER_ROAD)
    """Main-street lanes per direction."""
    side_lanes: int = Field(default=1, ge=1, le=C.MAX_LANES_PER_ROAD)
    """Side-street lanes per direction."""
    main_speed: float = Field(
        default=C.CORRIDOR_MAIN_SPEED, ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX
    )
    """Main-street speed limit, m/s."""
    side_speed: float = Field(
        default=C.CORRIDOR_SIDE_SPEED, ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX
    )
    """Side-street speed limit, m/s."""
    side_length: float = Field(default=C.CORRIDOR_SIDE_LENGTH, gt=0, le=C.MAX_COORDINATE)
    """Length of the side streets and of the main street's end links, m."""
    cycle: float = Field(default=C.CORRIDOR_CYCLE, gt=0, le=C.PHASE_DURATION_MAX * 2)
    """Signal cycle, s."""
    main_share: float = Field(default=C.CORRIDOR_MAIN_SHARE, gt=0, lt=1)
    """Share of the effective green given to the main street."""
    coordination: Literal["green_wave", "none"] = "green_wave"
    """Offsets for an eastbound green wave, or all offsets 0."""
    main_rate: float = Field(default=C.CORRIDOR_MAIN_RATE, gt=0)
    """Demand entering at each end of the main street, veh/h."""
    side_rate: float = Field(default=C.CORRIDOR_SIDE_RATE, gt=0)
    """Demand entering from each side-street end, veh/h."""
    turn_ratios: TurnRatios = Field(default_factory=_default_ratios)
    """Far-side / straight / near-side shares at the first junction."""
    arrival: Arrival = "poisson"
    """Arrival process of every flow."""
    duration: float = Field(default=C.DURATION, gt=0)
    """Simulated duration, s."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        lost = 2 * (C.SIGNAL_YELLOW + C.SIGNAL_ALL_RED)
        effective = self.cycle - lost
        for share, name in ((self.main_share, "main"), (1 - self.main_share, "side")):
            if effective * share < C.DEFAULT_MIN_GREEN_S:
                raise PydanticCustomError(
                    "uf_params",
                    "the {name} green ({g:g} s) is shorter than min-green; lengthen the cycle",
                    {"name": name, "g": effective * share},
                )
        radius = max(self.main_lanes, self.side_lanes) * C.LANE_WIDTH + C.SETBACK
        if (
            self.spacing - 2 * radius < C.MIN_LANE_LENGTH
            or self.side_length - radius < C.MIN_LANE_LENGTH
        ):
            raise PydanticCustomError("uf_params", "spacing or side_length is too short", {})
        return self


def _green(ins: list[str], outs: list[str], far: dict[str, str]) -> dict[str, str]:
    """Phase states: every movement from ``ins`` (no U-turns); far-side turns permissive."""
    return {
        f"{r}->{o}": ("g" if far.get(r) == o else "G")
        for r in ins
        for o in outs
        if o.split("_", 1)[1] != r.split("_", 1)[0]
    }


@register_generator("corridor", params=CorridorParams)
def generate(p: CorridorParams) -> Scenario:
    """Arterial with side streets; optional green wave."""
    n = p.intersections
    b = ScenarioBuilder(
        f"corridor_{n}",
        description=f"{n}-junction arterial, {p.coordination} coordination.",
        duration=p.duration,
    )
    xs = [i * p.spacing for i in range(n)]
    junction = [f"J{i}" for i in range(n)]
    for i, x in enumerate(xs):
        b.intersection(junction[i], (x, 0.0))
        b.boundary(f"N{i}", (x, p.side_length)).boundary(f"S{i}", (x, -p.side_length))
    b.boundary("W", (-p.side_length, 0.0)).boundary("E", (xs[-1] + p.side_length, 0.0))
    line = ["W", *junction, "E"]
    for a, z in pairwise(line):
        b.two_way(a, z, lanes=p.main_lanes, speed_limit=p.main_speed)
    for i in range(n):
        b.two_way(f"N{i}", junction[i], lanes=p.side_lanes, speed_limit=p.side_speed)
        b.two_way(junction[i], f"S{i}", lanes=p.side_lanes, speed_limit=p.side_speed)

    effective = p.cycle - 2 * (C.SIGNAL_YELLOW + C.SIGNAL_ALL_RED)
    g_main = round(effective * p.main_share)
    g_side = round(effective - g_main)
    for i in range(n):
        j = junction[i]
        west, east = line[i], line[i + 2]
        main_in = [f"{west}_{j}", f"{east}_{j}"]
        side_in = [f"N{i}_{j}", f"S{i}_{j}"]
        outs = [f"{j}_{west}", f"{j}_{east}", f"{j}_N{i}", f"{j}_S{i}"]

        far_main = {main_in[0]: f"{j}_N{i}", main_in[1]: f"{j}_S{i}"}  # left turns
        far_side = {side_in[0]: f"{j}_{east}", side_in[1]: f"{j}_{west}"}
        offset = (
            round(i * p.spacing / p.main_speed) % p.cycle if p.coordination == "green_wave" else 0
        )
        b.signal(
            j,
            phases=[
                {"id": "main", "green": _green(main_in, outs, far_main), "duration": g_main},
                {"id": "side", "green": _green(side_in, outs, far_side), "duration": g_side},
            ],
            offset=float(offset),
        )

    last = n - 1
    exits: dict[str, dict[str, str]] = {
        f"W_{junction[0]}": {
            "straight": f"{junction[last]}_E",
            "far": f"{junction[0]}_N0",
            "near": f"{junction[0]}_S0",
        },
        f"E_{junction[last]}": {
            "straight": f"{junction[0]}_W",
            "far": f"{junction[last]}_S{last}",
            "near": f"{junction[last]}_N{last}",
        },
    }
    rates = {f"W_{junction[0]}": p.main_rate, f"E_{junction[last]}": p.main_rate}
    for i in range(n):
        j = junction[i]
        exits[f"N{i}_{j}"] = {
            "straight": f"{j}_S{i}",
            "far": f"{junction[last]}_E",
            "near": f"{junction[0]}_W",
        }
        exits[f"S{i}_{j}"] = {
            "straight": f"{j}_N{i}",
            "far": f"{junction[0]}_W",
            "near": f"{junction[last]}_E",
        }
        rates[f"N{i}_{j}"] = rates[f"S{i}_{j}"] = p.side_rate
    od_flows(b, list(exits), exits, p.turn_ratios, rates, p.arrival)
    return b.build()
