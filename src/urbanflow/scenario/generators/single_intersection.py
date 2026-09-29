"""``single_intersection``: a 3- or 4-arm junction with any control kind (plan E.10)."""

from __future__ import annotations

from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from urbanflow.core import constants as C
from urbanflow.scenario.builder import ScenarioBuilder
from urbanflow.scenario.generators import register_generator
from urbanflow.scenario.generators._common import TurnRatios, od_flows
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import Arrival, SignalTemplate

__all__ = ["SingleIntersectionParams", "generate"]

# Arms counter-clockwise from east with their unit directions; a 3-arm junction drops S.
_ARMS: Final[dict[str, tuple[int, int]]] = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}
_FAR, _STRAIGHT, _NEAR = -1, 2, 1  # arm-index offsets of the exits (right-hand traffic)


def _default_ratios() -> TurnRatios:
    far, straight, near = C.SINGLE_TURN_RATIOS
    return TurnRatios(far=far, straight=straight, near=near)


class SingleIntersectionParams(BaseModel):
    """Parameters of :func:`generate`."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    arms: Literal[3, 4] = 4
    """Number of arms (3 = T-junction without the south arm)."""
    arm_length: float = Field(default=C.SINGLE_ARM_LENGTH, gt=0, le=C.MAX_COORDINATE)
    """Distance from the centre to each boundary node, m."""
    lanes: int = Field(default=2, ge=1, le=C.MAX_LANES_PER_ROAD)
    """Lanes per road and direction."""
    speed_limit: float = Field(default=C.SPEED_LIMIT, ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX)
    """Speed limit, m/s."""
    kind: Literal["signalized", "priority", "uncontrolled"] = "signalized"
    """Control of the centre node."""
    controller: str = "fixed_time"
    """Signal controller (signalized only)."""
    template: SignalTemplate = "two_phase"
    """Phase template (signalized only)."""
    green: float = Field(
        default=C.PHASE_DURATION, ge=C.DEFAULT_MIN_GREEN_S, le=C.PHASE_DURATION_MAX
    )
    """Green per phase, s."""
    yellow: float = Field(default=C.SIGNAL_YELLOW, ge=0, le=C.SIGNAL_INTERGREEN_MAX)
    """Yellow, s."""
    all_red: float = Field(default=C.SIGNAL_ALL_RED, ge=0, le=C.SIGNAL_INTERGREEN_MAX)
    """All-red, s."""
    demand_rate: float = Field(default=C.SINGLE_DEMAND_RATE, gt=0)
    """Demand per approach, veh/h."""
    approach_rates: dict[str, float] | None = None
    """Per-arm demand overrides, e.g. {"N": 900}, veh/h."""
    turn_ratios: TurnRatios = Field(default_factory=_default_ratios)
    """Far-side / straight / near-side shares of each approach."""
    arrival: Arrival = "poisson"
    """Arrival process of every flow."""
    duration: float = Field(default=C.DURATION, gt=0)
    """Simulated duration, s."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        arms = _arm_names(self.arms)
        unknown = sorted(set(self.approach_rates or {}) - set(arms))
        if unknown:
            raise PydanticCustomError(
                "uf_params",
                "approach_rates has unknown arms {arms} (arms: {valid})",
                {"arms": ", ".join(unknown), "valid": ", ".join(arms)},
            )
        if any(v <= 0 for v in (self.approach_rates or {}).values()):
            raise PydanticCustomError("uf_params", "approach_rates must be positive", {})
        radius = self.lanes * C.LANE_WIDTH + C.SETBACK
        if self.arm_length - radius < C.MIN_LANE_LENGTH:
            raise PydanticCustomError(
                "uf_params",
                "arm_length must be at least {n:g} m for {lanes} lanes",
                {"n": radius + C.MIN_LANE_LENGTH, "lanes": self.lanes},
            )
        peak = max([self.demand_rate, *(self.approach_rates or {}).values()])
        if self.arrival == "binomial" and peak > C.SECONDS_PER_HOUR / C.DT:
            raise PydanticCustomError(
                "uf_params",
                "binomial arrivals allow at most {max:g} veh/h per approach at dt={dt:g} s",
                {"max": C.SECONDS_PER_HOUR / C.DT, "dt": C.DT},
            )
        return self


def _arm_names(arms: int) -> list[str]:
    return list(_ARMS) if arms == len(_ARMS) else [a for a in _ARMS if a != "S"]


@register_generator("single_intersection", params=SingleIntersectionParams)
def generate(p: SingleIntersectionParams) -> Scenario:
    """3/4-arm junction, any control kind."""
    arms = _arm_names(p.arms)
    b = ScenarioBuilder(
        "single_intersection",
        description=f"{p.arms}-arm {p.kind} intersection, {p.lanes} lane(s) per direction.",
        speed_limit=p.speed_limit,
        duration=p.duration,
    )
    b.intersection("J", (0.0, 0.0), kind=p.kind)
    for arm in arms:
        dx, dy = _ARMS[arm]
        b.boundary(arm, (dx * p.arm_length, dy * p.arm_length))
        b.road(f"{arm}_in", arm, "J", lanes=p.lanes)
        b.road(f"{arm}_out", "J", arm, lanes=p.lanes)
    if p.kind == "signalized":
        b.signal(
            "J",
            controller=p.controller,
            template=p.template,
            green=p.green,
            yellow=p.yellow,
            all_red=p.all_red,
        )
    ring = list(_ARMS)
    exits: dict[str, dict[str, str]] = {}
    for arm in arms:
        i = ring.index(arm)
        turns = {"far": _FAR, "straight": _STRAIGHT, "near": _NEAR}
        exits[f"{arm}_in"] = {
            turn: f"{ring[(i + offset) % len(ring)]}_out"
            for turn, offset in turns.items()
            if ring[(i + offset) % len(ring)] in arms
        }
    rates = {f"{a}_in": (p.approach_rates or {}).get(a, p.demand_rate) for a in arms}
    od_flows(
        b,
        [f"{a}_in" for a in arms],
        exits,
        p.turn_ratios,
        rates,
        p.arrival,
        flow_id=lambda entry, _turn, exit_road: (
            f"{entry.removesuffix('_in')}_to_{exit_road.removesuffix('_out')}"
        ),
    )
    return b.build()
