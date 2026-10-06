"""``grid``: a rows x cols signalised grid with optional non-uniform spacing (plan E.10)."""

from __future__ import annotations

from itertools import accumulate, pairwise
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from urbanflow.core import constants as C
from urbanflow.scenario.builder import ScenarioBuilder
from urbanflow.scenario.generators import register_generator
from urbanflow.scenario.generators._common import TurnRatios, od_flows
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import Arrival, SignalTemplate

__all__ = ["GridParams", "generate"]


def _default_ratios() -> TurnRatios:
    far, straight, near = C.GRID_TURN_RATIOS
    return TurnRatios(far=far, straight=straight, near=near)


class GridParams(BaseModel):
    """Parameters of :func:`generate`."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    rows: int = Field(default=3, ge=1, le=C.GRID_MAX_DIM)
    """Rows of intersections (south to north)."""
    cols: int = Field(default=3, ge=1, le=C.GRID_MAX_DIM)
    """Columns of intersections (west to east)."""
    spacing: float = Field(default=C.GRID_SPACING, gt=0, le=C.MAX_COORDINATE)
    """Distance between neighbouring intersections, m (unless row/col spacing is given)."""
    row_spacing: list[float] | None = None
    """Non-uniform distances between consecutive rows (rows - 1 values), m."""
    col_spacing: list[float] | None = None
    """Non-uniform distances between consecutive columns (cols - 1 values), m."""
    lanes: int = Field(default=2, ge=1, le=C.MAX_LANES_PER_ROAD)
    """Lanes per road and direction."""
    speed_limit: float = Field(default=C.SPEED_LIMIT, ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX)
    """Speed limit, m/s."""
    boundary_length: float = Field(default=C.GRID_BOUNDARY_LENGTH, gt=0, le=C.MAX_COORDINATE)
    """Length of the entry/exit roads at the edge of the grid, m."""
    controller: str = "fixed_time"
    """Signal controller of every intersection."""
    template: SignalTemplate = "two_phase"
    """Phase template."""
    green: float = Field(
        default=C.PHASE_DURATION, ge=C.DEFAULT_MIN_GREEN_S, le=C.PHASE_DURATION_MAX
    )
    """Green per phase, s."""
    entry_rate: float = Field(default=C.GRID_ENTRY_RATE, gt=0)
    """Demand per entry road, veh/h."""
    turn_ratios: TurnRatios = Field(default_factory=_default_ratios)
    """Shares of each entry's demand that turn far-side / go straight / turn near-side at
    the first intersection (and then go straight to the edge)."""
    arrival: Arrival = "poisson"
    """Arrival process of every flow."""
    duration: float = Field(default=C.DURATION, gt=0)
    """Simulated duration, s."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        for name, values, n in (
            ("row_spacing", self.row_spacing, self.rows),
            ("col_spacing", self.col_spacing, self.cols),
        ):
            if values is not None and len(values) != n - 1:
                raise PydanticCustomError(
                    "uf_params",
                    "{name} needs {k} values (got {got})",
                    {"name": name, "k": n - 1, "got": len(values)},
                )
        radius = self.lanes * C.LANE_WIDTH + C.SETBACK
        gaps = [*(self.row_spacing or [self.spacing]), *(self.col_spacing or [self.spacing])]
        if min(gaps) - 2 * radius < C.MIN_LANE_LENGTH:
            raise PydanticCustomError(
                "uf_params",
                "spacing must be at least {n:g} m for {lanes} lanes",
                {"n": 2 * radius + C.MIN_LANE_LENGTH, "lanes": self.lanes},
            )
        if self.boundary_length - radius < C.MIN_LANE_LENGTH:
            raise PydanticCustomError(
                "uf_params",
                "boundary_length must be at least {n:g} m for {lanes} lanes",
                {"n": radius + C.MIN_LANE_LENGTH, "lanes": self.lanes},
            )
        return self


def _offsets(n: int, spacing: float, explicit: list[float] | None) -> list[float]:
    return [0.0, *accumulate(explicit if explicit is not None else [spacing] * (n - 1))]


@register_generator("grid", params=GridParams)
def generate(p: GridParams) -> Scenario:
    """R x C signalized grid with OD flows from every edge entry."""
    ys = _offsets(p.rows, p.spacing, p.row_spacing)
    xs = _offsets(p.cols, p.spacing, p.col_spacing)
    node = [[f"r{r}c{c}" for c in range(p.cols)] for r in range(p.rows)]
    b = ScenarioBuilder(
        f"grid_{p.rows}x{p.cols}",
        description=f"{p.rows}x{p.cols} signalized grid, {p.lanes} lane(s) per direction.",
        speed_limit=p.speed_limit,
        duration=p.duration,
    )
    for r in range(p.rows):
        for c in range(p.cols):
            b.intersection(node[r][c], (xs[c], ys[r]))
    edge = p.boundary_length
    for r in range(p.rows):
        b.boundary(f"bW{r}", (-edge, ys[r])).boundary(f"bE{r}", (xs[-1] + edge, ys[r]))
    for c in range(p.cols):
        b.boundary(f"bS{c}", (xs[c], -edge)).boundary(f"bN{c}", (xs[c], ys[-1] + edge))
    for r in range(p.rows):  # east-west streets
        line = [f"bW{r}", *node[r], f"bE{r}"]
        for a, z in pairwise(line):
            b.two_way(a, z, lanes=p.lanes)
    for c in range(p.cols):  # north-south avenues
        line = [f"bS{c}", *(node[r][c] for r in range(p.rows)), f"bN{c}"]
        for a, z in pairwise(line):
            b.two_way(a, z, lanes=p.lanes)

    top, right = p.rows - 1, p.cols - 1
    # entry road -> {turn: exit road}; turning happens once, at the first intersection.
    exits: dict[str, dict[str, str]] = {}
    for r in range(p.rows):
        exits[f"bW{r}_{node[r][0]}"] = {  # heading east: far = north, near = south
            "straight": f"{node[r][right]}_bE{r}",
            "far": f"{node[top][0]}_bN0",
            "near": f"{node[0][0]}_bS0",
        }
        exits[f"bE{r}_{node[r][right]}"] = {  # heading west: far = south, near = north
            "straight": f"{node[r][0]}_bW{r}",
            "far": f"{node[0][right]}_bS{right}",
            "near": f"{node[top][right]}_bN{right}",
        }
    for c in range(p.cols):
        exits[f"bS{c}_{node[0][c]}"] = {  # heading north: far = west, near = east
            "straight": f"{node[top][c]}_bN{c}",
            "far": f"{node[0][0]}_bW0",
            "near": f"{node[0][right]}_bE0",
        }
        exits[f"bN{c}_{node[top][c]}"] = {  # heading south: far = east, near = west
            "straight": f"{node[0][c]}_bS{c}",
            "far": f"{node[top][right]}_bE{top}",
            "near": f"{node[top][0]}_bW{top}",
        }
    od_flows(b, list(exits), exits, p.turn_ratios, p.entry_rate, p.arrival)
    b.signal_all(controller=p.controller, template=p.template, green=p.green)
    return b.build()
