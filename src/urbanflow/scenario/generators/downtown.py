"""``downtown``: a city centre of avenues and one-way streets (plan E.10).

Junctions ``r{r}c{c}`` lie on a ``rows x cols`` grid ``block`` metres apart. Every
``avenue_every``-th row and column (the middle one of each group) is a two-way avenue
(``avenue_lanes`` lanes per direction); the others are streets with ``street_lanes`` lanes,
one-way in alternating directions with ``one_way_streets`` (even rows eastbound, odd rows
westbound, even columns northbound, odd columns southbound). Each row and column ends in
boundaries ``bW{r}``/``bE{r}``/``bS{c}``/``bN{c}`` one block away. A junction on an avenue is
signalised (``controller``); street x street junctions are priority junctions.

Demand: ``od_rate`` veh/h in total, as origin/destination flows ``"{entry}>{destination}"``
from every boundary entry to (a) the boundary exits on other sides and, for ``core_share``
of the demand, (b) the streets of the central half of the grid. Pairs are weighted by
``lanes(origin) x lanes(destination)``; unreachable pairs are dropped, and if a group has
more pairs than its share of ``max_od_pairs`` they are sampled with ``seed``.
"""

from __future__ import annotations

from collections import deque
from itertools import pairwise
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from urbanflow.core import constants as C
from urbanflow.core.rng import RngStreams
from urbanflow.scenario.builder import ScenarioBuilder
from urbanflow.scenario.generators import register_generator
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import Arrival, SignalTemplate

__all__ = ["DowntownParams", "generate"]


class DowntownParams(BaseModel):
    """Parameters of :func:`generate`."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    rows: int = Field(default=C.DOWNTOWN_ROWS, ge=2, le=C.GRID_MAX_DIM)
    """Rows of junctions (south to north)."""
    cols: int = Field(default=C.DOWNTOWN_COLS, ge=2, le=C.GRID_MAX_DIM)
    """Columns of junctions (west to east)."""
    block: float = Field(default=C.DOWNTOWN_BLOCK, gt=0, le=C.MAX_COORDINATE)
    """Distance between neighbouring junctions, m."""
    avenue_every: int = Field(default=C.DOWNTOWN_AVENUE_EVERY, ge=1)
    """One avenue per this many rows / columns."""
    avenue_lanes: int = Field(default=C.DOWNTOWN_AVENUE_LANES, ge=1, le=C.MAX_LANES_PER_ROAD)
    """Avenue lanes per direction."""
    avenue_speed: float = Field(
        default=C.CORRIDOR_MAIN_SPEED, ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX
    )
    """Avenue speed limit, m/s."""
    street_lanes: int = Field(default=1, ge=1, le=C.MAX_LANES_PER_ROAD)
    """Street lanes (per direction if two-way)."""
    street_speed: float = Field(
        default=C.CORRIDOR_SIDE_SPEED, ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX
    )
    """Street speed limit, m/s."""
    one_way_streets: bool = True
    """Streets are one-way, in alternating directions."""
    controller: str = "fixed_time"
    """Signal controller of the avenue junctions."""
    template: SignalTemplate = "auto"
    """Phase template of the avenue junctions."""
    od_rate: float = Field(default=C.DOWNTOWN_OD_RATE, gt=0)
    """Total demand, veh/h."""
    core_share: float = Field(default=C.DOWNTOWN_CORE_SHARE, ge=0, le=1)
    """Share of the demand that ends on the streets of the centre."""
    max_od_pairs: int = Field(default=C.DOWNTOWN_MAX_OD_PAIRS, ge=1)
    """Origin/destination pairs at most (sampled with ``seed`` above that)."""
    arrival: Arrival = "poisson"
    """Arrival process of every flow."""
    duration: float = Field(default=C.DURATION, gt=0)
    """Simulated duration, s."""
    seed: int = Field(default=0, ge=0)
    """Seed of the pair sampling."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        radius = max(self.avenue_lanes, self.street_lanes) * C.LANE_WIDTH + C.SETBACK
        if self.block - 2 * radius < C.MIN_LANE_LENGTH:
            raise PydanticCustomError(
                "uf_params",
                "block must be at least {n:g} m for these lanes",
                {"n": 2 * radius + C.MIN_LANE_LENGTH},
            )
        return self


def _is_avenue(i: int, every: int) -> bool:
    return i % every == every // 2


@register_generator("downtown", params=DowntownParams)
def generate(p: DowntownParams) -> Scenario:
    """City centre: two-way avenues, one-way streets, mixed junction control."""
    k = p.avenue_every
    b = ScenarioBuilder(
        f"downtown_{p.rows}x{p.cols}",
        description=(
            f"{p.rows}x{p.cols} city centre: avenues every {k} blocks, "
            f"{'one-way' if p.one_way_streets else 'two-way'} streets, signalised avenue "
            "junctions and priority street junctions."
        ),
        duration=p.duration,
    )
    node = [[f"r{r}c{c}" for c in range(p.cols)] for r in range(p.rows)]
    top, right = (p.rows - 1) * p.block, (p.cols - 1) * p.block
    for r in range(p.rows):
        for c in range(p.cols):
            avenue = _is_avenue(r, k) or _is_avenue(c, k)
            b.intersection(
                node[r][c], (c * p.block, r * p.block), kind="signalized" if avenue else "priority"
            )
    for r in range(p.rows):
        b.boundary(f"bW{r}", (-p.block, r * p.block)).boundary(
            f"bE{r}", (right + p.block, r * p.block)
        )
    for c in range(p.cols):
        b.boundary(f"bS{c}", (c * p.block, -p.block)).boundary(
            f"bN{c}", (c * p.block, top + p.block)
        )

    lanes: dict[str, int] = {}
    avenues: set[str] = set()

    def street(line: list[str], i: int) -> None:
        if _is_avenue(i, k):
            spec = (p.avenue_lanes, p.avenue_speed)
        else:
            spec = (p.street_lanes, p.street_speed)
        if _is_avenue(i, k) or not p.one_way_streets:
            for a, z in pairwise(line):
                b.two_way(a, z, lanes=spec[0], speed_limit=spec[1])
                lanes[f"{a}_{z}"] = lanes[f"{z}_{a}"] = spec[0]
                if _is_avenue(i, k):
                    avenues.update((f"{a}_{z}", f"{z}_{a}"))
            return
        seq = line if i % 2 == 0 else line[::-1]
        for a, z in pairwise(seq):
            b.road(f"{a}_{z}", a, z, lanes=spec[0], speed_limit=spec[1])
            lanes[f"{a}_{z}"] = spec[0]

    for r in range(p.rows):  # west -> east (eastbound when one-way and even)
        street([f"bW{r}", *node[r], f"bE{r}"], r)
    for c in range(p.cols):  # south -> north (northbound when one-way and even)
        street([f"bS{c}", *(node[r][c] for r in range(p.rows)), f"bN{c}"], c)

    roads = {rid: tuple(rid.split("_", 1)) for rid in lanes}
    entries = sorted(r for r, (a, _) in roads.items() if a[0] == "b")
    exits = sorted(r for r, (_, z) in roads.items() if z[0] == "b")
    rows_core = range(p.rows // 4, p.rows - p.rows // 4)
    cols_core = range(p.cols // 4, p.cols - p.cols // 4)

    def core(j: str) -> bool:
        if j[0] == "b":
            return False
        r, c = (int(x) for x in j[1:].split("c"))
        return r in rows_core and c in cols_core

    centre = sorted(r for r, (a, z) in roads.items() if core(a) and core(z) and r not in avenues)
    succ = {  # every movement but U-turns is derived at a junction
        rid: [o for o, (a2, z2) in roads.items() if a2 == z and z2 != a]
        for rid, (a, z) in roads.items()
        if z[0] != "b"
    }

    def reachable(origin: str) -> set[str]:
        seen, todo = {origin}, deque([origin])
        while todo:
            for nxt in succ.get(todo.popleft(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return seen

    reach = {e: reachable(e) for e in entries}
    through = [
        (e, x)
        for e in entries
        for x in exits
        if x in reach[e] and roads[e][0][1] != roads[x][1][1]  # another side of the town
    ]
    to_centre = [(e, x) for e in entries for x in centre if x in reach[e]]
    rng = RngStreams(p.seed).stream("generator:downtown")
    n_centre = round(p.max_od_pairs * p.core_share) if to_centre else 0
    groups = (
        (
            through,
            p.od_rate * (1 - p.core_share) if to_centre else p.od_rate,
            p.max_od_pairs - n_centre,
        ),
        (to_centre, p.od_rate * p.core_share, n_centre),
    )
    for pairs, rate, limit in groups:
        if len(pairs) > limit:
            keep = np.sort(rng.choice(len(pairs), size=limit, replace=False))
            pairs = [pairs[i] for i in keep.tolist()]
        if not pairs:
            continue
        weights = [lanes[o] * lanes[d] for o, d in pairs]
        total = sum(weights)
        for (o, d), w in zip(pairs, weights, strict=True):
            b.flow(f"{o}>{d}", origin=o, destination=d, rate=rate * w / total, arrival=p.arrival)
    b.signal_all(controller=p.controller, template=p.template)
    return b.build()
