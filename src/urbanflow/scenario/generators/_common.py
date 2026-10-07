"""Helpers shared by the generators (plan E.10 §4.1)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from urbanflow.core.constants import RATIO_SUM_TOLERANCE, SECONDS_PER_HOUR, TIME_EPS
from urbanflow.core.types import IntersectionKind
from urbanflow.scenario.builder import ScenarioBuilder
from urbanflow.scenario.derive import RoadGraph, resolve, topology
from urbanflow.scenario.schema import Arrival, FlowSpec, ScenarioSpec

__all__ = [
    "TURNS",
    "Turn",
    "TurnRatios",
    "longest_boundary_path",
    "od_flows",
    "split_flows_by_profile",
]

Turn = Literal["far", "straight", "near"]
TURNS: Final[tuple[Turn, ...]] = ("far", "straight", "near")


class TurnRatios(BaseModel):
    """Shares of an approach's demand that turn far-side, go straight or turn near-side."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    far: float = Field(ge=0, le=1)
    """Far-side turn share (left in right-hand traffic)."""
    straight: float = Field(ge=0, le=1)
    """Straight-on share."""
    near: float = Field(ge=0, le=1)
    """Near-side turn share (right in right-hand traffic)."""

    @model_validator(mode="after")
    def _sums_to_one(self) -> Self:
        total = self.far + self.straight + self.near
        if abs(total - 1.0) > RATIO_SUM_TOLERANCE:
            raise PydanticCustomError(
                "uf_ratios", "turn ratios must sum to 1 (got {total})", {"total": total}
            )
        return self


def _default_flow_id(entry: str, turn: str, _exit: str) -> str:
    return f"{entry}:{turn}"


def od_flows(
    b: ScenarioBuilder,
    entries: Sequence[str],
    exits_by_turn: Mapping[str, Mapping[str, str]],
    ratios: TurnRatios,
    rate: float | Mapping[str, float],
    arrival: Arrival = "poisson",
    begin: float = 0.0,
    end: float | None = None,
    *,
    flow_id: Callable[[str, str, str], str] = _default_flow_id,
) -> list[str]:
    """Add one origin/destination flow per (entry road, available turn); return their ids.

    ``exits_by_turn[entry]`` maps ``far``/``straight``/``near`` to the exit road. When an
    entry lacks some turn (e.g. at a T-junction) the remaining ratios are renormalised, so
    each entry still receives its full ``rate`` (veh/h, per entry or one value for all).
    Flow ids come from ``flow_id(entry, turn, exit)`` (default ``"{entry}:{turn}"``).
    """
    created: list[str] = []
    for entry in entries:
        exits = exits_by_turn.get(entry, {})
        shares = {t: getattr(ratios, t) for t in TURNS if t in exits and getattr(ratios, t) > 0}
        total = sum(shares.values())
        entry_rate = rate[entry] if isinstance(rate, Mapping) else rate
        for turn, share in shares.items():
            fid = flow_id(entry, turn, exits[turn])
            b.flow(
                fid,
                origin=entry,
                destination=exits[turn],
                rate=entry_rate * share / total,
                arrival=arrival,
                begin=begin,
                end=end,
            )
            created.append(fid)
    return created


def longest_boundary_path(spec: ScenarioSpec) -> list[str]:
    """The longest of the shortest paths from a boundary entry road to a boundary exit road.

    Ties are broken by (entry, exit) ids. Returns ``[]`` if no exit is reachable.
    """
    resolved = resolve(spec)
    topo = topology(resolved)
    graph = RoadGraph(resolved)
    boundary = {j for j, kind in topo.kinds.items() if kind is IntersectionKind.boundary}
    exits = sorted(r for r, road in topo.roads.items() if road.to in boundary)
    best: tuple[float, list[str]] = (-math.inf, [])
    for entry in sorted(r for r, road in topo.roads.items() if road.from_ in boundary):
        lengths, prev = graph.tree(entry)
        for exit_road in exits:
            if exit_road in lengths and lengths[exit_road] > best[0]:
                best = (lengths[exit_road], graph.unwind(prev, entry, exit_road))
    return best[1]


def split_flows_by_profile(
    spec: ScenarioSpec, profile: Callable[[FlowSpec, float], float], slice: float
) -> ScenarioSpec:
    """Replace every flow by slices ``"{f}:{k}"`` with windows ``[k slice, (k+1) slice)``.

    Slice ``k`` runs at ``rate_f * profile(f, t)`` with ``t`` the midpoint of the slice's
    window clipped to the flow's own ``[begin, end)`` (end defaults to the simulation
    duration). Slices with a non-positive rate are dropped.
    ponytail: flows with ``count`` or without any end stay whole; a time-varying
    ``FlowSpec.profile`` field (AG.3 #10) would make this exact.
    """
    flows: list[FlowSpec] = []
    for flow in spec.demand.flows:
        end = flow.end if flow.end is not None else spec.simulation.duration
        if end is None or flow.count is not None:
            flows.append(flow)
            continue
        base = flow.rate if flow.rate is not None else SECONDS_PER_HOUR / float(flow.period or 1)
        for k in range(math.floor(flow.begin / slice), math.ceil(end / slice - TIME_EPS)):
            lo, hi = max(k * slice, flow.begin), min((k + 1) * slice, end)
            if hi - lo <= TIME_EPS:
                continue
            rate = base * profile(flow, (lo + hi) / 2)
            if rate > 0:
                update = {
                    "id": f"{flow.id}:{k}",
                    "rate": rate,
                    "period": None,
                    "begin": lo,
                    "end": hi,
                }
                flows.append(flow.model_copy(update=update))
    demand = spec.demand.model_copy(update={"flows": tuple(flows)})
    return spec.model_copy(update={"demand": demand})
