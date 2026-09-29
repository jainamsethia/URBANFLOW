"""Default resolution: fill every derivable field of a scenario spec (plan E.7 §1.5).

:func:`resolve` is pure, deterministic and idempotent: iteration is sorted and ties are
broken by id. Only the resolved spec reaches the network compiler, so defaults are applied
in exactly one place. Geometry here is plain ``math`` on the reference (median) polylines:
headings of the first/last segments and arc lengths.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import ValidationError

from urbanflow.core.constants import (
    GEOM_EPS,
    OPPOSING_MIN_ANGLE,
    SETBACK,
    STRAIGHT_MAX_ANGLE,
    UTURN_MIN_ANGLE,
)
from urbanflow.core.types import DriveSide, IntersectionKind, TurnKind
from urbanflow.scenario.schema import (
    DEFAULT_VEHICLE_TYPES,
    ConnectionSpec,
    DemandSpec,
    IntersectionSpec,
    MovementSpec,
    NetworkSpec,
    PhaseSpec,
    RoadSpec,
    ScenarioSpec,
    SignalSpec,
    SignalTemplate,
    TransitLineSpec,
    VehicleTypeSpec,
)

__all__ = [
    "SIGNAL_TEMPLATES",
    "RoadGraph",
    "Topology",
    "classify_turns",
    "group_approaches",
    "map_lanes",
    "merge_vehicle_type",
    "polyline_length",
    "resolve",
    "resolve_vehicle_types",
    "road_successors",
    "road_width",
    "shortest_road_path",
    "signal_from_template",
    "topology",
    "turn_table",
    "wrap_angle",
]

Coords = tuple[float, float]
TurnRow = tuple[str, str, float, TurnKind]  # (from_road, to_road, delta, turn)
ByRoad = Mapping[str, Sequence[MovementSpec]]
Green = dict[str, Literal["G", "g"]]  # movement id -> G (protected) / g (permissive)
Greens = list[Green]

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- geometry helpers
def wrap_angle(angle: float) -> float:
    """``angle`` wrapped to ``[-pi, pi)``."""
    return (angle + math.pi) % math.tau - math.pi


def polyline_length(points: Sequence[Coords]) -> float:
    """Arc length of a polyline, m."""
    return sum(math.dist(a, b) for a, b in pairwise(points))


def _end_heading(points: Sequence[Coords]) -> float:
    """Heading of the last non-degenerate segment (radians, CCW from +x)."""
    x1, y1 = points[-1]
    for x0, y0 in reversed(points[:-1]):
        if (x0, y0) != (x1, y1):
            return math.atan2(y1 - y0, x1 - x0)
    return 0.0


def _start_heading(points: Sequence[Coords]) -> float:
    """Heading of the first non-degenerate segment."""
    x0, y0 = points[0]
    for x1, y1 in points[1:]:
        if (x0, y0) != (x1, y1):
            return math.atan2(y1 - y0, x1 - x0)
    return 0.0


def road_width(network: NetworkSpec, road: RoadSpec) -> float:
    """Total width of a road's lanes, m (lane width, else road, else network default)."""
    default = road.lane_width if road.lane_width is not None else network.lane_width
    return sum(lane.width if lane.width is not None else default for lane in road.lanes)


def _road_speed(network: NetworkSpec, road: RoadSpec) -> float:
    return road.speed_limit if road.speed_limit is not None else network.speed_limit


# --------------------------------------------------------------------------- topology
@dataclass(frozen=True, slots=True)
class Topology:
    """Lookups over the placeable part of a spec.

    The first definition of each id wins; roads whose endpoints are unknown or identical
    are left out. ``kinds`` and ``radii`` are the effective (declared, else derived) values.
    """

    intersections: Mapping[str, IntersectionSpec]
    roads: Mapping[str, RoadSpec]
    points: Mapping[str, tuple[Coords, ...]]
    incoming: Mapping[str, tuple[str, ...]]
    outgoing: Mapping[str, tuple[str, ...]]
    kinds: Mapping[str, IntersectionKind]
    radii: Mapping[str, float]


def _derive_kind(n_in: int, n_out: int, n_neighbours: int) -> IntersectionKind:
    # An isolated node (0 neighbours, W101) is inert, so it is treated like a boundary.
    if n_neighbours <= 1:
        return IntersectionKind.boundary
    if n_in <= 2 and n_out <= 2 and n_neighbours == 2:
        return IntersectionKind.uncontrolled
    return IntersectionKind.signalized


def topology(spec: ScenarioSpec) -> Topology:
    """Index the network of ``spec`` (see :class:`Topology`)."""
    net = spec.network
    ints: dict[str, IntersectionSpec] = {}
    for ix in net.intersections:
        ints.setdefault(ix.id, ix)
    roads: dict[str, RoadSpec] = {}
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for road in net.roads:
        if road.id in roads or road.from_ not in ints or road.to not in ints:
            continue
        if road.from_ == road.to:
            continue
        roads[road.id] = road
        outgoing[road.from_].append(road.id)
        incoming[road.to].append(road.id)
    points = {
        rid: road.points or (ints[road.from_].point, ints[road.to].point)
        for rid, road in roads.items()
    }
    kinds: dict[str, IntersectionKind] = {}
    radii: dict[str, float] = {}
    for j, ix in ints.items():
        ins, outs = incoming.get(j, []), outgoing.get(j, [])
        neighbours = {roads[r].from_ for r in ins} | {roads[r].to for r in outs}
        kind = ix.kind or _derive_kind(len(ins), len(outs), len(neighbours))
        kinds[j] = kind
        if ix.radius is not None:
            radii[j] = ix.radius
        elif kind is IntersectionKind.boundary:
            radii[j] = 0.0
        else:
            widest = max((road_width(net, roads[r]) for r in (*ins, *outs)), default=0.0)
            radii[j] = widest + SETBACK
    return Topology(
        intersections=ints,
        roads=roads,
        points=points,
        incoming={j: tuple(v) for j, v in incoming.items()},
        outgoing={j: tuple(v) for j, v in outgoing.items()},
        kinds=kinds,
        radii=radii,
    )


# --------------------------------------------------------------------------- turns and lanes
def turn_table(topo: Topology, j: str) -> list[TurnRow]:
    """Every (from_road, to_road) pair at ``j`` with its heading change and turn kind."""
    rows: list[TurnRow] = []
    for r_in in sorted(topo.incoming.get(j, ())):
        theta_in = _end_heading(topo.points[r_in])
        origin = topo.roads[r_in].from_
        candidates = []
        for r_out in sorted(topo.outgoing.get(j, ())):
            delta = wrap_angle(_start_heading(topo.points[r_out]) - theta_in)
            is_u = topo.roads[r_out].to == origin and abs(delta) >= UTURN_MIN_ANGLE - GEOM_EPS
            candidates.append((r_out, delta, is_u))
        straight = min(
            (
                (abs(d), r)
                for r, d, is_u in candidates
                if not is_u and abs(d) <= STRAIGHT_MAX_ANGLE + GEOM_EPS
            ),
            default=None,
        )
        for r_out, delta, is_u in candidates:
            if is_u:
                turn = TurnKind.uturn
            elif straight is not None and r_out == straight[1]:
                turn = TurnKind.straight
            else:
                turn = TurnKind.left if delta > 0 else TurnKind.right
            rows.append((r_in, r_out, delta, turn))
    return rows


def classify_turns(spec: ScenarioSpec, j: str) -> dict[tuple[str, str], TurnKind]:
    """Turn kind of every (from_road, to_road) pair at intersection ``j`` (U-turns included)."""
    return {(a, b): turn for a, b, _, turn in turn_table(topology(spec), j)}


def _sides(drive_side: DriveSide) -> tuple[TurnKind, TurnKind]:
    """(far-side turn, near-side turn): left/right in right-hand traffic, mirrored otherwise."""
    if drive_side is DriveSide.right:
        return TurnKind.left, TurnKind.right
    return TurnKind.right, TurnKind.left


def map_lanes(
    n_in: int,
    movements: Sequence[tuple[str, float, TurnKind]],
    drive_side: DriveSide,
    n_out: Mapping[str, int],
) -> dict[str, list[tuple[int, int]]]:
    """Lane connections ``{movement id: [(from_lane, to_lane), ...]}`` for one incoming road.

    ``movements`` are ``(id, delta, turn)`` of the road's movements and ``n_out`` the lane
    count of each movement's target road. Movements are ordered from the median side
    outward; with enough lanes each gets a contiguous block (extra lanes go to the straight
    movement, else to the movement with the widest target), otherwise movement ``i`` gets
    lane ``round(i (n-1) / (m-1))`` with halves rounded up (Python's ``round`` would round
    them to even) and a straight movement shares all lanes. U-turns use lane 0. Target
    lanes follow E.7 §1.5 step 6.
    """
    far, _near = _sides(drive_side)
    sign = -1.0 if drive_side is DriveSide.right else 1.0
    uturns = [m for m in movements if m[2] is TurnKind.uturn]
    others = sorted(
        (m for m in movements if m[2] is not TurnKind.uturn), key=lambda m: (sign * m[1], m[0])
    )
    blocks: list[tuple[str, TurnKind, list[int]]] = []
    m = len(others)
    if m and n_in >= m:
        extra_to = next((mid for mid, _, t in others if t is TurnKind.straight), None)
        if extra_to is None:  # widest target; ties keep the first (median-most) movement
            extra_to = max(enumerate(others), key=lambda e: (n_out[e[1][0]], -e[0]))[1][0]
        start = 0
        for mid, _, turn in others:
            size = 1 + (n_in - m if mid == extra_to else 0)
            blocks.append((mid, turn, list(range(start, start + size))))
            start += size
    elif m:
        for i, (mid, _, turn) in enumerate(others):
            lanes = (
                list(range(n_in))
                if turn is TurnKind.straight
                else [math.floor(i * (n_in - 1) / (m - 1) + 0.5)]
            )
            blocks.append((mid, turn, lanes))
    result: dict[str, list[tuple[int, int]]] = {}
    for mid, turn, lanes in blocks:
        size, outs = len(lanes), n_out[mid]
        pairs = []
        for i, k in enumerate(lanes):
            if turn is TurnKind.straight:
                target = min(k, outs - 1)
            elif turn is far:
                target = min(i, outs - 1)
            else:  # near side
                target = max(outs - size + i, 0)
            pairs.append((k, target))
        result[mid] = pairs
    for mid, _, _ in uturns:
        result[mid] = [(0, 0)]
    return result


def group_approaches(bearings: Mapping[str, float]) -> list[tuple[str, ...]]:
    """Pair incoming roads into opposing axes; the rest are singletons.

    Roads pair greedily by descending opposition ``|wrap(b_r - b_s)| >= 135 deg`` (ties by
    id). Groups are ordered by ``min(bearing mod pi)``, then id.
    """
    ids = sorted(bearings)
    pairs = sorted(
        (
            (-abs(wrap_angle(bearings[r] - bearings[s])), r, s)
            for r, s in combinations(ids, 2)
            if abs(wrap_angle(bearings[r] - bearings[s])) >= OPPOSING_MIN_ANGLE - GEOM_EPS
        )
    )
    paired: set[str] = set()
    groups: list[tuple[str, ...]] = []
    for _, r, s in pairs:
        if r not in paired and s not in paired:
            groups.append((r, s))
            paired.update((r, s))
    groups += [(r,) for r in ids if r not in paired]
    return sorted(groups, key=lambda g: (min(bearings[r] % math.pi for r in g), g))


def _green(
    by_road: ByRoad,
    roads: Iterable[str],
    keep: Collection[TurnKind] | None = None,
    weak: Collection[TurnKind] = (),
) -> Green:
    """Green map of the movements out of ``roads`` (turns in ``keep``; ``weak`` ones get g)."""
    return {
        str(mov.id): ("g" if mov.turn in weak else "G")
        for road in roads
        for mov in by_road.get(road, ())
        if keep is None or mov.turn in keep
    }


def _two_phase(group: tuple[str, ...], by_road: ByRoad, far: TurnKind, _near: TurnKind) -> Greens:
    """One phase per group; far-side turns and U-turns are permissive on an axis."""
    return [_green(by_road, group, None, (far, TurnKind.uturn) if len(group) == 2 else ())]


def _protected_left(
    group: tuple[str, ...], by_road: ByRoad, far: TurnKind, near: TurnKind
) -> Greens:
    """An axis gets straight + near, then far + U-turn + near; a singleton gets everything."""
    if len(group) != 2:
        return [_green(by_road, group)]
    return [
        _green(by_road, group, (TurnKind.straight, near)),
        _green(by_road, group, (far, TurnKind.uturn, near)),
    ]


def _split(group: tuple[str, ...], by_road: ByRoad, _far: TurnKind, _near: TurnKind) -> Greens:
    """One phase per incoming road with all of its movements protected."""
    return [_green(by_road, (road,)) for road in group]


SIGNAL_TEMPLATES: Final[
    Mapping[str, Callable[[tuple[str, ...], ByRoad, TurnKind, TurnKind], Greens]]
] = MappingProxyType({"two_phase": _two_phase, "protected_left": _protected_left, "split": _split})
"""Phase templates (E.7 §1.5 step 8) by name: ``(group, movements by road, far, near)`` ->
the green maps of one approach group. Empty maps are dropped."""


def _auto_template(groups: Sequence[tuple[str, ...]]) -> SignalTemplate:
    """``auto``: two_phase with at most one singleton group, otherwise split."""
    return "two_phase" if sum(len(g) == 1 for g in groups) <= 1 else "split"


def signal_from_template(
    signal: SignalSpec,
    groups: Sequence[tuple[str, ...]],
    movements: Sequence[MovementSpec],
    drive_side: DriveSide,
    template: SignalTemplate | None = None,
) -> SignalSpec:
    """``signal`` with phases built from ``template`` (default: ``signal.template``).

    ``groups`` come from :func:`group_approaches`; ``movements`` must have ids and turns.
    Phases without movements are dropped; if none remain, ``signal`` is returned unchanged.
    """
    far, near = _sides(drive_side)
    chosen = template or signal.template
    if chosen == "auto":
        chosen = _auto_template(groups)
    by_road: dict[str, list[MovementSpec]] = defaultdict(list)
    for mov in movements:
        by_road[mov.from_road].append(mov)
    build = SIGNAL_TEMPLATES[chosen]
    greens = [green for group in groups for green in build(group, by_road, far, near) if green]
    # model_construct: movement ids are checked by validation (E006); resolve never raises
    phases = tuple(PhaseSpec.model_construct(id=f"p{k}", green=g) for k, g in enumerate(greens))
    if not phases:
        return signal
    return signal.model_copy(update={"phases": phases})


# --------------------------------------------------------------------------- resolution
def merge_vehicle_type(base: VehicleTypeSpec, override: VehicleTypeSpec) -> VehicleTypeSpec:
    """``base`` with the fields set on ``override`` (``speed_factor`` merges per field).

    Raises pydantic's ``ValidationError`` when the merged type breaks a cross-field rule
    (E007: emergency_decel >= decel, min <= mean <= max).
    """
    data: dict[str, Any] = base.model_dump()
    own = override.model_dump(exclude_unset=True)
    if "speed_factor" in own:
        own["speed_factor"] = {**data["speed_factor"], **own["speed_factor"]}
    return VehicleTypeSpec.model_validate({**data, **own})


def resolve_vehicle_types(declared: Sequence[VehicleTypeSpec]) -> tuple[VehicleTypeSpec, ...]:
    """Built-in types updated with the fields set on declared types of the same id (by id).

    Never raises: an override whose merge is invalid (E007, reported by validation) leaves
    the built-in type unchanged.
    """
    merged: dict[str, VehicleTypeSpec] = dict(DEFAULT_VEHICLE_TYPES)
    seen: set[str] = set()
    for vt in declared:
        if vt.id in seen:  # duplicate (E401): the first definition wins
            continue
        seen.add(vt.id)
        base = merged.get(vt.id)
        if base is None:
            merged[vt.id] = vt
            continue
        try:
            merged[vt.id] = merge_vehicle_type(base, vt)
        except ValidationError:
            continue
    return tuple(merged[k] for k in sorted(merged))


def _resolve_road(net: NetworkSpec, road: RoadSpec, topo: Topology) -> RoadSpec:
    width = road.lane_width if road.lane_width is not None else net.lane_width
    speed = _road_speed(net, road)
    lanes = tuple(
        lane.model_copy(
            update={
                "width": lane.width if lane.width is not None else width,
                "speed_limit": lane.speed_limit if lane.speed_limit is not None else speed,
            }
        )
        for lane in road.lanes
    )
    update: dict[str, Any] = {"lane_width": width, "speed_limit": speed, "lanes": lanes}
    ints = topo.intersections
    if road.points is None and road.from_ in ints and road.to in ints:
        update["points"] = (ints[road.from_].point, ints[road.to].point)
    return road.model_copy(update=update)


def _major_roads(net: NetworkSpec, topo: Topology, j: str) -> tuple[str, ...]:
    incoming = topo.incoming.get(j, ())
    if not incoming:
        return ()
    groups = group_approaches({r: _end_heading(topo.points[r]) for r in incoming})
    axes: list[tuple[str, ...]] = [g for g in groups if len(g) == 2] or groups

    def key(group: tuple[str, ...]) -> tuple[int, float, tuple[str, ...]]:
        lanes = sum(len(topo.roads[r].lanes) for r in group)
        speed = max(_road_speed(net, topo.roads[r]) for r in group)
        return (-lanes, -speed, tuple(sorted(group)))

    best: tuple[str, ...] = min(axes, key=key)
    return tuple(sorted(best))


def _resolve_movements(
    spec: ScenarioSpec, topo: Topology, ix: IntersectionSpec
) -> tuple[MovementSpec, ...]:
    net = spec.network
    table = {(a, b): (d, t) for a, b, d, t in turn_table(topo, ix.id)}
    if ix.movements is None:
        # model_construct: a too-long derived id is E006 in validation, not a crash here
        movements = [
            MovementSpec.model_construct(id=f"{a}->{b}", from_road=a, to_road=b, turn=t)
            for (a, b), (_, t) in sorted(table.items())
            if t is not TurnKind.uturn or net.allow_uturns
        ]
    else:
        movements = []
        for mov in ix.movements:
            key = (mov.from_road, mov.to_road)
            update: dict[str, Any] = {"id": mov.id or f"{key[0]}->{key[1]}"}
            if key in table and mov.turn is None:
                update["turn"] = table[key][1]
            movements.append(mov.model_copy(update=update))
    by_road: dict[str, list[int]] = defaultdict(list)
    for i, mov in enumerate(movements):
        if (mov.from_road, mov.to_road) in table and mov.turn is not None:
            by_road[mov.from_road].append(i)
    for road, idx in by_road.items():
        if all(movements[i].connections is not None for i in idx):
            continue
        rows = [
            (str(movements[i].id), table[(road, movements[i].to_road)][0], movements[i].turn)
            for i in idx
        ]
        n_out = {str(movements[i].id): len(topo.roads[movements[i].to_road].lanes) for i in idx}
        lanes = map_lanes(
            len(topo.roads[road].lanes),
            [(mid, d, t) for mid, d, t in rows if t is not None],
            net.drive_side,
            n_out,
        )
        for i in idx:
            mov = movements[i]
            if mov.connections is None:
                conns = tuple(ConnectionSpec(from_lane=a, to_lane=b) for a, b in lanes[str(mov.id)])
                movements[i] = mov.model_copy(update={"connections": conns})
    return tuple(movements)


def _resolve_intersection(
    spec: ScenarioSpec, topo: Topology, ix: IntersectionSpec
) -> IntersectionSpec:
    net = spec.network
    kind = ix.kind or topo.kinds[ix.id]
    update: dict[str, Any] = {"kind": kind, "radius": topo.radii[ix.id]}
    if ix.radius is not None:
        update["radius"] = ix.radius
    if kind is IntersectionKind.boundary:
        return ix.model_copy(update=update)
    movements = _resolve_movements(spec, topo, ix)
    if kind is IntersectionKind.priority:
        major = ix.major_roads if ix.major_roads is not None else _major_roads(net, topo, ix.id)
        update["major_roads"] = major
        movements = tuple(
            m
            if m.priority is not None
            else m.model_copy(update={"priority": "major" if m.from_road in major else "minor"})
            for m in movements
        )
    update["movements"] = movements
    template: SignalTemplate | None = None  # the template used for derived phases
    if kind is IntersectionKind.signalized:
        signal = ix.signal or SignalSpec()
        if signal.phases is None:
            bearings = {r: _end_heading(topo.points[r]) for r in topo.incoming.get(ix.id, ())}
            groups = group_approaches(bearings)
            template = signal.template if signal.template != "auto" else _auto_template(groups)
            signal = signal_from_template(signal, groups, movements, net.drive_side, template)
        else:
            phases = tuple(
                p if p.id is not None else p.model_copy(update={"id": f"p{k}"})
                for k, p in enumerate(signal.phases)
            )
            signal = signal.model_copy(update={"phases": phases})
        update["signal"] = signal
    _log.debug(
        "intersection %s: kind=%s radius=%.1f movements=%d template=%s",
        ix.id,
        kind,
        update["radius"],
        len(movements),
        template,
        extra={
            "intersection": ix.id,
            "kind": str(kind),
            "radius": update["radius"],
            "n_movements": len(movements),
            "template": template,
        },
    )
    return ix.model_copy(update=update)


def _resolve_demand(demand: DemandSpec, topo: Topology) -> DemandSpec:
    flows = tuple(
        f.model_copy(update={"vehicle_type": "car"})
        if f.vehicle_type is None and f.type_mix is None
        else f
        for f in demand.flows
    )
    transit: list[TransitLineSpec] = []
    for line in demand.transit:
        stops = []
        for k, stop in enumerate(line.stops):
            update: dict[str, Any] = {}
            if stop.id is None:
                update["id"] = f"{line.id}.{k}"
            if stop.lane is None and stop.road in topo.roads:
                update["lane"] = len(topo.roads[stop.road].lanes) - 1
            stops.append(stop.model_copy(update=update) if update else stop)
        transit.append(line.model_copy(update={"stops": tuple(stops)}))
    return demand.model_copy(update={"flows": flows, "transit": tuple(transit)})


def resolve(spec: ScenarioSpec, *, skip: frozenset[str] = frozenset()) -> ScenarioSpec:
    """The fully explicit form of ``spec`` (E.7 §1.5 steps 1-9).

    Intersections in ``skip`` (those touching broken ids during validation) are left as
    declared. Invalid references are tolerated (left unresolved) so validation can report
    them; a valid spec resolves completely.
    """
    topo = topology(spec)
    net = spec.network
    network = net.model_copy(
        update={
            "roads": tuple(_resolve_road(net, r, topo) for r in net.roads),
            "intersections": tuple(
                ix if ix.id in skip else _resolve_intersection(spec, topo, ix)
                for ix in net.intersections
            ),
        }
    )
    return spec.model_copy(
        update={
            "schema_": None,
            "network": network,
            "vehicle_types": resolve_vehicle_types(spec.vehicle_types),
            "demand": _resolve_demand(spec.demand, topo),
        }
    )


# --------------------------------------------------------------------------- road graph
class RoadGraph:
    """Road-to-road successor graph of a *resolved* spec with shortest paths by road length.

    Intersections in ``open_intersections`` connect every incoming road to every outgoing
    road (validation uses this for intersections it could not resolve, to avoid cascades).
    """

    def __init__(self, spec: ScenarioSpec, *, open_intersections: Collection[str] = ()) -> None:
        topo = topology(spec)
        self.lengths: dict[str, float] = {r: polyline_length(p) for r, p in topo.points.items()}
        succ: dict[str, set[str]] = {r: set() for r in topo.roads}
        for ix in spec.network.intersections:
            ins, outs = topo.incoming.get(ix.id, ()), topo.outgoing.get(ix.id, ())
            if ix.id in open_intersections:
                for r in ins:
                    succ[r].update(outs)
            for mov in ix.movements or ():
                if mov.from_road in ins and mov.to_road in outs:
                    succ[mov.from_road].add(mov.to_road)
        self.successors: dict[str, list[str]] = {r: sorted(s) for r, s in succ.items()}

    def tree(
        self, origin: str, target: str | None = None
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Dijkstra from ``origin``: (path length to each reached road, predecessor road).

        Path lengths include the origin road. With ``target`` the search stops once it is
        settled. Ties are broken by road id, so results are deterministic.
        """
        best = {origin: self.lengths[origin]}
        prev: dict[str, str] = {}
        heap = [(best[origin], origin)]
        while heap:
            cost, road = heapq.heappop(heap)
            if road == target:
                break
            if cost > best[road]:
                continue
            for nxt in self.successors[road]:
                new = cost + self.lengths[nxt]
                if new < best.get(nxt, math.inf):
                    best[nxt], prev[nxt] = new, road
                    heapq.heappush(heap, (new, nxt))
        return best, prev

    @staticmethod
    def unwind(prev: Mapping[str, str], origin: str, destination: str) -> list[str]:
        """The road sequence ``origin .. destination`` from a :meth:`tree` predecessor map."""
        path = [destination]
        while path[-1] != origin:
            path.append(prev[path[-1]])
        return path[::-1]

    def _leg(self, origin: str, destination: str) -> list[str] | None:
        best, prev = self.tree(origin, destination)
        return self.unwind(prev, origin, destination) if destination in best else None

    def path(self, origin: str, destination: str, via: Sequence[str] = ()) -> list[str] | None:
        """Shortest road sequence ``origin .. destination`` through ``via`` in order, or None."""
        stops = [origin, *via, destination]
        if any(r not in self.successors for r in stops):
            return None
        route = [origin]
        for a, b in pairwise(stops):
            leg = self._leg(a, b)
            if leg is None:
                return None
            route += leg[1:]
        return route


def road_successors(spec: ScenarioSpec) -> dict[str, list[str]]:
    """``{road: sorted successor roads}`` from the movements of a resolved spec."""
    return RoadGraph(spec).successors


def shortest_road_path(
    spec: ScenarioSpec, origin: str, destination: str, via: Sequence[str] = ()
) -> list[str] | None:
    """Shortest road path by length (heapq Dijkstra, ties by road id) in a resolved spec."""
    return RoadGraph(spec).path(origin, destination, via)
