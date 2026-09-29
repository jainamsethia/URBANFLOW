"""Compile a validated scenario into a :class:`CompiledNetwork` (plan E.5 rules 1-6, E.3).

The input is ``Scenario.resolved``, so the compiler applies no defaults of its own. Steps:
trim roads and offset lanes (``lanes.py``), build connectors (``connectors.py``), lay out
the link arrays, CSR structures and the global arc key, compute conflicts
(``conflicts.py``), then run the compile diagnostics. Errors (E802 short road, E905
degenerate geometry, E806 lane too short for a routed vehicle type) raise
``ScenarioValidationError`` with JSON paths; warnings (W304 double crossing, W302 crossing
protected movements) are kept in ``CompiledNetwork.report``.
"""

from __future__ import annotations

import logging
import math
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.errors import ScenarioValidationError, Severity, ValidationIssue
from urbanflow.core.logging import log_duration
from urbanflow.core.types import DriveSide, IntArray, IntersectionKind, LinkKind, TurnKind
from urbanflow.geometry import Polyline
from urbanflow.network.compiled import CompiledNetwork, CompileReport
from urbanflow.network.conflicts import ConflictKind, ConflictTable, compute_conflicts
from urbanflow.network.connectors import build_connector, connector_speed_limit, direction
from urbanflow.network.lanes import (
    DegenerateGeometryError,
    LaneGeom,
    build_lanes,
    resolved_value,
    trim_distance,
    trim_road,
)
from urbanflow.scenario.derive import RoadGraph
from urbanflow.scenario.schema import IntersectionSpec, MovementSpec, ScenarioSpec
from urbanflow.scenario.validate import issue

if TYPE_CHECKING:
    from urbanflow.scenario.scenario import Scenario

__all__ = ["CompileReport", "compile_network", "static_rank"]

_log = logging.getLogger("urbanflow.network")  # AB 6.4 "network compiled"


@dataclass(frozen=True, slots=True)
class _Roads:
    """Trimmed references and lanes of every road, in road order."""

    reference: list[Polyline]
    lanes: list[list[LaneGeom]]
    lane_start: list[int]

    @property
    def n_lanes(self) -> int:
        return self.lane_start[-1]


@dataclass(frozen=True, slots=True)
class _Connector:
    id: str
    movement: int
    intersection: int
    from_lane: int
    to_lane: int
    line: Polyline
    speed_limit: float
    width: float


def _raise_on_errors(issues: Sequence[ValidationIssue], scenario: Scenario) -> None:
    if any(i.severity is Severity.error for i in issues):
        source = str(scenario.path) if scenario.path is not None else scenario.name
        raise ScenarioValidationError(issues, source)


def _i32(values: Sequence[int] | IntArray) -> IntArray:
    return np.asarray(values, dtype=np.int32).reshape(-1)


def _csr(groups: Sequence[Sequence[int]]) -> tuple[IntArray, IntArray]:
    """(ptr, values) of a list of integer lists."""
    ptr = np.zeros(len(groups) + 1, dtype=np.int32)
    ptr[1:] = np.cumsum([len(g) for g in groups])
    return ptr, _i32([v for g in groups for v in g])


def _bearing(dx: float, dy: float) -> float:
    """Compass bearing of a direction: clockwise from north (+y), in [0, 2π).

    Bearings within ``GEOM_EPS`` below 2π (float noise around due north) wrap to 0.
    """
    bearing = math.atan2(dx, dy) % math.tau
    return 0.0 if bearing > math.tau - C.GEOM_EPS else bearing


def static_rank(kind: IntersectionKind, movement: MovementSpec, drive_side: DriveSide) -> int:
    """``mov_static_rank`` (E.7 §1.4): 0 signalised (rank comes from the signal state);
    priority: major straight or near-side turn 3, major far-side turn or U-turn 2, minor 1;
    uncontrolled 1."""
    if kind is IntersectionKind.signalized:
        return 0
    if kind is not IntersectionKind.priority or movement.priority != "major":
        return 1
    far = TurnKind.left if drive_side is DriveSide.right else TurnKind.right
    return 2 if movement.turn in (far, TurnKind.uturn) else 3


# --------------------------------------------------------------------------- rules 2-3
def _build_roads(spec: ScenarioSpec, ints: dict[str, IntersectionSpec]) -> _Roads:
    net = spec.network
    reference: list[Polyline] = []
    lanes: list[list[LaneGeom]] = []
    issues: list[ValidationIssue] = []
    for i, road in enumerate(net.roads):
        a, b = ints[road.from_], ints[road.to]
        ref = Polyline(resolved_value(road.points, f'points of road "{road.id}"'))
        ta, tb = trim_distance(ref.points[0], a), trim_distance(ref.points[-1], b)
        remaining = ref.length - ta - tb
        if remaining < C.MIN_LANE_LENGTH:
            issues.append(
                issue(
                    "E802",
                    ("network", "roads", i),
                    L=ref.length,
                    a=a.id,
                    b=b.id,
                    ra=ta,
                    rb=tb,
                    rem=remaining,
                    min=C.MIN_LANE_LENGTH,
                )
            )
            reference.append(ref)
            lanes.append([])
            continue
        trimmed = trim_road(ref, ta, tb)
        reference.append(trimmed)
        try:
            lanes.append(build_lanes(road, trimmed, net.drive_side))
        except DegenerateGeometryError as exc:
            issues.append(issue("E905", ("network", "roads", i), detail=str(exc)))
            lanes.append([])
    if issues:
        raise ScenarioValidationError(issues)
    start = np.concatenate(([0], np.cumsum([len(r.lanes) for r in net.roads])))
    return _Roads(reference, lanes, [int(v) for v in start])


# --------------------------------------------------------------------------- rule 4
def _build_connectors(
    scenario: Scenario, roads: _Roads, road_index: dict[str, int]
) -> list[_Connector]:
    out: list[_Connector] = []
    issues: list[ValidationIssue] = []
    declared = scenario.spec.network.intersections
    m = 0
    for j, ix in enumerate(scenario.resolved.network.intersections):
        # derived movements are not in the user's file: report those at the intersection
        explicit = declared[j].movements is not None
        for k, mov in enumerate(ix.movements or ()):
            r_in, r_out = road_index[mov.from_road], road_index[mov.to_road]
            conns = resolved_value(mov.connections, f'connections of movement "{mov.id}"')
            for c, conn in enumerate(conns):
                lane_in = roads.lanes[r_in][conn.from_lane]
                lane_out = roads.lanes[r_out][conn.to_lane]
                cid = f"{mov.from_road}_{conn.from_lane}->{mov.to_road}_{conn.to_lane}"
                try:
                    line = build_connector(
                        lane_in.line.points[-1],
                        direction(float(lane_in.line.headings[-1])),
                        lane_out.line.points[0],
                        direction(float(lane_out.line.headings[0])),
                        conn.shape,
                    )
                except DegenerateGeometryError as exc:
                    path: tuple[str | int, ...] = ("network", "intersections", j)
                    if explicit:
                        path += ("movements", k, "connections", c)
                    issues.append(issue("E905", path, detail=f'connector "{cid}": {exc}'))
                    continue
                speed = connector_speed_limit(
                    line.points, lane_in.speed_limit, lane_out.speed_limit
                )
                out.append(
                    _Connector(
                        id=cid,
                        movement=m,
                        intersection=j,
                        from_lane=roads.lane_start[r_in] + conn.from_lane,
                        to_lane=roads.lane_start[r_out] + conn.to_lane,
                        line=line,
                        speed_limit=speed,
                        width=lane_in.width,
                    )
                )
            m += 1
    if issues:
        raise ScenarioValidationError(issues)
    return out


# --------------------------------------------------------------------------- canonical order
def _approach_lanes(
    spec: ScenarioSpec, roads: _Roads, int_index: dict[str, int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Incoming and outgoing lanes of every intersection, clockwise by bearing from north."""
    incoming: list[list[tuple[float, int, int]]] = [[] for _ in spec.network.intersections]
    outgoing: list[list[tuple[float, int, int]]] = [[] for _ in spec.network.intersections]
    for r, road in enumerate(spec.network.roads):
        ref = roads.reference[r]
        h_in, h_out = float(ref.headings[-1]), float(ref.headings[0])
        into = _bearing(-math.cos(h_in), -math.sin(h_in))  # where the approach comes from
        away = _bearing(math.cos(h_out), math.sin(h_out))
        for k in range(len(road.lanes)):
            incoming[int_index[road.to]].append((into, r, roads.lane_start[r] + k))
            outgoing[int_index[road.from_]].append((away, r, roads.lane_start[r] + k))
    return (
        [[lane for *_, lane in sorted(rows)] for rows in incoming],
        [[lane for *_, lane in sorted(rows)] for rows in outgoing],
    )


# --------------------------------------------------------------------------- diagnostics
def _demand_routes(spec: ScenarioSpec) -> Iterator[tuple[tuple[str, ...], list[Sequence[str]]]]:
    """(vehicle type ids, road sequences) of every demand source; OD routes by shortest path."""
    graph: RoadGraph | None = None

    def od(origin: str, destination: str, via: Sequence[str]) -> list[Sequence[str]]:
        nonlocal graph
        graph = graph or RoadGraph(spec)
        path = graph.path(origin, destination, via)
        return [path] if path else []

    demand = spec.demand
    for flow in demand.flows:
        types = tuple(flow.type_mix) if flow.type_mix else (flow.vehicle_type or "car",)
        if flow.route is not None:
            yield types, [flow.route]
        elif flow.routes is not None:
            yield types, [choice.roads for choice in flow.routes]
        elif flow.origin is not None and flow.destination is not None:
            yield types, od(flow.origin, flow.destination, flow.via)
    for trip in demand.trips:
        if trip.route is not None:
            yield (trip.vehicle_type,), [trip.route]
        elif trip.origin is not None and trip.destination is not None:
            yield (trip.vehicle_type,), od(trip.origin, trip.destination, trip.via)
    for line in demand.transit:
        yield (line.vehicle_type,), [line.route]


def _short_lanes(spec: ScenarioSpec, net: CompiledNetwork) -> list[ValidationIssue]:
    """E806: a lane shorter than ``length + min_gap`` of a vehicle type routed onto it.

    Checked for every lane of a source's first road and every connector target lane along
    its routes. ponytail: OD sources are checked on their shortest path only; a dynamic
    router may pick other roads (the engine's watchdog then applies).
    """
    types = {t.id: t for t in spec.vehicle_types}
    checked: set[tuple[int, str]] = set()
    for type_ids, routes in _demand_routes(spec):
        for route in routes:
            first = net.road_index[route[0]]
            start = int(net.road_lane_start[first])
            lanes = list(range(start, start + int(net.road_n_lanes[first])))
            for a, b in pairwise(route):
                pair = (net.road_index[a], net.road_index[b])
                lanes += [int(net.conn_to_lane[c - net.n_lanes]) for c in net.road_pair_conns[pair]]
            checked.update((lane, t) for lane in lanes for t in type_ids)
    issues = []
    for lane, t in sorted(checked):
        need = types[t].length + types[t].min_gap
        length = float(net.link_length[lane])
        if length < need:
            path = ("network", "roads", int(net.link_road[lane]), "lanes")
            where = (*path, int(net.link_lane_index[lane]))
            issues.append(issue("E806", where, id=net.link_ids[lane], L=length, t=t, need=need))
    return issues


def _intersection_warnings(
    spec: ScenarioSpec,
    declared: ScenarioSpec,
    net: CompiledNetwork,
    crossed_twice: Sequence[tuple[int, int]],
) -> list[ValidationIssue]:
    """W304 (connectors crossing twice) and W302 (crossing movements both protected).

    ``spec`` is the resolved spec. W302 points at the phase when the phases are written in
    the file (``declared``), otherwise at the signal (derived phases have no file path).
    """
    crossing: dict[int, set[tuple[int, int]]] = {}
    for a, b, kind in zip(
        net.conf_a.tolist(), net.conf_b.tolist(), net.conf_kind.tolist(), strict=True
    ):
        if kind == ConflictKind.crossing:
            ma, mb = sorted((int(net.link_movement[a]), int(net.link_movement[b])))
            if ma != mb:
                crossing.setdefault(int(net.link_intersection[a]), set()).add((ma, mb))
    twice: dict[int, list[tuple[int, int]]] = {}
    for a, b in crossed_twice:
        twice.setdefault(int(net.link_intersection[a]), []).append((a, b))
    issues: list[ValidationIssue] = []
    for j, ix in enumerate(spec.network.intersections):
        base = ("network", "intersections", j)
        for a, b in twice.get(j, ()):
            issues.append(issue("W304", base, a=net.link_ids[a], b=net.link_ids[b]))
        pairs = crossing.get(j)
        if not pairs or ix.signal is None:
            continue
        signal = declared.network.intersections[j].signal
        explicit = signal is not None and signal.phases is not None
        for p, phase in enumerate(ix.signal.phases or ()):
            protected = {net.mov_index[m] for m, s in phase.green.items() if s == "G"}
            where = (*base, "signal", "phases", p) if explicit else (*base, "signal")
            for ma, mb in sorted(pairs):
                if ma in protected and mb in protected:
                    issues.append(issue("W302", where, a=net.mov_ids[ma], b=net.mov_ids[mb]))
    return issues


# --------------------------------------------------------------------------- compile
def compile_network(scenario: Scenario) -> CompiledNetwork:
    """Compile ``scenario.resolved`` into immutable arrays (E.3), conflicts included.

    Raises ``ScenarioValidationError`` for E802, E905 and E806; warnings (W302, W304) are
    returned in ``net.report``. The result is deterministic: compiling the same scenario
    twice gives equal arrays.
    """
    spec = scenario.resolved
    network = spec.network
    int_index = {ix.id: j for j, ix in enumerate(network.intersections)}
    road_index = {road.id: r for r, road in enumerate(network.roads)}
    with log_duration(_log, "network compiled") as extra:
        try:
            roads = _build_roads(spec, {ix.id: ix for ix in network.intersections})
            conns = _build_connectors(scenario, roads, road_index)
        except ScenarioValidationError as exc:
            _raise_on_errors(exc.issues, scenario)
            raise  # pragma: no cover - only errors are raised by the builders
        table = compute_conflicts(
            [c.line for c in conns],
            _i32([c.intersection for c in conns]),
            _i32([c.from_lane for c in conns]),
            _i32([c.to_lane for c in conns]),
            roads.n_lanes,
        )
        net = _assemble(scenario, roads, conns, table, int_index)
        issues = _short_lanes(spec, net) + _intersection_warnings(
            spec, scenario.spec, net, table.crossed_twice
        )
        _raise_on_errors(issues, scenario)
        extra |= {"lanes": net.n_lanes, "connectors": net.n_conn, "conflicts": net.n_conflicts}
    return replace(net, report=CompileReport(tuple(issues)))


def _assemble(
    scenario: Scenario,
    roads: _Roads,
    conns: Sequence[_Connector],
    table: ConflictTable,
    int_index: dict[str, int],
) -> CompiledNetwork:
    """Lay out the E.3 arrays from the road, lane, connector and conflict geometry."""
    network = scenario.resolved.network
    road_index = {road.id: r for r, road in enumerate(network.roads)}
    n_lanes, n_conn = roads.n_lanes, len(conns)
    lane_rows = [(r, lane) for r, lanes in enumerate(roads.lanes) for lane in lanes]
    lane_road = [r for r, _ in lane_rows]
    lane_index = [lane.index for _, lane in lane_rows]
    n_road_lanes = [len(road.lanes) for road in network.roads]
    kinds = [resolved_value(ix.kind, f'kind of "{ix.id}"') for ix in network.intersections]
    road_to = [int_index[road.to] for road in network.roads]

    # ---- links: lanes first, then connectors
    lines = [lane.line for _, lane in lane_rows] + [c.line for c in conns]
    link_ids = (
        *(f"{network.roads[r].id}_{k}" for r, k in zip(lane_road, lane_index, strict=True)),
        *(c.id for c in conns),
    )
    link_length = np.array([line.length for line in lines], dtype=np.float64)
    movements = [(j, m) for j, ix in enumerate(network.intersections) for m in ix.movements or ()]
    lane_out: list[list[int]] = [[] for _ in range(n_lanes)]
    mov_conn: list[list[int]] = [[] for _ in movements]
    pair_mask: dict[tuple[int, int], int] = {}
    pair_conns: dict[tuple[int, int], list[int]] = {}
    for c, conn in enumerate(conns):
        link = n_lanes + c
        lane_out[conn.from_lane].append(link)
        mov_conn[conn.movement].append(link)
        pair = (lane_road[conn.from_lane], lane_road[conn.to_lane])
        pair_mask[pair] = pair_mask.get(pair, 0) | 1 << lane_index[conn.from_lane]
        pair_conns.setdefault(pair, []).append(link)

    # ---- movements and intersections
    mov_ids = tuple(resolved_value(m.id, "movement id") for _, m in movements)
    signalized = np.array([k is IntersectionKind.signalized for k in kinds], dtype=bool)
    int_program = np.where(signalized, np.cumsum(signalized) - 1, -1)
    int_in, int_out = _approach_lanes(scenario.resolved, roads, int_index)

    # ---- global polyline arrays, local coordinates
    centres = np.array([ix.point for ix in network.intersections], dtype=np.float64)
    world = np.concatenate([line.points for line in lines] or [np.zeros((0, 2))])
    everything = np.concatenate((world, centres))
    origin = everything.min(axis=0)
    link_base = np.concatenate(([0.0], np.cumsum(link_length + C.LINK_BASE_GAP)[:-1]))
    speed = np.array(
        [resolved_value(road.speed_limit, "road speed_limit") for road in network.roads],
        dtype=np.float64,
    )
    headway_s = C.IDM_HEADWAY + (C.VEHICLE_LENGTH + C.IDM_MIN_GAP) / speed
    lane_out_ptr, lane_out_conn = _csr(lane_out)
    mov_conn_ptr, mov_conn_flat = _csr(mov_conn)
    int_in_ptr, int_in_lanes = _csr(int_in)
    int_out_ptr, int_out_lanes = _csr(int_out)
    return CompiledNetwork(
        scenario_hash=scenario.content_hash,
        geometry_crc=zlib.crc32(scenario.content_hash.encode("ascii")),
        drive_side=network.drive_side,
        road_ids=tuple(road_index),
        road_index=MappingProxyType(road_index),
        link_ids=link_ids,
        link_index=MappingProxyType({lid: k for k, lid in enumerate(link_ids)}),
        mov_ids=mov_ids,
        mov_index=MappingProxyType({mid: k for k, mid in enumerate(mov_ids)}),
        int_ids=tuple(int_index),
        int_index=MappingProxyType(dict(int_index)),
        vehicle_types=scenario.resolved.vehicle_types,
        report=CompileReport(),
        road_from=_i32([int_index[road.from_] for road in network.roads]),
        road_to=_i32(road_to),
        road_lane_start=_i32(roads.lane_start[:-1]),
        road_n_lanes=np.array(n_road_lanes, dtype=np.uint8),
        road_length=np.array([ref.length for ref in roads.reference], dtype=np.float64),
        road_speed_limit=speed,
        road_capacity_vph=np.array(n_road_lanes) * C.SECONDS_PER_HOUR / headway_s,
        link_kind=np.array(
            [LinkKind.lane.code] * n_lanes + [LinkKind.connector.code] * n_conn, dtype=np.uint8
        ),
        link_length=link_length,
        link_speed_limit=np.array(
            [lane.speed_limit for _, lane in lane_rows] + [c.speed_limit for c in conns],
            dtype=np.float64,
        ),
        link_width=np.array(
            [lane.width for _, lane in lane_rows] + [c.width for c in conns], dtype=np.float32
        ),
        link_road=_i32(lane_road + [-1] * n_conn),
        link_lane_index=np.array(lane_index + [-1] * n_conn, dtype=np.int16),
        link_intersection=_i32([road_to[r] for r in lane_road] + [c.intersection for c in conns]),
        link_movement=_i32([-1] * n_lanes + [c.movement for c in conns]),
        link_is_exit=np.array(
            [kinds[road_to[r]] is IntersectionKind.boundary for r in lane_road] + [False] * n_conn,
            dtype=np.bool_,
        ),
        lane_left=_i32([link - 1 if k > 0 else -1 for link, k in enumerate(lane_index)]),
        lane_right=_i32(
            [
                link + 1 if k < n_road_lanes[r] - 1 else -1
                for link, (r, k) in enumerate(zip(lane_road, lane_index, strict=True))
            ]
        ),
        lane_out_ptr=lane_out_ptr,
        lane_out_conn=lane_out_conn,
        conn_from_lane=_i32([c.from_lane for c in conns]),
        conn_to_lane=_i32([c.to_lane for c in conns]),
        road_pair_mask=MappingProxyType(pair_mask),
        road_pair_conns=MappingProxyType({k: tuple(v) for k, v in pair_conns.items()}),
        lane_detector_start=np.maximum(link_length[:n_lanes] - C.DETECTOR_LENGTH, 0.0),
        mov_intersection=_i32([j for j, _ in movements]),
        mov_from_road=_i32([road_index[m.from_road] for _, m in movements]),
        mov_to_road=_i32([road_index[m.to_road] for _, m in movements]),
        mov_turn=np.array(
            [resolved_value(m.turn, f'turn of "{m.id}"').code for _, m in movements],
            dtype=np.uint8,
        ),
        mov_static_rank=np.array(
            [static_rank(kinds[j], m, network.drive_side) for j, m in movements], dtype=np.uint8
        ),
        mov_conn_ptr=mov_conn_ptr,
        mov_conn=mov_conn_flat,
        int_kind=np.array([k.code for k in kinds], dtype=np.uint8),
        int_point=(centres - origin).reshape(-1, 2),
        int_radius=np.array(
            [resolved_value(ix.radius, "intersection radius") for ix in network.intersections],
            dtype=np.float64,
        ),
        int_program=_i32(int_program),
        int_in_ptr=int_in_ptr,
        int_in_lanes=int_in_lanes,
        int_out_ptr=int_out_ptr,
        int_out_lanes=int_out_lanes,
        conf_a=table.conf_a,
        conf_b=table.conf_b,
        conf_kind=table.conf_kind,
        conf_zone_a=table.conf_zone_a,
        conf_zone_b=table.conf_zone_b,
        conn_conf_ptr=table.conn_conf_ptr,
        conn_conf=table.conn_conf,
        conn_conf_side=table.conn_conf_side,
        pts=world - origin,
        link_pt_start=np.concatenate(([0], np.cumsum([len(ln.points) for ln in lines]))).astype(
            np.int32
        ),
        s_global=np.concatenate(
            [b + ln.cumulative for b, ln in zip(link_base, lines, strict=True)] or [np.zeros(0)]
        ),
        link_base=link_base,
        seg_heading=np.concatenate(
            [np.append(ln.headings, ln.headings[-1]) for ln in lines] or [np.zeros(0)]
        ),
        origin=origin,
        bbox=np.concatenate(([0.0, 0.0], everything.max(axis=0) - origin)),
    )
