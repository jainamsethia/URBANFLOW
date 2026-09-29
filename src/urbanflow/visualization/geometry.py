"""Render geometry: everything static a client needs to draw a compiled network (E.5 #7).

:func:`render_geometry` turns a :class:`CompiledNetwork` into a :class:`RenderGeometry`,
the JSON document the REST API serves and replays store as ``geometry.json``. Coordinates
are *local* (world = local + ``origin``), like frame ``xy``, and rounded to 1 mm. Large
collections are columnar (one list per field, equal lengths) so clients can build binary
deck.gl attributes directly:

* ``links``: every link in link-index order (lanes, then connectors) with kind, owner (road
  id of a lane, movement id of a connector), lane width and path;
* ``movements`` in movement-index order, with their connector links;
* ``signal_heads``: one per movement at the stop line of its median-most lane; heads that
  share a lane are spread across it (frames colour them by ``signals[movement]``);
* ``roads.surfaces``: the union of a road's lane quads (median edge + reversed curb edge);
* ``intersections.polygons``: convex hull of the lane-end corners and a 16-point radius disc
  (empty for boundary nodes);
* ``lane_markings`` (dashed between same-direction lanes, solid median and curb edges),
  ``stop_lines``, ``vehicle_types`` (frames' ``types[]`` index this list), ``origin``,
  ``bbox`` and ``geometry_crc``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Final, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbanflow.core import constants as C
from urbanflow.core.types import FloatArray, IntersectionKind, TurnKind, VehicleClass
from urbanflow.geometry import offset_polyline
from urbanflow.network.compiled import CompiledNetwork

__all__ = [
    "GEOMETRY_VERSION",
    "LaneMarkings",
    "RenderGeometry",
    "RenderIntersections",
    "RenderLinks",
    "RenderMovements",
    "RenderRoads",
    "RenderVehicleType",
    "SignalHeads",
    "StopLines",
    "convex_hull",
    "render_geometry",
]

GEOMETRY_VERSION: Final = 1
"""Format version of :class:`RenderGeometry`; bumped on any incompatible change."""

Point = tuple[float, float]
Path = list[Point]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)


class _Columns(_Model):
    """A columnar table: every field is a list and all lists have the same length."""

    @model_validator(mode="after")
    def _equal_lengths(self) -> Self:
        lengths = {name: len(getattr(self, name)) for name in type(self).model_fields}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"columns must have equal lengths, got {lengths}")
        return self


class RenderLinks(_Columns):
    """Links in link-index order: lanes first, then connectors."""

    id: list[str]
    """Lane ``"{road}_{k}"`` or connector ``"{lane_in}->{lane_out}"``."""
    kind: list[Literal["lane", "connector"]]
    owner: list[str]
    """Road id of a lane, movement id of a connector."""
    lane_width: list[float]
    """Drawing width, m."""
    path: list[Path]
    """Centreline, local coordinates."""


class RenderMovements(_Columns):
    """Movements in movement-index order (frame ``signals`` are indexed the same way)."""

    id: list[str]
    intersection: list[str]
    from_road: list[str]
    to_road: list[str]
    turn: list[TurnKind]
    links: list[list[int]]
    """Connector link indices of each movement."""


class SignalHeads(_Columns):
    """One signal head per movement, at the stop line."""

    movement: list[int]
    position: list[Point]


class RenderRoads(_Columns):
    """Roads in road-index order."""

    id: list[str]
    surfaces: list[Path]
    """Road surface polygon (median edge, then the curb edge reversed)."""


class RenderIntersections(_Columns):
    """Intersections in intersection-index order."""

    id: list[str]
    kind: list[IntersectionKind]
    point: list[Point]
    radius: list[float]
    polygons: list[Path]
    """Convex outline; empty for boundary nodes."""


class LaneMarkings(_Columns):
    """Painted lines: dashed separators between same-direction lanes, solid road edges."""

    road: list[int]
    dashed: list[bool]
    path: list[Path]


class StopLines(_Columns):
    """A stop line across the end of every lane that enters a non-boundary intersection."""

    link: list[int]
    path: list[Path]


class RenderVehicleType(_Model):
    """A vehicle type as drawn by clients (``vehicles_added.types[]`` index this list)."""

    id: str
    vclass: VehicleClass
    length: float
    width: float
    color: str | None


class RenderGeometry(_Model):
    """Static render geometry of a compiled network (local coordinates)."""

    version: Literal[1] = GEOMETRY_VERSION
    geometry_crc: int = Field(ge=0, lt=2**32)
    """Must equal the ``geometry_crc`` of frames drawn on this geometry."""
    origin: Point
    """World coordinates of the local origin."""
    bbox: tuple[float, float, float, float]
    """``[xmin, ymin, xmax, ymax]``, local coordinates."""
    links: RenderLinks
    movements: RenderMovements
    signal_heads: SignalHeads
    roads: RenderRoads
    intersections: RenderIntersections
    lane_markings: LaneMarkings
    stop_lines: StopLines
    vehicle_types: list[RenderVehicleType]


# --------------------------------------------------------------------------- helpers
def _path(points: FloatArray) -> Path:
    rounded = np.round(points, C.RENDER_COORD_DECIMALS).tolist()
    return [(x, y) for x, y in rounded]


def _right(heading: float) -> FloatArray:
    """Unit normal to the right of a heading."""
    return np.array((math.sin(heading), -math.cos(heading)))


def convex_hull(points: FloatArray) -> FloatArray:
    """Counter-clockwise convex hull (Andrew's monotone chain), no repeated first point."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return np.array(pts, dtype=np.float64).reshape(-1, 2)

    def half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]), dtype=np.float64)


def _lane_corners(net: CompiledNetwork, lane: int, at_end: bool) -> FloatArray:
    """The two edge points across a lane's start or end."""
    line = net.link_points(lane)
    k = int(net.link_pt_start[lane + 1]) - 1 if at_end else int(net.link_pt_start[lane])
    normal = _right(float(net.seg_heading[k])) * float(net.link_width[lane]) / 2
    point = line[-1] if at_end else line[0]
    return np.stack((point - normal, point + normal))


# --------------------------------------------------------------------------- sections
def _links(net: CompiledNetwork) -> RenderLinks:
    n_lanes = net.n_lanes
    owners = [net.road_ids[r] for r in net.link_road[:n_lanes].tolist()]
    owners += [net.mov_ids[m] for m in net.link_movement[n_lanes:].tolist()]
    return RenderLinks(
        id=list(net.link_ids),
        kind=["lane"] * n_lanes + ["connector"] * net.n_conn,
        owner=owners,
        lane_width=np.round(net.link_width.astype(np.float64), C.RENDER_COORD_DECIMALS).tolist(),
        path=[_path(net.link_points(link)) for link in range(net.n_links)],
    )


def _movements(net: CompiledNetwork) -> RenderMovements:
    ptr = net.mov_conn_ptr.tolist()
    return RenderMovements(
        id=list(net.mov_ids),
        intersection=[net.int_ids[j] for j in net.mov_intersection.tolist()],
        from_road=[net.road_ids[r] for r in net.mov_from_road.tolist()],
        to_road=[net.road_ids[r] for r in net.mov_to_road.tolist()],
        turn=[TurnKind.from_code(t) for t in net.mov_turn.tolist()],
        links=[net.mov_conn[ptr[m] : ptr[m + 1]].tolist() for m in range(net.n_movements)],
    )


def _signal_heads(net: CompiledNetwork) -> SignalHeads:
    ptr = net.mov_conn_ptr.tolist()
    head_lane: list[int] = []
    for m in range(net.n_movements):
        lanes = net.conn_from_lane[net.mov_conn[ptr[m] : ptr[m + 1]] - net.n_lanes]
        head_lane.append(int(lanes[np.argmin(net.link_lane_index[lanes])]))
    sharing: dict[int, list[int]] = defaultdict(list)
    for m, lane in enumerate(head_lane):
        sharing[lane].append(m)
    positions: list[Point] = []
    for m, lane in enumerate(head_lane):
        group = sharing[lane]
        end = net.link_points(lane)[-1]
        heading = float(net.seg_heading[int(net.link_pt_start[lane + 1]) - 1])
        shift = (group.index(m) - (len(group) - 1) / 2) * float(net.link_width[lane]) / len(group)
        positions.append(_path((end + shift * _right(heading))[None, :])[0])
    return SignalHeads(movement=list(range(net.n_movements)), position=positions)


def _road_edges(net: CompiledNetwork, road: int) -> list[tuple[FloatArray, bool]]:
    """(polyline, dashed) of the median edge, the lane separators and the curb edge."""
    first = int(net.road_lane_start[road])
    lanes = range(first, first + int(net.road_n_lanes[road]))
    half = [float(net.link_width[lane]) / 2 for lane in lanes]
    side = net.drive_side
    edges = [(offset_polyline(net.link_points(first), -half[0], side=side), False)]
    edges += [
        (offset_polyline(net.link_points(lane), half[k], side=side), True)
        for k, lane in enumerate(lanes)
    ]
    edges[-1] = (edges[-1][0], False)  # the curb edge is solid
    return edges


def _roads_and_markings(net: CompiledNetwork) -> tuple[RenderRoads, LaneMarkings]:
    surfaces: list[Path] = []
    marks: dict[str, list[Any]] = {"road": [], "dashed": [], "path": []}
    for road in range(net.n_roads):
        edges = _road_edges(net, road)
        surfaces.append(_path(np.concatenate((edges[0][0], edges[-1][0][::-1]))))
        for line, dashed in edges:
            marks["road"].append(road)
            marks["dashed"].append(dashed)
            marks["path"].append(_path(line))
    roads = RenderRoads(id=list(net.road_ids), surfaces=surfaces)
    return roads, LaneMarkings(**marks)


def _intersections(net: CompiledNetwork) -> RenderIntersections:
    boundary = IntersectionKind.boundary.code
    angles = np.linspace(0.0, math.tau, C.INTERSECTION_DISC_POINTS, endpoint=False)
    disc = np.column_stack((np.cos(angles), np.sin(angles)))
    polygons: list[Path] = []
    for j in range(net.n_intersections):
        if int(net.int_kind[j]) == boundary:
            polygons.append([])
            continue
        ins = net.int_in_lanes[net.int_in_ptr[j] : net.int_in_ptr[j + 1]].tolist()
        outs = net.int_out_lanes[net.int_out_ptr[j] : net.int_out_ptr[j + 1]].tolist()
        corners = [_lane_corners(net, lane, at_end=True) for lane in ins]
        corners += [_lane_corners(net, lane, at_end=False) for lane in outs]
        ring = net.int_point[j] + float(net.int_radius[j]) * disc
        polygons.append(_path(convex_hull(np.concatenate([ring, *corners]))))
    return RenderIntersections(
        id=list(net.int_ids),
        kind=[IntersectionKind.from_code(k) for k in net.int_kind.tolist()],
        point=_path(net.int_point),
        radius=net.int_radius.tolist(),
        polygons=polygons,
    )


def _stop_lines(net: CompiledNetwork) -> StopLines:
    boundary = IntersectionKind.boundary.code
    lanes = [
        lane
        for lane in range(net.n_lanes)
        if int(net.int_kind[net.link_intersection[lane]]) != boundary
    ]
    return StopLines(
        link=lanes, path=[_path(_lane_corners(net, lane, at_end=True)) for lane in lanes]
    )


def render_geometry(net: CompiledNetwork) -> RenderGeometry:
    """The :class:`RenderGeometry` of ``net`` (see the module docstring for the contents)."""
    roads, markings = _roads_and_markings(net)
    return RenderGeometry(
        geometry_crc=net.geometry_crc,
        origin=_path(net.origin[None, :])[0],
        bbox=(*np.round(net.bbox, C.RENDER_COORD_DECIMALS).tolist(),),
        links=_links(net),
        movements=_movements(net),
        signal_heads=_signal_heads(net),
        roads=roads,
        intersections=_intersections(net),
        lane_markings=markings,
        stop_lines=_stop_lines(net),
        vehicle_types=[
            RenderVehicleType(
                id=t.id, vclass=t.vclass, length=t.length, width=t.width, color=t.color
            )
            for t in net.vehicle_types
        ],
    )
