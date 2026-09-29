"""Road trimming and lane polylines (plan E.5 rules 2-3).

A road's reference polyline is its median edge. Each end touching a non-boundary
intersection is cut back along arc length to the intersection's radius; lane ``k`` is then
the miter offset of the trimmed reference by ``o_k = sum(w_i, i < k) + w_k / 2`` toward the
curb (right of travel for ``drive_side="right"``, left otherwise).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from urbanflow.core.constants import MITER_LIMIT
from urbanflow.core.types import BoolArray, DriveSide, FloatArray, IntersectionKind
from urbanflow.geometry import Polyline, offset_polyline
from urbanflow.scenario.schema import IntersectionSpec, RoadSpec

__all__ = [
    "DegenerateGeometryError",
    "LaneGeom",
    "build_lanes",
    "lane_offsets",
    "resolved_value",
    "trim_distance",
    "trim_road",
]


class DegenerateGeometryError(ValueError):
    """A lane or connector polyline that cannot be driven (reported as E905)."""


@dataclass(frozen=True, slots=True)
class LaneGeom:
    """Geometry and limits of one lane of a road."""

    index: int
    """Lane index within the road (0 = median lane)."""
    offset: float
    """Distance of the lane centreline from the median edge, m."""
    width: float
    """Lane width, m."""
    speed_limit: float
    """Speed limit, m/s."""
    line: Polyline
    """Centreline, in travel direction."""


def resolved_value[T](value: T | None, what: str) -> T:
    """``value`` of a derived spec field; ``None`` means the spec was not resolved."""
    if value is None:
        raise TypeError(f"{what} is unset: compile_network needs Scenario.resolved")
    return value


def trim_distance(end: Sequence[float] | FloatArray, ix: IntersectionSpec) -> float:
    """Arc length cut from a road end at ``ix``: ``max(0, r - |p_end - c|)``; 0 at boundaries."""
    if ix.kind is IntersectionKind.boundary:
        return 0.0
    radius = resolved_value(ix.radius, f'radius of intersection "{ix.id}"')
    return max(0.0, radius - math.dist((float(end[0]), float(end[1])), ix.point))


def trim_road(reference: Polyline, start: float, end: float) -> Polyline:
    """The reference with ``start`` metres cut from its start and ``end`` from its end.

    The cut runs along arc length (not along the first segment), so it never overshoots
    on curved roads.
    """
    return reference.cut(start, reference.length - end)


def lane_offsets(widths: Sequence[float]) -> list[float]:
    """Centreline offset of every lane from the median edge: ``sum(w_i, i < k) + w_k / 2``."""
    edges = np.concatenate(([0.0], np.cumsum(widths)))
    return [float(v) for v in edges[:-1] + np.asarray(widths) / 2]


def _bevels(reference: FloatArray, miter_limit: float) -> BoolArray:
    """Vertices ``offset_polyline`` bevels (miter factor above the limit), endpoints False."""
    vec = np.diff(reference, axis=0)
    tangent = vec / np.hypot(vec[:, 0], vec[:, 1])[:, None]
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    denom = 1.0 + np.einsum("ij,ij->i", normal[:-1], normal[1:])
    return np.concatenate(([False], denom < 2 / miter_limit**2, [False]))


def _reversed_segment(reference: FloatArray, lane: FloatArray) -> int | None:
    """Index of the first reference segment whose offset runs backwards (or has length 0).

    Offset segment ``i`` runs from the last point emitted for vertex ``i`` to the first
    point emitted for vertex ``i + 1`` (a bevelled vertex emits two points).
    """
    bevel = _bevels(reference, MITER_LIMIT)
    counts = 1 + bevel.astype(np.intp)
    first = np.cumsum(counts) - counts
    seg = lane[first[1:]] - lane[first[:-1] + bevel[:-1]]
    dots = np.einsum("ij,ij->i", seg, np.diff(reference, axis=0))
    bad = np.flatnonzero(dots <= 0)
    return int(bad[0]) if len(bad) else None


def build_lanes(road: RoadSpec, reference: Polyline, drive_side: DriveSide) -> list[LaneGeom]:
    """Lane centrelines of a resolved ``road`` from its *trimmed* reference polyline.

    Raises :class:`DegenerateGeometryError` if a lane has zero length or reverses
    direction, which happens when a lane's offset exceeds the radius of a bend.
    """
    width = resolved_value(road.lane_width, f'lane_width of road "{road.id}"')
    speed = resolved_value(road.speed_limit, f'speed_limit of road "{road.id}"')
    widths = [lane.width if lane.width is not None else width for lane in road.lanes]
    lanes: list[LaneGeom] = []
    for k, (lane, offset) in enumerate(zip(road.lanes, lane_offsets(widths), strict=True)):
        pts = offset_polyline(reference.points, offset, side=drive_side)
        bad = _reversed_segment(reference.points, pts)
        if bad is not None:
            raise DegenerateGeometryError(
                f"lane {k} reverses direction on segment {bad} of the trimmed road "
                f"(its {offset:.1f} m offset exceeds the bend radius)"
            )
        lanes.append(
            LaneGeom(
                index=k,
                offset=offset,
                width=widths[k],
                speed_limit=lane.speed_limit if lane.speed_limit is not None else speed,
                line=Polyline(pts),
            )
        )
    return lanes
