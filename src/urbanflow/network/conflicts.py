"""Conflict zones between connectors of one intersection (plan E.5 rule 5).

For every pair of connectors (a < b, by link id) at an intersection whose bounding boxes
overlap:

* **diverging** (same ``from_lane``): zone ``[0, d_sep]`` on each, where ``d_sep`` is the
  first arc length at which the other polyline is farther than ``W_max`` (``CONFLICT_WIDTH``).
  Compiled for geometry and reporting only; admission ignores them (F.3).
* **merging** (same ``to_lane``): zone ``[L - d_sep, L]`` on each, ``d_sep`` from the end.
* **crossing**: at the first crossing along ``a`` (arc positions ``s_a``, ``s_b``, angle θ),
  zones ``[s - h, s + h]`` with ``h = (W/2)/sin θ' + (W/2)/tan θ'`` and ``θ' = min(θ, π - θ)``
  the acute angle between the centre lines. Below 15° the closed form blows up, so the zone
  is the arc interval around the crossing where the other polyline is closer than ``W_max``
  (the ``d_sep`` sampler). Pairs that cross more than once get one merged zone per
  connector and are reported (W304).

Zones are clipped to each connector. ``conn_conf`` lists the conflicts of every connector
sorted by ``z_in`` on that connector; ``conn_conf_side`` is 0 where the connector is ``a``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from urbanflow.core.constants import CONFLICT_WIDTH, LOW_ANGLE_CROSSING, POSITION_EPS
from urbanflow.core.types import FloatArray, IntArray, UIntArray
from urbanflow.geometry import (
    Polyline,
    bbox_overlap,
    bounding_box,
    segment_intersections,
    separation_distance,
)

__all__ = [
    "ConflictKind",
    "ConflictTable",
    "compute_conflicts",
    "crossing_half_length",
    "crossing_zones",
]

Zone = tuple[float, float]


class ConflictKind(IntEnum):
    """``conf_kind`` codes."""

    crossing = 0
    merging = 1
    diverging = 2


@dataclass(frozen=True, slots=True, eq=False)
class ConflictTable:
    """Conflict arrays (E.3). ``conf_a``/``conf_b`` are connector *link* ids with a < b;
    ``conn_conf_ptr`` is indexed by connector index (link id minus the lane count)."""

    conf_a: IntArray
    conf_b: IntArray
    conf_kind: UIntArray
    conf_zone_a: FloatArray
    """``[z_in, z_out]`` along connector a, m (shape ``(C, 2)``)."""
    conf_zone_b: FloatArray
    conn_conf_ptr: IntArray
    conn_conf: IntArray
    conn_conf_side: UIntArray
    crossed_twice: tuple[tuple[int, int], ...]
    """Connector link-id pairs that cross more than once (W304); their zones were merged."""

    def __len__(self) -> int:
        return len(self.conf_a)


def crossing_half_length(theta: float, width: float = CONFLICT_WIDTH) -> float:
    """Half-length ``(W/2)/sin θ' + (W/2)/tan θ'`` of a crossing zone, ``θ' = min(θ, π - θ)``.

    Using the acute angle keeps obtuse crossings from shrinking the zone: a permissive left
    turn crossing the opposing straight at θ ≈ 133° gets 2h ≈ 6.9 m, not 1.3 m.
    """
    acute = min(theta, math.pi - theta)
    half = width / 2
    return half / math.sin(acute) + half / math.tan(acute)


def _near_interval(line: Polyline, other: FloatArray, s: float, width: float) -> Zone:
    """Arc interval around ``s`` on ``line`` where ``other`` stays within ``width``."""
    lo, hi = s, s
    if s > POSITION_EPS:
        lo -= separation_distance(line.cut(0.0, s).points, other, width, from_end=True)
    if line.length - s > POSITION_EPS:
        hi += separation_distance(line.cut(s, line.length).points, other, width)
    return lo, hi


def crossing_zones(
    a: Polyline, b: Polyline, crossings: FloatArray, width: float = CONFLICT_WIDTH
) -> tuple[Zone, Zone]:
    """Zones on ``a`` and ``b`` covering every crossing row ``(s_a, s_b, θ)``, clipped."""
    lo_a = lo_b = math.inf
    hi_a = hi_b = -math.inf
    for s_a, s_b, theta in crossings.tolist():
        if min(theta, math.pi - theta) < LOW_ANGLE_CROSSING:
            za = _near_interval(a, b.points, s_a, width)
            zb = _near_interval(b, a.points, s_b, width)
        else:
            h = crossing_half_length(theta, width)
            if not h > 0:  # the plan's compile-time assertion, kept under python -O
                raise RuntimeError(f"crossing half-length must be positive, got {h}")
            za, zb = (s_a - h, s_a + h), (s_b - h, s_b + h)
        lo_a, hi_a = min(lo_a, za[0]), max(hi_a, za[1])
        lo_b, hi_b = min(lo_b, zb[0]), max(hi_b, zb[1])
    return _clip((lo_a, hi_a), a.length), _clip((lo_b, hi_b), b.length)


def _clip(zone: Zone, length: float) -> Zone:
    return max(0.0, zone[0]), min(length, zone[1])


def _pair(a: Polyline, b: Polyline, same_from: bool, same_to: bool) -> tuple[int, Zone, Zone, int]:
    """(kind, zone on a, zone on b, number of crossings) of one overlapping pair."""
    # d_sep is summed over the (possibly reversed) polyline, so L - d_sep can come out as
    # -1e-15: every zone is clipped to its connector.
    if same_from:
        d_a = separation_distance(a.points, b.points, CONFLICT_WIDTH)
        d_b = separation_distance(b.points, a.points, CONFLICT_WIDTH)
        za, zb = _clip((0.0, d_a), a.length), _clip((0.0, d_b), b.length)
        return ConflictKind.diverging, za, zb, 0
    if same_to:
        d_a = separation_distance(a.points, b.points, CONFLICT_WIDTH, from_end=True)
        d_b = separation_distance(b.points, a.points, CONFLICT_WIDTH, from_end=True)
        za = _clip((a.length - d_a, a.length), a.length)
        zb = _clip((b.length - d_b, b.length), b.length)
        return ConflictKind.merging, za, zb, 0
    rows = segment_intersections(a.points, b.points)
    if not len(rows):
        return -1, (0.0, 0.0), (0.0, 0.0), 0
    za, zb = crossing_zones(a, b, rows)
    return ConflictKind.crossing, za, zb, len(rows)


def compute_conflicts(
    lines: Sequence[Polyline],
    intersection: IntArray,
    from_lane: IntArray,
    to_lane: IntArray,
    first_link: int,
) -> ConflictTable:
    """Conflict table of connectors ``0 .. n-1`` (link ids ``first_link + c``).

    ``intersection``, ``from_lane`` and ``to_lane`` give each connector's intersection and
    lane links. Pairs are prefiltered by bounding-box overlap per intersection (vectorised);
    output order is intersection, then ``a``, then ``b``, so it is deterministic.
    """
    n = len(lines)
    boxes = np.array([bounding_box(line.points) for line in lines]).reshape(n, 4)
    rows: list[tuple[int, int, int, float, float, float, float]] = []
    twice: list[tuple[int, int]] = []
    order = np.argsort(intersection, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(intersection[order])) + 1)
    for idx in groups:
        if len(idx) < 2:
            continue
        ii, jj = np.nonzero(np.triu(bbox_overlap(boxes[idx], boxes[idx]), 1))
        for p, q in zip(idx[ii].tolist(), idx[jj].tolist(), strict=True):
            same_from, same_to = bool(from_lane[p] == from_lane[q]), bool(to_lane[p] == to_lane[q])
            kind, za, zb, crossings = _pair(lines[p], lines[q], same_from, same_to)
            if kind < 0:
                continue
            rows.append((first_link + p, first_link + q, kind, *za, *zb))
            if crossings > 1:
                twice.append((first_link + p, first_link + q))
    table = np.array(rows, dtype=np.float64).reshape(len(rows), 7)
    conf_a, conf_b = table[:, 0].astype(np.int32), table[:, 1].astype(np.int32)
    zone_a, zone_b = table[:, 3:5], table[:, 5:7]
    # CSR over connectors, each list sorted by z_in on that connector (ties: conflict index)
    conn = np.concatenate((conf_a, conf_b)) - first_link
    z_in = np.concatenate((zone_a[:, 0], zone_b[:, 0]))
    conf = np.tile(np.arange(len(rows), dtype=np.int32), 2)
    side = np.repeat(np.array([0, 1], dtype=np.uint8), len(rows))
    sort = np.lexsort((conf, z_in, conn))
    ptr = np.concatenate(([0], np.cumsum(np.bincount(conn, minlength=n)))).astype(np.int32)
    return ConflictTable(
        conf_a=conf_a,
        conf_b=conf_b,
        conf_kind=table[:, 2].astype(np.uint8),
        conf_zone_a=np.ascontiguousarray(zone_a),
        conf_zone_b=np.ascontiguousarray(zone_b),
        conn_conf_ptr=ptr,
        conn_conf=conf[sort],
        conn_conf_side=side[sort],
        crossed_twice=tuple(twice),
    )
