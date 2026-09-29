"""Vectorised polyline crossings, lateral separation and bounding boxes (plan E.5 rule 5)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from urbanflow.core.constants import GEOM_EPS, POSITION_EPS, SEPARATION_SAMPLE_STEP
from urbanflow.core.types import BoolArray, FloatArray
from urbanflow.geometry.polyline import Polyline, as_points

__all__ = [
    "bbox_overlap",
    "bounding_box",
    "first_crossing",
    "point_polyline_distance",
    "segment_intersections",
    "separation_distance",
]


def _cross(u: FloatArray, v: FloatArray) -> FloatArray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def segment_intersections(a_pts: ArrayLike, b_pts: ArrayLike) -> FloatArray:
    """Every crossing of polylines ``a`` and ``b`` as rows ``(s_a, s_b, angle)``, sorted by s_a.

    ``s_a``/``s_b`` are arc lengths on each polyline and ``angle`` ∈ [0, π] is the angle
    between their tangent directions there. Segments are closed, so touching endpoints
    count; parallel and collinear segments never cross. A crossing at a shared vertex is
    reported once.
    """
    # ponytail: dense (na x nb) segment grid; fine for connectors (<= 17 points each),
    # callers prefilter pairs with bbox_overlap. Sweep-line if long polylines ever need it.
    a, b = Polyline(a_pts), Polyline(b_pts)
    p, r = a.points[:-1], np.diff(a.points, axis=0)
    q, s = b.points[:-1], np.diff(b.points, axis=0)
    la, lb = np.diff(a.cumulative), np.diff(b.cumulative)
    rxs = _cross(r[:, None], s[None, :])
    qp = q[None, :] - p[:, None]
    crossing = np.abs(rxs) > GEOM_EPS * la[:, None] * lb[None, :]
    den = np.where(crossing, rxs, 1.0)
    t = _cross(qp, s[None, :]) / den
    u = _cross(qp, r[:, None]) / den
    lo, hi = -GEOM_EPS, 1.0 + GEOM_EPS
    i, j = np.nonzero(crossing & (t >= lo) & (t <= hi) & (u >= lo) & (u <= hi))
    s_a = a.cumulative[i] + np.clip(t[i, j], 0.0, 1.0) * la[i]
    s_b = b.cumulative[j] + np.clip(u[i, j], 0.0, 1.0) * lb[j]
    angle = np.arctan2(np.abs(rxs[i, j]), np.einsum("ij,ij->i", r[i], s[j]))
    rows = np.column_stack((s_a, s_b, angle))[np.lexsort((s_b, s_a))]
    new = (np.diff(rows[:, 0]) > POSITION_EPS) | (np.abs(np.diff(rows[:, 1])) > POSITION_EPS)
    return rows[np.concatenate(([True], new))] if len(rows) else rows


def first_crossing(a_pts: ArrayLike, b_pts: ArrayLike) -> tuple[float, float, float] | None:
    """The first crossing along ``a`` as ``(s_a, s_b, angle)``, or None if they don't cross."""
    rows = segment_intersections(a_pts, b_pts)
    if not len(rows):
        return None
    s_a, s_b, angle = (float(v) for v in rows[0])
    return s_a, s_b, angle


def point_polyline_distance(points: ArrayLike, line_pts: ArrayLike) -> FloatArray:
    """Euclidean distance from each of ``points`` (``(n, 2)``) to the polyline ``line_pts``."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    line = Polyline(line_pts)
    origin, seg = line.points[:-1], np.diff(line.points, axis=0)
    rel = pts[:, None, :] - origin[None, :, :]
    norm2 = np.einsum("mk,mk->m", seg, seg)  # 0 only if a tiny segment underflows
    proj = np.einsum("nmk,mk->nm", rel, seg)
    t = np.clip(np.divide(proj, norm2, out=np.zeros_like(proj), where=norm2 > 0), 0.0, 1.0)
    gap = rel - t[..., None] * seg[None, :, :]
    dist: FloatArray = np.sqrt(np.einsum("nmk,nmk->nm", gap, gap).min(axis=1))
    return dist


def separation_distance(
    a_pts: ArrayLike,
    b_pts: ArrayLike,
    width: float,
    from_end: bool = False,
    *,
    step: float = SEPARATION_SAMPLE_STEP,
) -> float:
    """First arc length along ``a`` where its distance to ``b`` exceeds ``width`` (d_sep).

    For diverging connectors (common start). With ``from_end`` the arc length is measured
    backwards from ``a``'s end, for merging connectors (common end); the zone is then
    ``[L - d_sep, L]``. ``a`` is sampled every ``step`` metres and the threshold crossing is
    interpolated linearly between samples. Capped at ``a``'s length.
    """
    pts = as_points(a_pts)
    line = Polyline(pts[::-1] if from_end else pts)
    s = np.linspace(0.0, line.length, max(2, math.ceil(line.length / step) + 1))
    dist = point_polyline_distance(line.point_at(s), b_pts)
    over = np.flatnonzero(dist > width)
    if not len(over):
        return line.length
    k = int(over[0])
    if k == 0:
        return 0.0
    frac = (width - dist[k - 1]) / (dist[k] - dist[k - 1])
    return float(s[k - 1] + frac * (s[k] - s[k - 1]))


def bounding_box(points: ArrayLike) -> FloatArray:
    """``[xmin, ymin, xmax, ymax]`` of a point set."""
    pts = as_points(points)
    return np.concatenate((pts.min(axis=0), pts.max(axis=0)))


def bbox_overlap(a: ArrayLike, b: ArrayLike) -> BoolArray:
    """Pairwise overlap of boxes ``a`` (``(n, 4)``) and ``b`` (``(m, 4)``) -> ``bool[n, m]``.

    Boxes are closed: touching boxes overlap. Single ``(4,)`` boxes are accepted.
    """
    ba = np.atleast_2d(np.asarray(a, dtype=np.float64))[:, None, :]
    bb = np.atleast_2d(np.asarray(b, dtype=np.float64))[None, :, :]
    return (
        (ba[..., 0] <= bb[..., 2])
        & (bb[..., 0] <= ba[..., 2])
        & (ba[..., 1] <= bb[..., 3])
        & (bb[..., 1] <= ba[..., 3])
    )
