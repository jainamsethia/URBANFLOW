"""Arc-length parameterised polylines, miter offsets and discrete curvature (plan E.5).

Headings are radians counter-clockwise from +x. "Right" and "left" are relative to the
direction of travel (point order).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from urbanflow.core.constants import MITER_LIMIT, POSITION_EPS
from urbanflow.core.types import DriveSide, FloatArray, IntArray

__all__ = ["Polyline", "as_points", "curvature_radii", "offset_polyline"]


def as_points(points: ArrayLike) -> FloatArray:
    """``points`` as a float64 ``(n, 2)`` array with ``n >= 2`` finite rows (a new array)."""
    pts = np.array(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError(f"a polyline needs an (n >= 2, 2) array of points, got shape {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("polyline points must be finite")
    return pts


def _segments(pts: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Segment vectors and lengths; rejects repeated consecutive points."""
    vec = np.diff(pts, axis=0)
    length = np.hypot(vec[:, 0], vec[:, 1])
    if (length <= 0).any():
        k = int(np.flatnonzero(length <= 0)[0]) + 1
        raise ValueError(f"point {k} repeats the previous point; consecutive points must differ")
    return vec, length


class Polyline:
    """An immutable 2-D polyline with cumulative arc lengths.

    ``points`` is ``(n, 2)``, ``cumulative[i]`` the arc length at point ``i`` and
    ``headings[i]`` the heading of segment ``i`` (``n - 1`` entries). All are read-only.
    """

    __slots__ = ("cumulative", "headings", "points")

    def __init__(self, points: ArrayLike) -> None:
        pts = as_points(points)
        vec, length = _segments(pts)
        cumulative = np.concatenate(([0.0], np.cumsum(length)))
        headings = np.arctan2(vec[:, 1], vec[:, 0])
        for arr in (pts, cumulative, headings):
            arr.setflags(write=False)
        self.points: FloatArray = pts
        self.cumulative: FloatArray = cumulative
        self.headings: FloatArray = headings

    @property
    def length(self) -> float:
        return float(self.cumulative[-1])

    def _locate(self, s: ArrayLike) -> tuple[FloatArray, IntArray]:
        """Clamped arc lengths and the index of the segment containing each."""
        sc = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        idx = np.searchsorted(self.cumulative, sc, side="right") - 1
        return sc, np.clip(idx, 0, len(self.points) - 2)

    def point_at(self, s: ArrayLike) -> FloatArray:
        """Point(s) at arc length ``s`` (clamped to ``[0, length]``); shape ``s.shape + (2,)``."""
        sc, i = self._locate(s)
        p0, p1 = self.points[i], self.points[i + 1]
        span = self.cumulative[i + 1] - self.cumulative[i]  # 0 if the segment underflows
        t = np.divide(sc - self.cumulative[i], span, out=np.zeros_like(sc), where=span > 0)
        return p0 + t[..., None] * (p1 - p0)

    def heading_at(self, s: ArrayLike) -> FloatArray:
        """Heading of the segment containing arc length ``s`` (a vertex belongs to the
        segment starting there; the end belongs to the last segment)."""
        return self.headings[self._locate(s)[1]]

    def cut(self, s0: float, s1: float) -> Polyline:
        """The part between arc lengths ``s0 < s1`` (clamped), cut along arc length."""
        a, b = (float(v) for v in np.clip((s0, s1), 0.0, self.length))
        if b - a <= POSITION_EPS:
            raise ValueError(f"cut [{s0}, {s1}] of a {self.length:.3f} m polyline is empty")
        inner = (self.cumulative > a + POSITION_EPS) & (self.cumulative < b - POSITION_EPS)
        return Polyline(np.vstack((self.point_at(a), self.points[inner], self.point_at(b))))


def offset_polyline(
    points: ArrayLike,
    distance: float,
    miter_limit: float = MITER_LIMIT,
    *,
    side: DriveSide = DriveSide.right,
) -> FloatArray:
    """Offset a polyline ``distance`` metres to the ``side`` of travel.

    Interior vertices move along the angle bisector by ``distance / cos(φ/2)`` (φ the
    turning angle), which keeps every segment exactly ``distance`` away. Where that miter
    factor exceeds ``miter_limit`` the vertex is bevelled into two points. A negative
    ``distance`` offsets to the other side. Lanes use ``side=network.drive_side``.
    """
    pts = as_points(points)
    if distance == 0:
        return pts
    vec, length = _segments(pts)
    tangent = vec / length[:, None]
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))  # right of travel
    off = distance if side is DriveSide.right else -distance
    n_in, n_out = normal[:-1], normal[1:]
    denom = 1.0 + np.einsum("ij,ij->i", n_in, n_out)  # 2 cos²(φ/2)
    bevel = denom < 2.0 / miter_limit**2  # 1/cos(φ/2) > miter_limit
    miter = pts[1:-1] + off * (n_in + n_out) / np.where(bevel, 1.0, denom)[:, None]

    counts = np.ones(len(pts), dtype=np.intp)
    counts[1:-1] += bevel
    start = np.cumsum(counts) - counts
    out = np.empty((int(counts.sum()), 2))
    out[0] = pts[0] + off * normal[0]
    out[-1] = pts[-1] + off * normal[-1]
    out[start[1:-1]] = np.where(bevel[:, None], pts[1:-1] + off * n_in, miter)
    out[start[1:-1][bevel] + 1] = pts[1:-1][bevel] + off * n_out[bevel]
    return out


def curvature_radii(points: ArrayLike) -> FloatArray:
    """Circumradius of every three consecutive points (``n - 2`` values; inf if collinear)."""
    pts = as_points(points)
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ca = (np.linalg.norm(q - p, axis=1) for p, q in ((a, b), (b, c), (c, a)))
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    num = ab * bc * ca
    return np.divide(num, 2.0 * cross, out=np.full_like(num, np.inf), where=cross > 0)
