"""Connector geometry and curvature speed limits (plan E.5 rule 4).

A connector joins the end of an incoming lane to the start of an outgoing lane. Without an
explicit ``shape`` it is a cubic Bezier approximating a circular arc
(:func:`urbanflow.geometry.bezier_connector`). Its speed limit is
``min(v_in, v_out, sqrt(a_lat * R_min))``, with ``R_min`` the smallest three-point
circumradius of the sampled polyline; this replaces CityFlow's 10000 m/s connector limit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from urbanflow.core.constants import POSITION_EPS, TURN_LATERAL_ACCEL
from urbanflow.core.types import FloatArray
from urbanflow.geometry import Polyline, bezier_connector, curvature_radii
from urbanflow.network.lanes import DegenerateGeometryError

__all__ = ["build_connector", "connector_speed_limit", "direction"]


def direction(heading: float) -> FloatArray:
    """Unit vector of a heading (radians, counter-clockwise from +x)."""
    return np.array((math.cos(heading), math.sin(heading)))


def build_connector(
    start: ArrayLike,
    d_start: ArrayLike,
    end: ArrayLike,
    d_end: ArrayLike,
    shape: Sequence[tuple[float, float]] | None = None,
) -> Polyline:
    """Connector polyline from ``start`` (heading ``d_start``) to ``end`` (heading ``d_end``).

    An explicit ``shape`` is used as-is with its first and last points snapped to ``start``
    and ``end`` (points that then repeat their predecessor are dropped). Raises
    :class:`DegenerateGeometryError` for a connector of zero length.
    """
    s = np.asarray(start, dtype=np.float64).reshape(2)
    e = np.asarray(end, dtype=np.float64).reshape(2)
    if shape is not None:
        pts = np.array(shape, dtype=np.float64)
        pts[0], pts[-1] = s, e
        keep = np.concatenate(([True], (np.diff(pts, axis=0) != 0).any(axis=1)))
        pts = pts[keep]
        if len(pts) < 2:
            raise DegenerateGeometryError("the snapped connector shape has zero length")
        return Polyline(pts)
    if math.dist((s[0], s[1]), (e[0], e[1])) <= POSITION_EPS:
        raise DegenerateGeometryError(
            "the connector has zero length (the lanes meet at one point; "
            "give the intersection a radius)"
        )
    return Polyline(bezier_connector(s, d_start, e, d_end))


def connector_speed_limit(points: ArrayLike, v_in: float, v_out: float) -> float:
    """``min(v_in, v_out, sqrt(TURN_LATERAL_ACCEL * R_min))`` for a connector polyline, m/s."""
    radii = curvature_radii(points)
    r_min = float(radii.min()) if len(radii) else math.inf
    return min(v_in, v_out, math.sqrt(TURN_LATERAL_ACCEL * r_min))
