"""Cubic Bézier connectors that approximate circular arcs (plan E.5 rule 4)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from urbanflow.core.constants import CONNECTOR_MIN_SAMPLES, CONNECTOR_SAMPLE_ANGLE, GEOM_EPS
from urbanflow.core.types import FloatArray

__all__ = ["bezier_connector", "circular_handle", "connector_samples", "turn_angle"]


def _unit(v: ArrayLike, what: str) -> FloatArray:
    arr = np.asarray(v, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(arr))
    if not 0 < norm < math.inf:
        raise ValueError(f"{what} must be a non-zero finite direction")
    return arr / norm


def turn_angle(d_start: ArrayLike, d_end: ArrayLike) -> float:
    """Angle θ ∈ [0, π] between two directions (0 = straight on, π = U-turn)."""
    a, b = _unit(d_start, "d_start"), _unit(d_end, "d_end")
    return math.atan2(abs(a[0] * b[1] - a[1] * b[0]), float(a @ b))


def circular_handle(theta: float, chord: float) -> float:
    """Bézier handle length ``h = 4/3 · tan(θ/4) · R`` with ``R = chord / (2 sin(θ/2))``.

    Evaluated as the equal closed form ``chord / (3 cos²(θ/4))``, which has no 0/0 at
    θ → 0 (limit ``chord / 3``, a straight line) and gives ``2/3 · chord`` for a U-turn.
    """
    return chord / (3.0 * math.cos(theta / 4.0) ** 2)


def connector_samples(theta: float) -> int:
    """Sample count ``max(2, ceil(θ / (π/16)) + 1)`` for a connector turning by ``θ``."""
    steps = math.ceil(theta / CONNECTOR_SAMPLE_ANGLE - GEOM_EPS)
    return max(CONNECTOR_MIN_SAMPLES, steps + 1)


def bezier_connector(
    start: ArrayLike,
    d_start: ArrayLike,
    end: ArrayLike,
    d_end: ArrayLike,
    n: int | None = None,
) -> FloatArray:
    """Sample the cubic ``S, S + h·ds, E - h·de, E`` at ``n`` uniform parameter values.

    ``d_start`` is the travel direction leaving ``start`` (end of the incoming lane) and
    ``d_end`` the direction arriving at ``end`` (start of the outgoing lane). Endpoints and
    end tangents are exact. ``n`` defaults to :func:`connector_samples`.
    """
    s = np.asarray(start, dtype=np.float64).reshape(2)
    e = np.asarray(end, dtype=np.float64).reshape(2)
    ds, de = _unit(d_start, "d_start"), _unit(d_end, "d_end")
    chord = float(np.linalg.norm(e - s))
    if not 0 < chord < math.inf:
        raise ValueError("connector start and end must be finite and distinct")
    theta = turn_angle(ds, de)
    count = connector_samples(theta) if n is None else n
    if count < CONNECTOR_MIN_SAMPLES:
        raise ValueError(f"a connector needs at least {CONNECTOR_MIN_SAMPLES} samples, got {n}")
    h = circular_handle(theta, chord)
    ctrl = np.stack((s, s + h * ds, e - h * de, e))
    t = np.linspace(0.0, 1.0, count)[:, None]
    u = 1.0 - t
    pts: FloatArray = (
        u**3 * ctrl[0] + 3 * u**2 * t * ctrl[1] + 3 * u * t**2 * ctrl[2] + t**3 * ctrl[3]
    )
    return pts
