"""Lane capacity from the IDM equilibrium flow (plan E.3 ``road_capacity_vph``, E.8 W501).

In IDM steady state at speed ``v`` the bumper-to-bumper gap is ``s_e(v) = (s0 + v T) /
sqrt(1 - (v/v0)^delta)``, so a lane carries ``q(v) = v / (s_e(v) + l)`` vehicles per
second. The capacity is the peak of ``q`` over ``0 < v < v0`` (Treiber & Kesting 2013,
Traffic Flow Dynamics, ch. 11). The simpler ``3600 / (T + (l + s0)/v)`` drops the
``(v/v0)^delta`` term and overstates it (2244 instead of 1792 veh/h at 13.89 m/s).

The peak is found on a fixed grid of ``CAPACITY_GRID_POINTS`` speeds ``v = v0 k / N``
(k = 1 .. N-1). ``q`` is smooth with one interior maximum, so the grid error is second
order in the spacing: below 1e-5 relative for N = 1000. Pure stdlib (the network
compiler and scenario validation both use it; results are cached per argument tuple).
"""

from __future__ import annotations

import math
from functools import lru_cache

from urbanflow.core.constants import (
    CAPACITY_GRID_POINTS,
    IDM_DELTA,
    IDM_HEADWAY,
    IDM_MIN_GAP,
    SECONDS_PER_HOUR,
    VEHICLE_LENGTH,
)

__all__ = ["idm_capacity_vph"]


@lru_cache(maxsize=1024)
def idm_capacity_vph(
    v0: float,
    *,
    headway: float = IDM_HEADWAY,
    min_gap: float = IDM_MIN_GAP,
    length: float = VEHICLE_LENGTH,
    delta: float = IDM_DELTA,
) -> float:
    """Peak IDM equilibrium flow of one lane at desired speed ``v0``, veh/h.

    ``max over 0 < v < v0 of 3600 v / (s_e(v) + length)`` with ``s_e(v) = (min_gap + v
    headway) / sqrt(1 - (v/v0)^delta)``; the defaults are the built-in car's.
    """
    n = CAPACITY_GRID_POINTS
    best = 0.0
    for k in range(1, n):
        u = k / n
        v = v0 * u
        gap = (min_gap + v * headway) / math.sqrt(1.0 - u**delta)
        best = max(best, v / (gap + length))
    return SECONDS_PER_HOUR * best
