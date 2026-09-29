"""Acceleration, safe-speed cap and ballistic update of every running vehicle (plan G.1-G.3).

Final acceleration ``a = min(a(leader), a(obstacle), a_lim)``: the car-following model is
called twice, for the leader and for the virtual obstacle (a stationary leader), and the
result is clamped to ``[-b_emerg, a]``. Speed-limit anticipation ``a_lim`` brakes toward
the vehicle's own desired speed on its next planned link, and the no-overshoot rule keeps
a positive acceleration from passing ``max(v, v0)`` within the step (see
:func:`anticipation` for the last step before a slower link). ``speed_override`` replaces
``v0``. The G.2 cap then uses ``C = min`` over the leader
(``g + v_L^2/(2 b_L) - s_m``) and every active obstacle (``g`` itself), and G.3 integrates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.types import FloatArray, IntArray
from urbanflow.engine.leaders import Leaders
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.vehicles.car_following import (
    CarFollowingInputs,
    CarFollowingModel,
    ballistic,
    no_overshoot,
    safe_speed,
    stop_budget,
)
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = ["Longitudinal", "anticipation", "expire_overrides", "longitudinal_step"]


@dataclass(frozen=True, slots=True)
class Longitudinal:
    """Output of :func:`longitudinal_step`, aligned with ``run``."""

    dx: FloatArray
    """Distance travelled this step, m."""
    v_start: FloatArray
    """Speed at the start of the step, m/s (``veh.speed`` now holds the new speed)."""
    violations: int
    """Vehicles whose G.3 update broke the safe-speed cap (``safety_cap_violations``)."""


def expire_overrides(veh: VehicleTable, run: IntArray, time: float) -> None:
    """Clear ``set_speed`` overrides whose ``override_until`` has been reached."""
    h = run[~np.isnan(veh.speed_override[run]) & (veh.override_until[run] <= time)]
    veh.speed_override[h] = np.nan
    veh.override_until[h] = np.inf


def _next_link(
    net: CompiledNetwork, veh: VehicleTable, types: VehicleTypes, run: IntArray
) -> tuple[FloatArray, FloatArray]:
    """``(v_n, d)``: own desired speed on the next planned link (NaN if none) and the
    distance to it. The next planned link is the planned connector of a lane vehicle or
    the ``to_lane`` of a connector vehicle; ``v_n = min(max_speed, f * limit)`` or the
    speed override."""
    n_lanes = net.n_lanes
    link = veh.link[run].astype(np.intp)
    nxt = veh.next_conn[run].astype(np.intp)  # lanes: the planned connector
    conn = link >= n_lanes
    nxt[conn] = net.conn_to_lane[link[conn] - n_lanes]  # connectors: their to_lane
    ti = veh.type_idx[run]
    factor = veh.speed_factor[run].astype(np.float64)
    v_n = np.minimum(types.max_speed[ti], factor * net.link_speed_limit[np.maximum(nxt, 0)])
    override = veh.speed_override[run]
    v_n = np.where(np.isnan(override), v_n, override)
    return np.where(nxt >= 0, v_n, np.nan), net.link_length[link] - veh.pos[run]


def anticipation(
    net: CompiledNetwork, veh: VehicleTable, types: VehicleTypes, run: IntArray, dt: float
) -> tuple[FloatArray, FloatArray]:
    """G.1 speed-limit anticipation: ``(a_lim, v0_end)`` toward the next planned link.

    With ``r = (v^2 - v_n^2) / (2 d)`` over the distance ``d`` to the next planned link,
    ``a_lim = -min(r, b_emerg)`` when ``v_n < v`` and ``r > b/2``; otherwise ``+inf``.
    Discrete-time completion: when the link is within reach of this step
    (``d <= v dt + a dt^2/2``), a vehicle faster than ``v_n`` brakes whatever the
    threshold, but only down to ``v_n`` at the end of the step: ``a_lim = max(-min(r,
    b_emerg), (v_n - v)/dt)`` (braking at ``r`` for the whole step would continue on the
    link after reaching ``v_n`` there, well below it); and ``v0_end = min(v0, v_n)`` is the
    no-overshoot target, so the step ends at no more than ``v_n`` (``v0_end = v0``
    otherwise).
    """
    v_n, d = _next_link(net, veh, types, run)
    ti = veh.type_idx[run]
    v = veh.speed[run]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (v * v - v_n * v_n) / (2 * d)
    reach = d <= v * dt + types.accel[ti] * dt * dt / 2
    brake = (v_n < v) & ((r > C.ANTICIPATION_DECEL_FACTOR * types.decel[ti]) | reach)
    a_lim = np.where(brake, -np.minimum(r, types.emergency_decel[ti]), np.inf)
    a_lim = np.where(reach & brake, np.maximum(a_lim, (v_n - v) / dt), a_lim)
    override = veh.speed_override[run]
    v0 = np.where(np.isnan(override), veh.v0[run], override)
    return a_lim, np.where(reach & (v_n < v0), v_n, v0)


def longitudinal_step(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    model: CarFollowingModel,
    run: IntArray,
    leaders: Leaders,
    obstacle_gap: FloatArray,
    cap_gap: FloatArray,
    *,
    dt: float,
    rng: np.random.Generator,
) -> Longitudinal:
    """G.1-G.3 for ``run``: writes ``speed`` and ``accel``; returns ``dx``.

    ``obstacle_gap`` is the IDM gap to the virtual obstacle (``inf`` if none) and
    ``cap_gap`` the distance to every active obstacle point for the G.2 cap (``inf`` if
    none). Overrides must already be expired (:func:`expire_overrides`).
    """
    ti = veh.type_idx[run].astype(np.intp)
    p = types.params.gather(ti)
    v = veh.speed[run].copy()
    override = veh.speed_override[run]
    v0 = np.where(np.isnan(override), veh.v0[run], override)
    uid = veh.uid[run]
    lead = CarFollowingInputs(v, v0, leaders.gap, leaders.leader_speed, dt, uid, rng)
    obst = CarFollowingInputs(v, v0, obstacle_gap, np.zeros(run.size), dt, uid, rng)
    a = np.minimum(model.acceleration(lead, p), model.acceleration(obst, p))
    a_lim, v0_end = anticipation(net, veh, types, run, dt)
    a = no_overshoot(np.clip(np.minimum(a, a_lim), -p.b_emerg, p.a), v, v0_end, dt)
    b_leader = types.emergency_decel[veh.type_idx[np.maximum(leaders.leader, 0)]]
    budget = np.minimum(
        stop_budget(leaders.gap, leaders.leader_speed, b_leader, C.SAFETY_MARGIN), cap_gap
    )
    v_safe = safe_speed(v, budget, p.b_hat, dt)
    v_new, dx, violation = ballistic(v, a, budget, v_safe, p.b_emerg, dt)
    veh.speed[run] = v_new
    veh.accel[run] = (v_new - v) / dt
    return Longitudinal(dx, v, int(violation.sum()))
