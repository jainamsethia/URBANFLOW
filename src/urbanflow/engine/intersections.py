"""Intersection admission at stop lines (plan F.3), for unsignalised intersections.

For a running vehicle on a lane that is **not** on the last road of its route, with
``d = L - pos``, planned connector ``c`` and decision distance
``D = v^2/(2b) + v dt + DECISION_MARGIN``:

* ``next_conn = -1`` (a lane change is needed): an obstacle at the lane end, ``held``;
* candidates are the first uncommitted vehicle of each lane. They are admitted one by one
  in the order (rank desc, ``d / max(v, 1)`` asc, uid asc), where rank is
  ``mov_static_rank`` (priority 3/2/1, uncontrolled 1):

  1. outside the decision zone (``d > D``): free approach, no obstacle;
  2. exit space ("don't block the box"): ``free(to) - reserved(to) >= len + s0``, where
     ``free`` is :attr:`Leaders.lane_rear` of the target lane: the rear of its most
     upstream body, a vehicle on one of its outgoing connectors hanging back over its end
     included (its length if none);
  3. conflicts: for every crossing or merging zone on ``c`` the occupancy window
     ``W_i = [T-(d + z_in) - tau, T+(d + z_out + len) + tau]`` must not intersect the
     window of any foe: vehicles on the foe connector that have not cleared the zone,
     vehicles that left it for its ``to_lane`` (``lock_conn`` = the foe connector) while
     their rear, at ``L_foe + pos - len`` in foe coordinates, is still before the zone's
     end, committed vehicles on the lane feeding it, and (for a higher-rank foe movement) the
     nearest uncommitted vehicle there if it plans the foe connector (one queued behind
     an uncommitted vehicle of another movement cannot arrive first);
  4. pass: commit (``commit_seq``, ``reserved(to) += len + s0``); fail but able to stop
     (``v^2 <= 2 d b_emerg``): obstacle at the stop line, ``held``; otherwise force-commit
     (``forced``, counted, also reserving).

Obstacles give the IDM gap ``d + s0 - STOP_LINE_CLEARANCE`` (the front stops 0.5 m before
the line) and the G.2 cap gap ``d`` with ``v_L = 0``, ``s_m = 0``.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.types import FloatArray, IntArray
from urbanflow.engine.leaders import Leaders, connector_conflicts
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.network.conflicts import ConflictKind
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = ["Admission", "JunctionIndex", "admit", "eta", "reservations"]

Conflict = tuple[int, float, float, float, float]
"""``(foe connector link, z_in, z_out, foe z_in, foe z_out)``."""


@dataclass(frozen=True, slots=True)
class JunctionIndex:
    """Static per-connector data of the admission loop, as Python lists (built once)."""

    conflicts: tuple[tuple[Conflict, ...], ...]
    """Per connector index: its crossing and merging conflicts, by ``z_in``."""
    rank: tuple[int, ...]
    """Per connector link id: static rank of its movement (lanes: 0)."""

    @classmethod
    def build(cls, net: CompiledNetwork) -> JunctionIndex:
        cc = connector_conflicts(net, (ConflictKind.crossing, ConflictKind.merging))
        rows = list(
            zip(
                cc.other.tolist(),
                cc.zone[:, 0].tolist(),
                cc.zone[:, 1].tolist(),
                cc.other_zone[:, 0].tolist(),
                cc.other_zone[:, 1].tolist(),
                strict=True,
            )
        )
        ptr = cc.ptr.tolist()
        movement = net.link_movement
        ranks = np.r_[net.mov_static_rank, 0]  # index -1 (lanes) reads the padded 0
        rank = ranks[movement]
        return cls(
            conflicts=tuple(tuple(rows[a:b]) for a, b in pairwise(ptr)),
            rank=tuple(rank.tolist()),
        )


@dataclass(frozen=True, slots=True)
class Admission:
    """Output of :func:`admit`; per-vehicle arrays are aligned with ``run``."""

    obstacle_gap: FloatArray
    """IDM gap to the virtual stationary obstacle, m (``inf`` if none)."""
    cap_gap: FloatArray
    """G.2 distance to the obstacle point, m (``inf`` if none)."""
    commits: int
    """Vehicles committed this step, forced ones included."""
    forced: int
    """Force-commits this step (``forced_commits``)."""
    next_seq: int
    """The next unused ``commit_seq`` value."""


def eta(x: float, v: float, a: float, vc: float) -> float:
    """F.3 arrival time over ``x`` metres from speed ``v``: accelerate at ``a`` to ``vc``.

    ``T = (sqrt(v^2 + 2ax) - v)/a`` for ``x <= x_a = max(0, vc^2 - v^2)/(2a)``, else
    ``max(0, vc - v)/a + (x - x_a)/vc`` (so ``x/vc`` once ``v >= vc``); 0 for ``x <= 0``.
    """
    if x <= 0:
        return 0.0
    xa = max(0.0, vc * vc - v * v) / (2 * a)
    if x <= xa:
        return (math.sqrt(v * v + 2 * a * x) - v) / a
    if vc <= 0:
        return math.inf
    return max(0.0, vc - v) / a + (x - xa) / vc


def reservations(
    net: CompiledNetwork, veh: VehicleTable, types: VehicleTypes, run: IntArray
) -> FloatArray:
    """``reserved(l)``: sum of ``len + s0`` of committed vehicles headed for lane ``l``.

    Committed vehicles are those admitted on their approach lane and those on a connector
    (``next_conn`` is then the connector itself); both are not yet on ``l``.
    """
    reserved = np.zeros(net.n_lanes)
    h = run[veh.committed[run]]
    to = net.conn_to_lane[veh.next_conn[h] - net.n_lanes]
    np.add.at(reserved, to, veh.length[h] + types.min_gap[veh.type_idx[h]])
    return reserved


def admit(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    run: IntArray,
    leaders: Leaders,
    remaining: IntArray,
    reserved: FloatArray,
    index: JunctionIndex,
    *,
    dt: float,
    next_seq: int,
) -> Admission:
    """F.3 admission of this step's stop-line candidates (see the module docstring).

    Writes ``committed``, ``commit_seq``, ``forced`` and ``held``; updates ``reserved`` in
    place. ``remaining`` comes from :func:`~urbanflow.engine.leaders.remaining_roads`.
    """
    n, n_lanes = run.size, net.n_lanes
    obstacle = np.full(n, np.inf)
    cap = np.full(n, np.inf)
    veh.held[run] = False
    if n == 0:
        return Admission(obstacle, cap, 0, 0, next_seq)
    link = veh.link[run].astype(np.intp)
    pos = veh.pos[run]
    speed = veh.speed[run]
    length = veh.length[run].astype(np.float64)
    ti = veh.type_idx[run]
    s0 = types.min_gap[ti]
    nc = veh.next_conn[run].astype(np.intp)
    on_lane = link < n_lanes
    d = net.link_length[link] - pos
    continuing = on_lane & (remaining > 0)
    held: list[int] = np.flatnonzero(continuing & (nc < 0)).tolist()  # lane end

    # candidates: the first uncommitted vehicle of each lane, in admission order
    perm = leaders.perm
    free_lane = perm[(on_lane & ~veh.committed[run])[perm]]
    fl = link[free_lane]
    front = free_lane[np.r_[fl[1:] != fl[:-1], True]] if free_lane.size else free_lane
    cand = front[continuing[front] & (nc[front] >= 0)]
    rank_of = index.rank
    cand_rank = np.array([rank_of[c] for c in nc[cand].tolist()], dtype=np.intp)
    urgency = d[cand] / np.maximum(speed[cand], C.MIN_EFFECTIVE_SPEED)
    cand = cand[np.lexsort((veh.uid[run][cand], urgency, -cand_rank))]

    # personal cruise speed on the vehicle's (planned or current) connector
    conn = np.where(on_lane, nc, link)
    factor = veh.speed_factor[run].astype(np.float64)
    cruise = np.minimum(types.max_speed[ti], factor * net.link_speed_limit[np.maximum(conn, 0)])
    override = veh.speed_override[run]
    cruise = np.where(np.isnan(override), cruise, override)

    # plain lists for the sequential loop (F.3 complexity note)
    run_l = run.tolist()
    pos_l, v_l, len_l, d_l = pos.tolist(), speed.tolist(), length.tolist(), d.tolist()
    a_l, vc_l = types.accel[ti].tolist(), cruise.tolist()
    b_l, be_l = types.decel[ti].tolist(), types.emergency_decel[ti].tolist()
    s0_l, nc_l = s0.tolist(), nc.tolist()
    committed_l = veh.committed[run].tolist()
    lock_l = veh.lock_conn[run].tolist()
    link_len_l: list[float] = net.link_length.tolist()
    perm_l: list[int] = perm.tolist()
    slink_l: list[int] = link[perm].tolist()
    to_lane_l, from_lane_l = net.conn_to_lane.tolist(), net.conn_from_lane.tolist()
    tau = C.GAP_ACCEPT_MARGIN

    def window(j: int, x_in: float, x_out: float, inside: bool) -> tuple[float, float]:
        v, a, vc = v_l[j], a_l[j], vc_l[j]
        start = -math.inf if inside else eta(x_in, v, a, max(v, vc)) - tau
        return start, eta(x_out, v, a * C.ETA_END_ACCEL_FACTOR, vc) + tau

    def on_link(lk: int) -> list[int]:
        """Positions of the vehicles on link ``lk``, upstream first."""
        return perm_l[bisect_left(slink_l, lk) : bisect_right(slink_l, lk)]

    def conflicts_ok(i: int) -> bool:
        ci, di = nc_l[i], d_l[i]
        for foe, z_in, z_out, fz_in, fz_out in index.conflicts[ci - n_lanes]:
            s_i, e_i = window(i, di + z_in, di + z_out + len_l[i], False)
            for j in on_link(foe):  # on the foe connector, zone not cleared
                if pos_l[j] - len_l[j] >= fz_out:
                    continue
                s, e = window(j, fz_in - pos_l[j], fz_out - pos_l[j] + len_l[j], pos_l[j] >= fz_in)
                if s <= e_i and s_i <= e:
                    return False
            for j in on_link(to_lane_l[foe - n_lanes]):  # left it, rear still on it
                if pos_l[j] >= len_l[j]:
                    break  # upstream first: every later rear is on the lane
                p = link_len_l[foe] + pos_l[j]  # front in foe coordinates (>= fz_in)
                if lock_l[j] != foe or p - len_l[j] >= fz_out:
                    continue
                s, e = window(j, fz_in - p, fz_out - p + len_l[j], True)
                if s <= e_i and s_i <= e:
                    return False
            # committed vehicles planning the foe connector, and (higher-rank movement
            # only) the lane's first uncommitted vehicle if it plans it: vehicles queued
            # behind an uncommitted one cannot reach the stop line before it
            higher = rank_of[foe] > rank_of[ci]
            for j in reversed(on_link(from_lane_l[foe - n_lanes])):  # nearest first
                if nc_l[j] == foe and (committed_l[j] or higher):
                    s, e = window(j, d_l[j] + fz_in, d_l[j] + fz_out + len_l[j], False)
                    if s <= e_i and s_i <= e:
                        return False
                higher = higher and committed_l[j]
        return True

    seq, commits, forced = next_seq, 0, 0
    for i in cand.tolist():
        di, vi = d_l[i], v_l[i]
        # (signal rules for r/y movements come first at signalised intersections, P3)
        if di > vi * vi / (2 * b_l[i]) + vi * dt + C.DECISION_MARGIN:
            continue  # free approach
        to = to_lane_l[nc_l[i] - n_lanes]
        need = len_l[i] + s0_l[i]
        ok = float(leaders.lane_rear[to]) - reserved[to] >= need and conflicts_ok(i)
        # can it still stop? (v^2/(2d) <= b_emerg, with G.3's float tolerance: the cap
        # leaves a held vehicle exactly on this boundary)
        if not ok and vi * vi <= 2 * di * (be_l[i] + C.BALLISTIC_FLOOR):
            held.append(i)  # stop at the line
            continue
        h = run_l[i]
        veh.committed[h] = committed_l[i] = True
        veh.commit_seq[h] = seq
        seq += 1
        reserved[to] += need
        commits += 1
        if not ok:  # cannot stop any more: force-commit
            veh.forced[h] = True
            forced += 1

    stop = np.asarray(held, dtype=np.intp)
    obstacle[stop] = d[stop] + s0[stop] - C.STOP_LINE_CLEARANCE
    cap[stop] = d[stop]
    veh.held[run[stop]] = True
    return Admission(obstacle, cap, commits, forced, seq)
