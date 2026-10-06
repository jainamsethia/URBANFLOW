"""Intersection admission at stop lines and zone locks inside intersections (plan F.3, H.5).

**Admission** (:func:`admit`). For a running vehicle on a lane that is **not** on the last
road of its route, with ``d = L - pos``, planned connector ``c`` and decision distance
``D = v^2/(2b) + v dt + DECISION_MARGIN``:

* ``next_conn = -1`` (a lane change is needed): an obstacle at the lane end, ``held``;
* candidates are the first uncommitted vehicle of each lane. They are admitted one by one
  in the order (rank desc, ``d / max(v, 1)`` asc, uid asc), where rank is the movement's
  signal state code at signalised intersections (G 3, g 2; y 1 and r 0 only order the
  loop) and ``mov_static_rank`` elsewhere (priority 3/2/1, uncontrolled 1):

  1. signal (signalised movements): red -> obstacle at the stop line if ``v^2/(2d) <=
     b_emerg``, otherwise a force-commit counted in ``red_runs`` (only after a forced
     ``set_phase`` or with yellow = 0); yellow -> obstacle if ``v^2/(2d) <=
     YELLOW_MAX_DECEL``, otherwise a *dilemma* vehicle: it skips step 2 and commits if
     steps 3 and 4 pass, else stops if it still can, else is force-committed (step 5);
     G, g and unsignalised movements continue;
  2. outside the decision zone (``d > D``): free approach, no obstacle;
  3. exit space ("don't block the box"): ``free(to) - reserved(to) >= len + s0``, where
     ``free`` is :attr:`Leaders.lane_free` of the target lane (B.2 #25): the rear of its
     most upstream body (a vehicle on one of its outgoing connectors hanging back over its
     end included; the lane length if none) plus ``v^2 / (2 b_emerg)`` of that body, the
     distance it still travels even under emergency braking, so a moving platoon does not
     look parked;
  4. conflicts: for every crossing or merging zone on ``c`` the occupancy window
     ``W_i = [T-(d + z_in) - tau, T+(d + z_out + len) + tau]`` must not intersect the
     window of any foe: vehicles on the foe connector that have not cleared the zone,
     vehicles that left it for its ``to_lane`` whose lock is still held (``lock_conn`` =
     the foe connector) while their rear, at ``L_foe + pos - len`` in foe coordinates, is
     before the zone's end, committed vehicles on the lane feeding it, and (for a
     higher-rank foe movement) the nearest uncommitted vehicle there if it plans the foe
     connector (one queued behind an uncommitted vehicle of another movement cannot
     arrive first);
  5. pass: commit (``commit_seq``, ``reserved(to) += len + s0``); fail but able to stop
     (``v^2 <= 2 d b_emerg``): obstacle at the stop line, ``held``; otherwise force-commit
     (``forced``, counted in ``forced_commits``, also reserving).

**End-of-green clearing** ("sneakers", B.2 #25). While a permissive (g) movement shows
the yellow that ends its green, the candidate planning it that waited at its stop line
since before the yellow commits if the exit space of step 3 passes, without step 4's
windows. Eligibility is the caller's per-lane ``sneakers`` array: in every green step,
admission records there the uid of the lane's candidate it held at the line (within
``DECISION_MARGIN`` of it) on a g movement; the caller resets it to -1 outside the yellow,
so in the yellow it names the vehicle held at the line in the last green step. A vehicle
that arrived or braked during the yellow is not eligible, and a commit consumes the entry
(``SNEAKERS_PER_PHASE`` = 1 per approach lane and yellow). It must still be within
``DECISION_MARGIN`` of the line. Sneakers are admitted after every other candidate of the
step, so the dilemma vehicles of the opposing movement commit first. Their zone locks
(below) then order them after every earlier-committed foe and before the next phase's
vehicles.

Obstacles give the IDM gap ``d + s0 - STOP_LINE_CLEARANCE`` (the front stops 0.5 m before
the line) and the G.2 cap gap ``d`` with ``v_L = 0``, ``s_m = 0``. Committed vehicles never
re-check the light (H.5). The foes of step 4 need no separate "signal permits" test: at a
signalised intersection a rank above the candidate's (at least y) means g or G.

**Zone locks** (:func:`grant_zones`, after admission). All lock state lives in vehicle
columns: ``granted`` and ``lock_conn`` (the connector whose zones are locked), plus
``forced`` and ``commit_seq``. A vehicle's front and rear on connector ``c`` are ``pos``
(and ``pos - len``) on ``c``, shifted by ``-L_from`` on its ``from_lane`` and ``+L_c`` on
its ``to_lane``. Its *remaining* zones are the crossing and merging zones of ``c`` whose
``z_out`` its rear has not passed; ``v`` holds (locks) the remaining zones of
``lock_conn[v]``. Committed vehicles are processed in ``commit_seq`` order:

* no remaining zone (a connector without crossing or merging zones): granted at once;
* force-committed (``forced``): granted at once (a *force grant*);
* otherwise it requests the whole remaining set once the path distance ``dist`` to its
  first remaining zone is ``<= D + a dt^2/2`` (``D`` plus the extra distance a step of
  acceleration can cover, so no zone is reached before its request). The request is
  granted iff (a) no zone of the set is locked by another vehicle (the owner's rear is
  before the zone's ``z_out`` on the foe connector); (b) no committed, not-yet-granted
  foe with a smaller ``commit_seq`` is predicted to arrive before it clears (``T-`` of the
  foe to its ``z_in`` minus tau < ``T+`` of the vehicle past its ``z_out`` plus its
  length, plus tau); (c) its step-4 leader, if it leaves the same approach lane (the
  lane itself, its connector or a diverging sibling), is not committed-but-ungranted;
  and (d) it does not still hold the lock of the previous connector (with E806-sized
  lanes that lock is gone before the front reaches the next stop line);
* not granted and ``v^2 > 2 dist b_emerg`` (it cannot stop before the zone): force grant,
  ``forced`` = True;
* not granted: an obstacle at the first remaining zone's ``z_in`` (IDM gap ``dist + s0 -
  STOP_LINE_CLEARANCE``, G.2 cap gap ``dist``), ``held``.

A grant sets ``granted`` and ``lock_conn = c``. A force grant **cascades** to every
committed-but-ungranted vehicle ahead in the step-4 leader chain from the same approach
lane (they are granted and ``forced`` too), and **revokes** the grant of every other
owner of an overlapping zone that is not forced, has not entered any of its zones and can
still stop before the first (``v^2 <= 2 dist b_emerg``): its lock is released,
``granted`` = False, it is held this step and requests again from the next. Overlaps
left over are counted in ``zone_conflicts`` (and exempt from invariant I5 because one
side is forced). A force grant while the previous lock is still held keeps that lock and
takes the new one in the first later step it is free (overlaps counted).

Locks are released by :func:`release_locks` once the rear has passed the last ``z_out`` of
``lock_conn`` (``forced`` is cleared then too, unless the vehicle is committed again),
and on revocation, teleport, arrival and removal.

:func:`signal_lookahead` adds the F.1 step-4 rule: the stop line that ends a committed or
connector vehicle's lookahead is an IDM obstacle when its movement is r or y and the
vehicle can still stop there by step 1's test (B.2 #25).
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.types import (
    SIGNAL_CODE_UNSIGNALISED,
    BoolArray,
    FloatArray,
    IntArray,
    IntersectionKind,
    SignalState,
)
from urbanflow.engine.leaders import ConnectorConflicts, Leaders, connector_conflicts
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.network.conflicts import ConflictKind
from urbanflow.routing.base import RouteTable
from urbanflow.signals.program import StateArray
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = [
    "Admission",
    "Grants",
    "JunctionIndex",
    "admit",
    "eta",
    "grant_zones",
    "link_signal_state",
    "release_locks",
    "reservations",
    "signal_lookahead",
]

Conflict = tuple[int, float, float, float, float]
"""``(foe connector link, z_in, z_out, foe z_in, foe z_out)``."""


@dataclass(frozen=True, slots=True)
class JunctionIndex:
    """Static per-connector data of admission and zone locks (built once)."""

    conflicts: tuple[tuple[Conflict, ...], ...]
    """Per connector index: its crossing and merging conflicts, by ``z_in``."""
    rank: tuple[int, ...]
    """Per link id: static rank of its movement (lanes and signalised movements: 0)."""
    movement_of: Mapping[tuple[int, int], int]
    """``(from road, to road) -> movement``."""
    zones: ConnectorConflicts
    """The same crossing and merging conflicts as arrays (invariant I5)."""
    lock_end: FloatArray
    """Per connector index: the last ``z_out`` of its zones (``-inf`` if none): a lock on
    it is released once the rear passes this point."""

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
        pairs = zip(net.mov_from_road.tolist(), net.mov_to_road.tolist(), strict=True)
        lock_end = np.full(net.n_conn, -np.inf)
        owner = np.repeat(np.arange(net.n_conn), np.diff(cc.ptr))
        np.maximum.at(lock_end, owner, cc.zone[:, 1])
        return cls(
            conflicts=tuple(tuple(rows[a:b]) for a, b in pairwise(ptr)),
            rank=tuple(rank.tolist()),
            movement_of=MappingProxyType({pair: m for m, pair in enumerate(pairs)}),
            zones=cc,
            lock_end=lock_end,
        )


def link_signal_state(net: CompiledNetwork, movement_state: StateArray | None) -> IntArray:
    """Per link: the signal state code of its movement (255 for lanes and unsignalised)."""
    if movement_state is None:
        return np.full(net.n_links, SIGNAL_CODE_UNSIGNALISED, dtype=np.intp)
    padded: IntArray = np.r_[movement_state, SIGNAL_CODE_UNSIGNALISED].astype(np.intp)
    return padded[net.link_movement]


@dataclass(frozen=True, slots=True)
class Admission:
    """Output of :func:`admit`; per-vehicle arrays are aligned with ``run``."""

    obstacle_gap: FloatArray
    """IDM gap to the virtual stationary obstacle, m (``inf`` if none)."""
    cap_gap: FloatArray
    """G.2 distance to the obstacle point, m (``inf`` if none)."""
    hold_link: IntArray
    """Link of the obstacle point of a held vehicle (its lane; -1 if not held)."""
    hold_pos: FloatArray
    """Position of that point on ``hold_link`` (the lane end; NaN if not held)."""
    commits: int
    """Vehicles committed this step, forced ones and sneakers included."""
    forced: int
    """Force-commits by steps 1 (yellow dilemma) and 5 this step (``forced_commits``)."""
    red_runs: int
    """Force-commits at red this step (``red_runs``)."""
    sneakers: int
    """End-of-green clearing commits this step (included in ``commits``)."""
    next_seq: int
    """The next unused ``commit_seq`` value."""


@dataclass(frozen=True, slots=True)
class Grants:
    """Output of :func:`grant_zones`; per-vehicle arrays are aligned with ``run``."""

    obstacle_gap: FloatArray
    """IDM gap to the first remaining zone of a vehicle whose request was not granted, m
    (``inf`` if none)."""
    cap_gap: FloatArray
    """G.2 distance to that zone's ``z_in``, m (``inf`` if none)."""
    hold_link: IntArray
    """The held vehicle's connector (-1 if not held by a zone)."""
    hold_pos: FloatArray
    """``z_in`` of the zone on it (NaN if not held by a zone)."""
    granted: int
    """Grants this step, force grants included."""
    forced: int
    """Force grants this step (force-committed, cannot stop, cascade)."""
    revoked: int
    """Grants revoked by force grants this step."""
    conflicts: int
    """Overlapping owners a force grant could not revoke (``zone_conflicts``)."""


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


def _cruise(
    net: CompiledNetwork, veh: VehicleTable, types: VehicleTypes, run: IntArray, conn: IntArray
) -> FloatArray:
    """Personal cruise speed on connector ``conn`` (F.3): ``min(max_speed, f v_c)`` or the
    speed override."""
    ti = veh.type_idx[run]
    factor = veh.speed_factor[run].astype(np.float64)
    cruise = np.minimum(types.max_speed[ti], factor * net.link_speed_limit[np.maximum(conn, 0)])
    override = veh.speed_override[run]
    return np.where(np.isnan(override), cruise, override)


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
    movement_state: StateArray | None = None,
    permissive: BoolArray | None = None,
    sneakers: IntArray | None = None,
) -> Admission:
    """F.3 admission of this step's stop-line candidates (see the module docstring).

    Writes ``committed``, ``commit_seq``, ``forced`` and ``held``; updates ``reserved`` in
    place. ``remaining`` comes from :func:`~urbanflow.engine.leaders.remaining_roads`;
    ``movement_state`` is ``SignalRuntime.movement_state`` (None: no signals).
    ``permissive`` marks the movements showing the yellow that ends a g
    (:meth:`~urbanflow.signals.state.SignalRuntime.permissive_yellow`) and ``sneakers``
    holds per lane the uid of the vehicle eligible for end-of-green clearing, -1 if none
    (updated in place: set for candidates held at the line on a g movement, consumed by a
    commit); both None: no end-of-green clearing.
    """
    n, n_lanes = run.size, net.n_lanes
    obstacle = np.full(n, np.inf)
    cap = np.full(n, np.inf)
    hold_link = np.full(n, -1, dtype=np.intp)
    hold_pos = np.full(n, np.nan)
    veh.held[run] = False
    if n == 0:
        return Admission(obstacle, cap, hold_link, hold_pos, 0, 0, 0, 0, next_seq)
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
    state = link_signal_state(net, movement_state)
    signalised = state != SIGNAL_CODE_UNSIGNALISED
    rank_of: list[int] = np.where(signalised, state, index.rank).tolist()
    state_of: list[int] = state.tolist()
    cand_rank = np.array([rank_of[c] for c in nc[cand].tolist()], dtype=np.intp)
    urgency = d[cand] / np.maximum(speed[cand], C.MIN_EFFECTIVE_SPEED)
    cand = cand[np.lexsort((veh.uid[run][cand], urgency, -cand_rank))]
    sneak_of: list[bool] = (
        [False] * net.n_links
        if permissive is None or sneakers is None
        else np.r_[permissive, False][net.link_movement].tolist()
    )

    # personal cruise speed on the vehicle's (planned or current) connector
    cruise = _cruise(net, veh, types, run, np.where(on_lane, nc, link))

    # plain lists for the sequential loop (F.3 complexity note). ponytail: a Python loop over
    # the candidates; engine/accel/_numba.py::admit with this signature when it dominates
    run_l = run.tolist()
    pos_l, v_l, len_l, d_l = pos.tolist(), speed.tolist(), length.tolist(), d.tolist()
    a_l, vc_l = types.accel[ti].tolist(), cruise.tolist()
    b_l, be_l = types.decel[ti].tolist(), types.emergency_decel[ti].tolist()
    s0_l, nc_l, link_l = s0.tolist(), nc.tolist(), link.tolist()
    committed_l = veh.committed[run].tolist()
    uid_l: list[int] = veh.uid[run].tolist()
    lock_l = veh.lock_conn[run].tolist()
    link_len_l: list[float] = net.link_length.tolist()
    perm_l: list[int] = perm.tolist()
    slink_l: list[int] = link[perm].tolist()
    to_lane_l, from_lane_l = net.conn_to_lane.tolist(), net.conn_from_lane.tolist()
    free_l: list[float] = leaders.lane_free.tolist()
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

    seq, commits, forced, red_runs = next_seq, 0, 0, 0

    def commit(i: int, ok: bool) -> None:
        nonlocal seq, commits
        h = run_l[i]
        veh.committed[h] = committed_l[i] = True
        veh.commit_seq[h] = seq
        veh.forced[h] = not ok
        seq += 1
        reserved[to_lane_l[nc_l[i] - n_lanes]] += len_l[i] + s0_l[i]
        commits += 1

    red, yellow, permissive_g = SignalState.r.code, SignalState.y.code, SignalState.g.code
    eligible: list[int] = [-1] * n_lanes if sneakers is None else sneakers.tolist()
    sneaks: list[int] = []
    for i in cand.tolist():
        di, vi, ci = d_l[i], v_l[i], nc_l[i]
        # can it still stop? (v^2/(2d) <= b_emerg, with G.3's float tolerance: the cap
        # leaves a held vehicle exactly on this boundary)
        can_stop = vi * vi <= 2 * di * (be_l[i] + C.BALLISTIC_FLOOR)
        light = state_of[ci]
        if (
            light == yellow
            and sneak_of[ci]
            and eligible[link_l[i]] == uid_l[i]
            and di <= C.DECISION_MARGIN
        ):
            sneaks.append(i)  # end-of-green clearing, after everybody else
            continue
        if (light == red and can_stop) or (
            light == yellow and vi * vi <= 2 * di * C.YELLOW_MAX_DECEL
        ):
            held.append(i)  # stop at the line
            continue
        to = to_lane_l[ci - n_lanes]
        need = len_l[i] + s0_l[i]
        if light == red:  # cannot stop before a red line (after set_phase or yellow = 0)
            ok, red_runs = False, red_runs + 1
        else:
            if light != yellow and di > vi * vi / (2 * b_l[i]) + vi * dt + C.DECISION_MARGIN:
                continue  # free approach (a yellow dilemma vehicle decides now)
            ok = free_l[to] - reserved[to] >= need and conflicts_ok(i)
            if not ok and can_stop:
                held.append(i)
                continue
            forced += not ok  # cannot stop any more: force-commit
        commit(i, ok)

    sneaked = 0
    for i in sneaks:
        to = to_lane_l[nc_l[i] - n_lanes]
        if sneakers is not None and free_l[to] - reserved[to] >= len_l[i] + s0_l[i]:
            sneakers[link_l[i]] = -1  # consumed: one per lane and yellow
            sneaked += 1
            commit(i, True)
        else:
            held.append(i)
    if sneakers is not None:  # eligibility: held at the line while its movement is g
        for i in held:
            ci = nc_l[i]
            if ci >= 0 and state_of[ci] == permissive_g and d_l[i] <= C.DECISION_MARGIN:
                sneakers[link_l[i]] = uid_l[i]

    stop = np.asarray(held, dtype=np.intp)
    obstacle[stop] = d[stop] + s0[stop] - C.STOP_LINE_CLEARANCE
    cap[stop] = d[stop]
    hold_link[stop] = link[stop]
    hold_pos[stop] = net.link_length[link[stop]]
    veh.held[run[stop]] = True
    return Admission(obstacle, cap, hold_link, hold_pos, commits, forced, red_runs, sneaked, seq)


def grant_zones(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    run: IntArray,
    leaders: Leaders,
    index: JunctionIndex,
    *,
    dt: float,
) -> Grants:
    """F.3 zone locks of the committed vehicles (see the module docstring).

    Run after :func:`admit` with the leaders recomputed on its commits. Writes
    ``granted``, ``forced``, ``lock_conn`` and ``held``.
    """
    n, n_lanes = run.size, net.n_lanes
    obstacle = np.full(n, np.inf)
    cap = np.full(n, np.inf)
    hold_link = np.full(n, -1, dtype=np.intp)
    hold_pos = np.full(n, np.nan)
    committed = veh.committed[run]
    todo = np.flatnonzero(committed)
    if todo.size == 0:
        return Grants(obstacle, cap, hold_link, hold_pos, 0, 0, 0, 0)
    todo = todo[np.lexsort((veh.uid[run][todo], veh.commit_seq[run][todo]))]
    link = veh.link[run].astype(np.intp)
    conn = np.where(link < n_lanes, veh.next_conn[run], link)
    lead = leaders.leader
    lead_k = np.where(lead >= 0, np.searchsorted(run, lead), -1)
    ti = veh.type_idx[run]
    cruise = _cruise(net, veh, types, run, conn)

    # plain lists for the sequential loop (F.3 complexity note). ponytail: a Python loop over
    # the committed vehicles; engine/accel/_numba.py::grant with this signature if needed
    run_l: list[int] = run.tolist()
    link_l: list[int] = link.tolist()
    conn_l: list[int] = conn.tolist()
    lead_l: list[int] = lead_k.tolist()
    pos_l: list[float] = veh.pos[run].tolist()
    v_l: list[float] = veh.speed[run].tolist()
    len_l: list[float] = veh.length[run].astype(np.float64).tolist()
    a_l, b_l = types.accel[ti].tolist(), types.decel[ti].tolist()
    be_l, s0_l = types.emergency_decel[ti].tolist(), types.min_gap[ti].tolist()
    vc_l: list[float] = cruise.tolist()
    committed_l: list[bool] = committed.tolist()
    granted_l: list[bool] = veh.granted[run].tolist()
    forced_l: list[bool] = veh.forced[run].tolist()
    seq_l: list[int] = veh.commit_seq[run].tolist()
    lock_l: list[int] = veh.lock_conn[run].tolist()
    link_len: list[float] = net.link_length.tolist()
    to_l: list[int] = net.conn_to_lane.tolist()
    from_l: list[int] = net.conn_from_lane.tolist()
    tau, late = C.GAP_ACCEPT_MARGIN, C.ETA_END_ACCEL_FACTOR
    owners: dict[int, list[int]] = {}  # connector -> positions holding its lock
    for k, c in enumerate(lock_l):
        if c >= 0:
            owners.setdefault(c, []).append(k)
    pending: dict[int, list[int]] = {}  # connector -> committed positions planning it
    for k in todo.tolist():
        pending.setdefault(conn_l[k], []).append(k)
    revoked: list[int] = []
    waiting: list[int] = []
    counts = {"granted": 0, "forced": 0, "conflicts": 0}

    def front_on(k: int, c: int) -> float:
        """Front of ``k`` in the coordinates of connector ``c`` (inf: not on its path)."""
        lk, ci = link_l[k], c - n_lanes
        if lk == c:
            return pos_l[k]
        if lk == from_l[ci]:
            return pos_l[k] - link_len[lk]
        if lk == to_l[ci]:
            return pos_l[k] + link_len[c]
        return math.inf

    def remaining(k: int, c: int) -> list[Conflict]:
        rear = front_on(k, c) - len_l[k]
        return [z for z in index.conflicts[c - n_lanes] if z[2] > rear]

    def approach(k: int) -> int:
        """The lane ``k`` comes from (its lane, or its connector's ``from_lane``)."""
        lk = link_l[k]
        return lk if lk < n_lanes else from_l[lk - n_lanes]

    def overlapping(k: int, zones: list[Conflict]) -> list[int]:
        """Rule (a): other owners of a lock on one of ``zones``."""
        out: list[int] = []
        for foe, _, _, _, fz_out in zones:
            for v in owners.get(foe, ()):
                if v != k and v not in out and front_on(v, foe) - len_l[v] < fz_out:
                    out.append(v)
        return out

    def yields(k: int, c: int, zones: list[Conflict]) -> bool:
        """Rule (b): an earlier-committed ungranted foe arrives before ``k`` clears."""
        front = front_on(k, c)
        for foe, _, z_out, fz_in, fz_out in zones:
            end = eta(z_out + len_l[k] - front, v_l[k], a_l[k] * late, vc_l[k]) + tau
            for j in pending.get(foe, ()):
                if granted_l[j] or seq_l[j] >= seq_l[k]:
                    continue
                fj = front_on(j, foe)
                if fj - len_l[j] >= fz_out:
                    continue
                vj = v_l[j]
                start = -math.inf if fj >= fz_in else eta(fz_in - fj, vj, a_l[j], max(vj, vc_l[j]))
                if start - tau < end:
                    return True
        return False

    def leader_waits(k: int) -> bool:
        """Rule (c): the leader from the same approach lane is committed but ungranted."""
        j = lead_l[k]
        return j >= 0 and committed_l[j] and not granted_l[j] and approach(j) == approach(k)

    def revocable(v: int) -> bool:
        cv = lock_l[v]
        zones = remaining(v, cv)
        if forced_l[v] or not zones:
            return False
        dist = zones[0][1] - front_on(v, cv)
        return dist > 0 and v_l[v] ** 2 <= 2 * dist * (be_l[v] + C.BALLISTIC_FLOOR)

    def acquire(k: int, c: int, zones: list[Conflict], forcing: bool) -> None:
        if lock_l[k] not in (-1, c):
            return  # the previous connector's lock is still held: take this one later
        if forcing:
            for v in overlapping(k, zones):
                if revocable(v):
                    owners[lock_l[v]].remove(v)
                    lock_l[v], granted_l[v] = -1, False
                    veh.lock_conn[run_l[v]], veh.granted[run_l[v]] = -1, False
                    revoked.append(v)
                else:
                    counts["conflicts"] += 1
        if lock_l[k] != c:
            lock_l[k] = c
            owners.setdefault(c, []).append(k)
            veh.lock_conn[run_l[k]] = c

    def grant(k: int, forcing: bool) -> None:
        granted_l[k] = True
        veh.granted[run_l[k]] = True
        counts["granted"] += 1
        if forcing:
            forced_l[k] = True
            veh.forced[run_l[k]] = True
            counts["forced"] += 1
        zones = remaining(k, conn_l[k])
        if zones:
            acquire(k, conn_l[k], zones, forcing)

    def force(k: int) -> None:
        grant(k, True)
        lane, j, hops = approach(k), lead_l[k], 0
        while j >= 0 and committed_l[j] and approach(j) == lane and hops < n:  # cascade
            if not granted_l[j]:
                grant(j, True)
            j, hops = lead_l[j], hops + 1

    for k in todo.tolist():
        if k in revoked:
            continue  # requests again next step
        c = conn_l[k]
        zones = remaining(k, c)
        if granted_l[k]:
            if forced_l[k] and lock_l[k] == -1 and zones:  # made while the old lock was held
                acquire(k, c, zones, True)
            continue
        if not zones:
            grant(k, False)
            continue
        if forced_l[k]:
            force(k)
            continue
        vk = v_l[k]
        dist = zones[0][1] - front_on(k, c)
        reach = vk * vk / (2 * b_l[k]) + vk * dt + a_l[k] * dt * dt / 2 + C.DECISION_MARGIN
        if dist > reach:
            continue  # no request yet
        if (
            lock_l[k] in (-1, c)
            and not overlapping(k, zones)
            and not yields(k, c, zones)
            and not leader_waits(k)
        ):
            grant(k, False)
        elif vk * vk > 2 * dist * (be_l[k] + C.BALLISTIC_FLOOR):
            force(k)  # cannot stop before the zone any more
        else:
            waiting.append(k)

    for k in waiting + revoked:
        if granted_l[k]:
            continue
        c = conn_l[k]
        z_in = remaining(k, c)[0][1]
        dist = z_in - front_on(k, c)
        obstacle[k] = dist + s0_l[k] - C.STOP_LINE_CLEARANCE
        cap[k] = dist
        hold_link[k], hold_pos[k] = c, z_in
        veh.held[run_l[k]] = True
    return Grants(
        obstacle,
        cap,
        hold_link,
        hold_pos,
        counts["granted"],
        counts["forced"],
        len(revoked),
        counts["conflicts"],
    )


def release_locks(
    net: CompiledNetwork, veh: VehicleTable, index: JunctionIndex, handles: IntArray
) -> None:
    """Release the locks of ``handles`` whose rear has passed the last ``z_out`` of
    ``lock_conn`` (or that left its path); ``forced`` goes with the lock unless the
    vehicle is committed again."""
    h = handles[veh.lock_conn[handles] >= 0]
    c = veh.lock_conn[h].astype(np.intp)
    ci = c - net.n_lanes
    link = veh.link[h]
    shift = np.select(
        [link == c, link == net.conn_to_lane[ci], link == net.conn_from_lane[ci]],
        [0.0, net.link_length[c], -net.link_length[net.conn_from_lane[ci]]],
        np.inf,
    )
    done = h[veh.pos[h] + shift - veh.length[h] >= index.lock_end[ci]]
    veh.lock_conn[done] = -1
    veh.forced[done] &= veh.committed[done]


def signal_lookahead(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    run: IntArray,
    stop_gap: FloatArray,
    routes: RouteTable,
    index: JunctionIndex,
    movement_state: StateArray,
) -> FloatArray:
    """F.1 step 4: IDM obstacle gaps at the stop lines ending the lookahead (``inf`` if none).

    For a connector vehicle, or a committed lane vehicle looking through its planned
    connector, :attr:`Leaders.stop_gap` ``d`` is the distance to the stop line at the end of
    the connector's ``to_lane`` (always a G.2 cap obstacle). It is also an IDM obstacle,
    with gap ``d + s0 - STOP_LINE_CLEARANCE``, when the movement from ``route[cursor+1]``
    to ``route[cursor+2]`` is y and ``v^2/(2d) <= YELLOW_MAX_DECEL``, or r and ``v^2/(2d)
    <= b_emerg`` (B.2 #25: the test of F.3 step 1, so nobody brakes for a yellow it will
    drive through).
    """
    out = np.full(run.size, np.inf)
    idx = np.flatnonzero(np.isfinite(stop_gap))
    h = run[idx]
    link = veh.link[h].astype(np.intp)
    conn = np.where(link >= net.n_lanes, link, veh.next_conn[h])
    to = net.conn_to_lane[conn - net.n_lanes]
    sig = net.int_kind[net.link_intersection[to]] == IntersectionKind.signalized.code
    idx, h, to = idx[sig], h[sig], to[sig]
    if idx.size == 0:
        return out
    # ponytail: a Python loop over the (few) vehicles looking into a signalised lane
    after = [
        int(routes.get(r)[c + 2])
        for r, c in zip(veh.route_id[h].tolist(), veh.route_cursor[h].tolist(), strict=True)
    ]
    roads = net.link_road[to].tolist()
    mov = [index.movement_of[pair] for pair in zip(roads, after, strict=True)]
    light = movement_state[mov]
    v2, d = veh.speed[h] ** 2, stop_gap[idx]
    b_emerg = types.emergency_decel[veh.type_idx[h]] + C.BALLISTIC_FLOOR  # as in admit()
    stop = ((light == SignalState.y.code) & (v2 <= 2 * d * C.YELLOW_MAX_DECEL)) | (
        (light == SignalState.r.code) & (v2 <= 2 * d * b_emerg)
    )
    i = idx[stop]
    out[i] = stop_gap[i] + types.min_gap[veh.type_idx[run[i]]] - C.STOP_LINE_CLEARANCE
    return out
