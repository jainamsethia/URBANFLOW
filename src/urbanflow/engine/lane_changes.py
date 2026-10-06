"""F.1 sub-step 5: lane changes with MOBIL plus route-mandatory changes (plan G.4).

Candidates (vectorised): running vehicles on a lane, not committed to the stop line, off
cooldown and not dwelling. A change is *mandatory* when the current lane has no connector
to the next road of the route (``next_conn == -1`` before the last road); otherwise it is
*discretionary* and only allowed farther than ``LC_NO_DISCRETIONARY_ZONE`` from the stop
line.

For each adjacent lane ``l'`` the vehicle keeps its relative position (``pos' = pos L'/L``)
and is checked against its would-be leader and follower on ``l'`` (lane vehicles, connector
vehicles about to enter ``l'`` and connector rears still overhanging ``l'``):

* **safety**: both new gaps positive, the space reserved by committed inbound vehicles left
  free, no cut-in ahead of a committed follower, the G.2 stopping inequality holding for
  the vehicle behind its new leader and for the new follower behind it, and the new
  follower's IDM deceleration within ``lc_safe_decel`` (MOBIL safety);
* **route**: mandatory changes only move toward a lane that serves the next road;
  discretionary changes never leave such a lane for one that does not;
* **incentive** (Kesting, Treiber & Helbing 2007): ``a~_i - a_i + p[(a~_n - a_n) +
  (a~_o - a_o)] > a_th + beta`` with ``beta = -beta_m`` toward a valid lane for mandatory
  changes, ``beta_m = min(10, bias D_m / max(d, 1))`` and ``p = 0`` for mandatory changes
  closer than 50 m.

Accepted changes run sequentially in (mandatory first, incentive desc, uid) order, each
re-validated against vehicles already moved into the same lane this step. A change is
instantaneous for the longitudinal model; ``lat_offset`` decays for rendering only. There
are no shadow vehicles, so lane counts stay exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.events import EventBuffer, EventType
from urbanflow.core.types import FloatArray, IntArray
from urbanflow.routing.lanes import plan_connector
from urbanflow.vehicles.car_following import CarFollowingInputs, desired_speed, stop_budget

if TYPE_CHECKING:
    from urbanflow.engine.leaders import Leaders
    from urbanflow.network.compiled import CompiledNetwork
    from urbanflow.routing.base import RouteTable
    from urbanflow.vehicles.car_following import CarFollowingModel
    from urbanflow.vehicles.table import VehicleTable
    from urbanflow.vehicles.types import VehicleTypes

__all__ = ["LaneChanges", "lane_change_step"]


@dataclass(frozen=True, slots=True)
class LaneChanges:
    """Handles that changed lane this step and their previous lanes."""

    handles: IntArray
    from_links: IntArray


def _idm(
    model: CarFollowingModel,
    types: VehicleTypes,
    veh: VehicleTable,
    h: IntArray,
    gap: FloatArray,
    leader_speed: FloatArray,
    v0: FloatArray,
    dt: float,
    rng: np.random.Generator,
) -> FloatArray:
    p = types.params.gather(veh.type_idx[h].astype(np.intp))
    x = CarFollowingInputs(veh.speed[h], v0, gap, leader_speed, dt, veh.uid[h], rng)
    return model.acceleration(x, p)


def lane_change_step(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    model: CarFollowingModel,
    routes: RouteTable,
    run: IntArray,
    leaders: Leaders,
    reserved: FloatArray,
    events: EventBuffer,
    *,
    dt: float,
    step: int,
    time: float,
    rng: np.random.Generator,
) -> LaneChanges:
    """Evaluate and execute lane changes for ``run`` (see the module docstring)."""
    none = LaneChanges(np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp))
    n_lanes = net.n_lanes
    link = veh.link[run].astype(np.intp)
    on_lane = link < n_lanes
    if not on_lane.any():
        return none
    lane_len = net.link_length[:n_lanes]
    pos = veh.pos[run]
    d = np.where(on_lane, lane_len[np.minimum(link, n_lanes - 1)] - pos, 0.0)
    cursor = veh.route_cursor[run].astype(np.intp)
    last = cursor + 1 >= routes.lengths[veh.route_id[run]]
    mandatory = on_lane & (veh.next_conn[run] < 0) & ~last
    cand = (
        on_lane
        & ~veh.committed[run]
        & (veh.lc_cooldown[run] <= 0)
        & (veh.dwell_left[run] <= 0)
        & (mandatory | (d > C.LC_NO_DISCRETIONARY_ZONE))
    )
    if not cand.any():
        return none

    # sorted lane occupancy: key = link * K + pos (uid breaks ties)
    lane_run = run[on_lane]
    keys_all = link[on_lane] * C.LC_SORT_KEY_SCALE + pos[on_lane]
    order = np.lexsort((veh.uid[lane_run], keys_all))
    s_h = lane_run[order]
    s_key = keys_all[order]
    s_link = veh.link[s_h].astype(np.intp)

    # connector vehicles: the nearest one about to enter each lane, and rears still
    # overhanging the lane they left
    conn_run = run[~on_lane]
    enter_front = np.full(n_lanes, -np.inf)
    enter_h = np.full(n_lanes, -1, dtype=np.intp)
    overhang = lane_len.copy()
    if conn_run.size:
        k = veh.link[conn_run].astype(np.intp) - n_lanes
        to = net.conn_to_lane[k].astype(np.intp)
        frm = net.conn_from_lane[k].astype(np.intp)
        front = veh.pos[conn_run] - net.link_length[n_lanes + k]  # <= 0 on the to-lane axis
        by = np.lexsort((veh.uid[conn_run], front, to))  # per to-lane, the nearest last
        last = np.r_[to[by][1:] != to[by][:-1], True]
        enter_front[to[by][last]] = front[by][last]
        enter_h[to[by][last]] = conn_run[by][last]
        rear = lane_len[frm] + veh.pos[conn_run] - veh.length[conn_run]
        np.minimum.at(overhang, frm, rear)

    ci = np.flatnonzero(cand)
    h_c = run[ci]
    l_c = link[ci]
    pos_c = pos[ci]
    len_c = veh.length[h_c].astype(np.float64)
    v_c = veh.speed[h_c]
    d_c = d[ci]
    mand_c = mandatory[ci]
    lane_index = net.link_lane_index[l_c].astype(np.int64)
    valid = veh.valid_mask[h_c].astype(np.int64)

    # current situation of the candidates (from the step-4 leaders)
    v0_c = veh.v0[h_c]
    a_now = _idm(model, types, veh, h_c, leaders.gap[ci], leaders.leader_speed[ci], v0_c, dt, rng)
    # old follower: the vehicle right behind on the same lane
    idx_self = np.searchsorted(s_key, l_c * C.LC_SORT_KEY_SCALE + pos_c, side="left")
    has_old = (idx_self > 0) & (s_link[np.maximum(idx_self - 1, 0)] == l_c)
    o_h = np.where(has_old, s_h[np.maximum(idx_self - 1, 0)], 0)
    gap_oi = np.where(has_old, pos_c - len_c - veh.pos[o_h], np.inf)

    best_gain = np.full(ci.size, -np.inf)
    best_target = np.full(ci.size, -1, dtype=np.intp)
    best_pos = np.zeros(ci.size)
    best_lead = np.full(ci.size, -1, dtype=np.intp)
    best_fol = np.full(ci.size, -1, dtype=np.intp)
    b_hat = types.b_hat
    ti_c = veh.type_idx[h_c].astype(np.intp)

    for side in (net.lane_left, net.lane_right):
        target = side[l_c].astype(np.intp)
        ok = target >= 0
        if not ok.any():
            continue
        tgt = np.maximum(target, 0)
        tgt_index = net.link_lane_index[tgt].astype(np.int64)
        tgt_valid = ((valid >> tgt_index) & 1).astype(bool)
        cur_valid = ((valid >> lane_index) & 1).astype(bool)
        # mandatory: move toward a valid lane only; discretionary: never leave a valid lane
        # for an invalid one close to the junction
        toward = np.zeros(ci.size, dtype=bool)
        if mand_c.any():
            n_road = net.road_n_lanes[net.link_road[l_c]].astype(np.int64)
            for j in np.flatnonzero(mand_c & ok):
                bits = [b for b in range(int(n_road[j])) if (valid[j] >> b) & 1]
                if bits:
                    here = int(lane_index[j])
                    nearest = min(bits, key=lambda b: (abs(b - here), b))
                    toward[j] = abs(nearest - int(tgt_index[j])) < abs(nearest - here)
        # discretionary changes never leave a lane that serves the route (B.2 #25: G.4's
        # "allowed beyond D_m" left vehicles stuck changing back into standing queues)
        ok &= np.where(mand_c, toward, tgt_valid | ~cur_valid)
        if not ok.any():
            continue
        pos_t = pos_c * lane_len[tgt] / lane_len[l_c]
        # new leader / follower on the target lane
        q = np.searchsorted(s_key, tgt * C.LC_SORT_KEY_SCALE + pos_t, side="right")
        has_lead = (q < s_h.size) & (s_link[np.minimum(q, s_h.size - 1)] == tgt)
        lead_h = np.where(has_lead, s_h[np.minimum(q, s_h.size - 1)], 0)
        gap_l = np.where(
            has_lead,
            veh.pos[lead_h] - veh.length[lead_h] - pos_t,
            overhang[tgt] - pos_t,  # a connector rear still on the lane, else the lane end
        )
        v_l = np.where(has_lead, veh.speed[lead_h], 0.0)
        no_lead_on_lane = ~has_lead & (overhang[tgt] >= lane_len[tgt])
        gap_l = np.where(no_lead_on_lane, np.inf, gap_l)
        has_fol = (q > 0) & (s_link[np.maximum(q - 1, 0)] == tgt)
        fol_h = np.where(has_fol, s_h[np.maximum(q - 1, 0)], enter_h[tgt])
        fol_front = np.where(has_fol, veh.pos[fol_h], enter_front[tgt])
        has_fol |= enter_h[tgt] >= 0
        fol_h = np.maximum(fol_h, 0)
        gap_f = np.where(has_fol, pos_t - len_c - fol_front, np.inf)
        ok &= (gap_l > 0) & (gap_f > 0) & (pos_t - len_c >= reserved[tgt])
        ok &= ~(has_fol & veh.committed[fol_h] & (veh.link[fol_h] == tgt))
        # G.2 feasibility (no collision even under emergency braking)
        b_l = types.emergency_decel[veh.type_idx[lead_h]]
        budget_i = np.where(
            np.isfinite(gap_l),
            stop_budget(np.where(np.isfinite(gap_l), gap_l, 0.0), v_l, b_l),
            np.inf,
        )
        ok &= v_c**2 / (2 * b_hat[veh.type_idx[h_c]]) <= budget_i
        v_f = veh.speed[fol_h]
        b_i = types.emergency_decel[veh.type_idx[h_c]]
        budget_f = stop_budget(np.where(has_fol, gap_f, 0.0), v_c, b_i)
        ok &= ~has_fol | (v_f**2 / (2 * b_hat[veh.type_idx[fol_h]]) <= budget_f)
        if not ok.any():
            continue
        # accelerations before and after (MOBIL)
        v0_t = desired_speed(net, veh, types, h_c, tgt)
        a_new = _idm(model, types, veh, h_c, gap_l, v_l, v0_t, dt, rng)
        f_gap_now = np.where(
            has_lead & has_fol, veh.pos[lead_h] - veh.length[lead_h] - fol_front, np.inf
        )
        a_f_now = _idm(model, types, veh, fol_h, f_gap_now, v_l, veh.v0[fol_h], dt, rng)
        a_f_new = _idm(model, types, veh, fol_h, gap_f, v_c, veh.v0[fol_h], dt, rng)
        safe_decel = types.lc_safe_decel[veh.type_idx[fol_h]]
        ok &= ~has_fol | (a_f_new >= -safe_decel)
        a_o_now = _idm(model, types, veh, o_h, gap_oi, v_c, veh.v0[o_h], dt, rng)
        gap_o_after = gap_oi + len_c + leaders.gap[ci]
        a_o_new = _idm(
            model, types, veh, o_h, gap_o_after, leaders.leader_speed[ci], veh.v0[o_h], dt, rng
        )
        politeness = np.where(
            mand_c & (d_c < C.LC_POLITENESS_OFF_DISTANCE), 0.0, types.politeness[ti_c]
        )
        others = np.where(has_fol, a_f_new - a_f_now, 0.0) + np.where(
            has_old, a_o_new - a_o_now, 0.0
        )
        gain = a_new - a_now + politeness * others
        beta_m = np.minimum(
            C.LC_MANDATORY_BIAS_MAX,
            C.LC_MANDATORY_BIAS
            * C.LC_MANDATORY_DISTANCE
            / np.maximum(d_c, C.LC_URGENCY_MIN_DISTANCE),
        )
        margin = gain - types.lc_threshold[ti_c] + np.where(mand_c, beta_m, 0.0)
        better = ok & (margin > 0) & (margin > best_gain)
        best_gain = np.where(better, margin, best_gain)
        best_target = np.where(better, tgt, best_target)
        best_pos = np.where(better, pos_t, best_pos)
        best_lead = np.where(better, np.where(has_lead, lead_h, -1), best_lead)
        best_fol = np.where(better, np.where(has_fol, fol_h, -1), best_fol)

    chosen = np.flatnonzero(best_target >= 0)
    if not chosen.size:
        return none
    # sequential execution: mandatory first, then by incentive, then uid
    seq = chosen[np.lexsort((veh.uid[h_c[chosen]], -best_gain[chosen], ~mand_c[chosen]))]
    # neighbours of an executed change re-decide next step (simultaneous decisions would
    # otherwise move a follower into the gap its leader just took)
    cur_lead = leaders.leader[ci]
    old_fol = np.where(has_old, o_h, -1)
    changed: set[int] = set()
    moved: dict[int, list[tuple[float, float]]] = {}
    done_h: list[int] = []
    done_from: list[int] = []
    s0 = types.min_gap
    for j in seq.tolist():
        h = int(h_c[j])
        tgt_j = int(best_target[j])
        near = {int(cur_lead[j]), int(old_fol[j]), int(best_lead[j]), int(best_fol[j])}
        if near & changed:
            continue
        front_j = float(best_pos[j])
        rear_j = front_j - float(len_c[j])
        gap_needed = float(s0[veh.type_idx[h]])
        if any(
            rear_j < f + gap_needed and r < front_j + gap_needed for f, r in moved.get(tgt_j, [])
        ):
            continue  # conflicts with a vehicle that moved into this lane this step
        moved.setdefault(tgt_j, []).append((front_j, rear_j))
        old = int(veh.link[h])
        veh.link[h] = tgt_j
        veh.pos[h] = front_j
        width = float(net.link_width[tgt_j])
        toward_median = net.link_lane_index[tgt_j] < net.link_lane_index[old]
        veh.lat_offset[h] = -width if toward_median else width  # drawn at the old lane first
        veh.lc_cooldown[h] = C.LC_COOLDOWN
        veh.lane_changes[h] += 1
        route = routes.get(int(veh.route_id[h]))
        veh.next_conn[h] = plan_connector(net, tgt_j, route, int(veh.route_cursor[h]))
        veh.v0[h] = desired_speed(net, veh, types, np.array([h]), np.array([tgt_j]))[0]
        done_h.append(h)
        changed.add(h)
        done_from.append(old)
    if not done_h:
        return none
    hs = np.asarray(done_h, dtype=np.intp)
    froms = np.asarray(done_from, dtype=np.intp)
    events.append_many(
        EventType.vehicle_changed_lane,
        step,
        time,
        handles=hs,
        uids=veh.uid[hs],
        links=veh.link[hs],
        aux=froms,
    )
    return LaneChanges(hs, froms)


def decay_lane_change_state(
    veh: VehicleTable, run: IntArray, net: CompiledNetwork, dt: float
) -> None:
    """Cooldowns tick down; the visual lateral offset decays to 0 over LC_VISUAL_DURATION."""
    if not run.size:
        return
    cd = veh.lc_cooldown[run]
    veh.lc_cooldown[run] = np.maximum(0.0, cd - dt)
    lat = veh.lat_offset[run]
    moving = lat != 0
    if moving.any():
        width = net.link_width[veh.link[run]].astype(np.float32)
        step = width / np.float32(C.LC_VISUAL_DURATION) * np.float32(dt)
        veh.lat_offset[run] = np.where(
            lat > 0, np.maximum(0.0, lat - step), np.minimum(0.0, lat + step)
        ).astype(np.float32)
