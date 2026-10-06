"""Leader search (plan F.1 step 4): same link, lookahead, diverge and merge siblings."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from urbanflow.core import constants as C
from urbanflow.engine.leaders import (
    ConnectorConflicts,
    SiblingGroups,
    connector_conflicts,
    expand_csr,
)
from urbanflow.network import CompiledNetwork
from urbanflow.network.conflicts import ConflictKind

from .conftest import World

ROUTE_EW = ("E_in", "W_out")


def _lead(w: World, h: int) -> tuple[int, float, float]:
    lead = w.leaders()
    k = int(np.searchsorted(w.run(), h))
    return int(lead.leader[k]), float(lead.gap[k]), float(lead.leader_speed[k])


def test_expand_csr() -> None:
    ptr = np.array([0, 2, 2, 5])
    which, entry = expand_csr(ptr, np.array([2, 0, 1]))
    assert which.tolist() == [0, 0, 0, 1, 1]
    assert entry.tolist() == [2, 3, 4, 0, 1]


def test_connector_conflicts_match_the_table(junction: World) -> None:
    net = junction.net
    cc: ConnectorConflicts = connector_conflicts(net, (ConflictKind.merging,))
    n_merge = int((net.conf_kind == ConflictKind.merging).sum())
    assert cc.ptr[-1] == 2 * n_merge  # each conflict is listed on both connectors
    assert set(cc.kind.tolist()) == {int(ConflictKind.merging)}
    for c in range(net.n_conn):
        mine = net.n_lanes + c
        for e in range(cc.ptr[c], cc.ptr[c + 1]):
            other = int(cc.other[e])
            k = next(
                k
                for k in range(net.n_conflicts)
                if {int(net.conf_a[k]), int(net.conf_b[k])} == {mine, other}
            )
            own = net.conf_zone_a[k] if net.conf_a[k] == mine else net.conf_zone_b[k]
            assert cc.zone[e].tolist() == own.tolist()
        zones = cc.zone[cc.ptr[c] : cc.ptr[c + 1], 0]
        assert np.all(np.diff(zones) >= 0)  # sorted by z_in


def test_same_link_leader(junction: World) -> None:
    a = junction.place("E_in_0", 100.0, 10.0, route=ROUTE_EW)
    b = junction.place("E_in_0", 150.0, 7.0, route=ROUTE_EW)
    assert _lead(junction, a) == (b, 150.0 - 5.0 - 100.0, 7.0)
    assert _lead(junction, b)[0] == -1  # uncommitted: the lookahead stops at the stop line
    lead = junction.leaders()
    assert lead.order.tolist() == [a, b]
    assert lead.tail[junction.link("E_in_0")] == a
    assert lead.tail[junction.link("W_out_0")] == -1


def test_lookahead_through_the_intersection(junction: World) -> None:
    w = junction
    lane, conn = w.link("E_in_0"), w.link("E_in_0->W_out_0")
    length = w.net.link_length
    i = w.place("E_in_0", float(length[lane]) - 2.0, 10.0, route=ROUTE_EW, committed=True)
    j = w.place("W_out_0", 20.0, 4.0, route=("W_out",))
    assert _lead(w, i) == (j, pytest.approx(2.0 + length[conn] + 15.0), 4.0)
    k = w.place("E_in_0->W_out_0", 8.0, 6.0, route=ROUTE_EW)  # nearer: on the connector
    assert _lead(w, i)[:2] == (k, pytest.approx(2.0 + 3.0))
    assert _lead(w, k)[:2] == (j, pytest.approx(length[conn] - 8.0 + 15.0))


def test_uncommitted_vehicle_stops_at_its_stop_line(junction: World) -> None:
    w = junction
    lane = w.net.link_length[w.link("E_in_0")]
    i = w.place("E_in_0", float(lane) - 2.0, 10.0, route=ROUTE_EW)
    w.place("W_out_0", 20.0, 4.0, route=("W_out",))
    assert _lead(w, i)[0] == -1
    # ...but it sees the tail of its planned connector (the diverge rule includes c_i)
    k = w.place("E_in_0->W_out_0", 8.0, 6.0, route=ROUTE_EW)
    assert _lead(w, i)[:2] == (k, pytest.approx(2.0 + 3.0))


def test_lookahead_distance(junction: World) -> None:
    w = junction
    w.place("W_out_0", 20.0, 4.0, route=("W_out",))
    slow = w.place("E_in_0", 10.0, 0.0, route=ROUTE_EW, committed=True)
    fast = w.place("E_in_0", 120.0, 30.0, route=ROUTE_EW, committed=True)
    lead = w.leaders()
    run = w.run().tolist()
    # the slow one's leader is `fast` on its own link; `fast` looks 272 m ahead
    reach = 30.0**2 / (2 * C.IDM_DECEL) + 30.0 * C.IDM_HEADWAY + C.IDM_MIN_GAP
    assert reach > C.LOOKAHEAD_MIN_DISTANCE
    assert lead.leader[run.index(slow)] == fast
    assert lead.leader[run.index(fast)] >= 0
    w2 = World(w.net)
    w2.place("W_out_0", 20.0, 4.0, route=("W_out",))
    far = w2.place("E_in_0", 10.0, 0.0, route=ROUTE_EW, committed=True)  # d > 200 m, v = 0
    assert _lead(w2, far)[0] == -1


def test_stop_gap_of_connector_and_committed_vehicles(corridor_world: World) -> None:
    w = corridor_world
    net = w.net
    length = net.link_length
    route = ("W_J1", "J1_J2", "J2_E")
    on_conn = w.place("W_J1_0->J1_J2_1", 5.0, 8.0, route=route)
    on_lane = w.place("W_J1_0", float(length[0]) - 3.0, 8.0, route=route, committed=True)
    free = w.place("W_J1_1", 100.0, 8.0, route=route)
    last = w.place("J2_E_0", 10.0, 8.0, route=("J2_E",))
    lead = w.leaders()
    run = w.run().tolist()
    to_len = float(length[w.link("J1_J2_1")])
    conn_len = float(length[w.link("W_J1_0->J1_J2_1")])
    assert lead.stop_gap[run.index(on_conn)] == pytest.approx(conn_len - 5.0 + to_len)
    assert lead.stop_gap[run.index(on_lane)] == pytest.approx(3.0 + conn_len + to_len)
    assert math.isinf(lead.stop_gap[run.index(free)])  # its own line: admission's job
    assert math.isinf(lead.stop_gap[run.index(last)])
    # a connector into the last road has no stop line ahead (the vehicle arrives)
    w2 = World(net)
    h = w2.place("J1_J2_1->J2_E_1", 2.0, 8.0, route=("J1_J2", "J2_E"))
    assert math.isinf(w2.leaders().stop_gap[0]) and w2.run().tolist() == [h]


def test_merge_alignment_by_distance_to_end(junction: World) -> None:
    w = junction
    length = w.net.link_length
    straight, right = "W_in_0->E_out_0", "S_in_0->E_out_0"
    zone = next(
        z for z in w.junctions.conflicts[w.link(straight) - w.net.n_lanes] if z[0] == w.link(right)
    )
    i = w.place(straight, zone[1] + 1.0, 5.0, route=("W_in", "E_out"))  # inside the merge zone
    j = w.place(right, 3.0, 2.0, route=("S_in", "E_out"))
    rem_i = float(length[w.link(straight)]) - float(w.veh.pos[i])
    rem_j = float(length[w.link(right)]) - 3.0
    assert rem_j < rem_i
    assert _lead(w, i) == (j, pytest.approx(rem_i - rem_j - 5.0), 2.0)
    assert _lead(w, j)[0] == -1  # i is farther from the merge point


def test_no_merge_leader_before_the_merge_zone(junction: World) -> None:
    """P4 repro: a left turner waiting at its first crossing zone (0.78 m into a 10.7 m
    connector) and a right turner entering the 5.6 m sibling that merges with it "overlap"
    by distance to the end (9.9 m < 5.1 m + 5 m) although they are metres apart. Before
    its merge zone the paths are more than W_max apart and the zone lock orders them."""
    w = junction
    left, right = "N_in_0->E_out_0", "S_in_0->E_out_0"
    z_in = next(
        z[1] for z in w.junctions.conflicts[w.link(left) - w.net.n_lanes] if z[0] == w.link(right)
    )
    i = w.place(left, 0.78, 0.0, route=("N_in", "E_out"))
    j = w.place(right, 0.57, 2.3, route=("S_in", "E_out"))
    rem_i = float(w.net.link_length[w.link(left)]) - 0.78
    rem_j = float(w.net.link_length[w.link(right)]) - 0.57
    assert z_in > 0.78 and rem_j < rem_i < rem_j + C.VEHICLE_LENGTH  # would overlap
    assert _lead(w, i)[0] == -1
    w.veh.pos[i] = z_in + 0.1  # inside its merge zone: aligned again
    rem_i = float(w.net.link_length[w.link(left)]) - z_in - 0.1
    w.veh.pos[j] = float(w.net.link_length[w.link(right)]) - (rem_i - 2.0)
    assert _lead(w, i)[:2] == (j, pytest.approx(2.0 - C.VEHICLE_LENGTH))


def test_a_rear_hanging_back_over_a_merging_sibling_counts_from_the_merge_zone(
    junction: World,
) -> None:
    """P4 review repro (2x2 grid with trucks, I4 "gap -0.735 m"): a bus that left the
    short right-turn connector still hangs back ~10 m over it. A left turner waiting at its
    first zone 0.78 m into its 10.7 m connector saw that rear as if it lay on its own path
    (gap (L - 0.78) + pos - 12 < 0) although the paths only meet in the merge zone."""
    w = junction
    left, right = w.link("N_in_0->E_out_0"), w.link("S_in_0->E_out_0")
    length = w.net.link_length
    z_in = next(z[1] for z in w.junctions.conflicts[left - w.net.n_lanes] if z[0] == right)
    bus = w.place("E_out_0", 1.6, 2.6, route=("E_out",), vtype="bus")
    w.veh.lock_conn[bus] = right  # came from the right turn, lock still held
    rear = 1.6 - float(w.veh.length[bus])
    assert rear < z_in - float(length[left])  # hangs back past the merge zone's start
    i = w.place("N_in_0->E_out_0", 0.78, 0.0, route=("N_in", "E_out"))
    assert z_in > 0.78 and float(length[left]) - 0.78 + rear < 0  # the old, false overlap
    assert _lead(w, i)[:2] == (bus, pytest.approx(z_in - 0.78))
    # a committed lane vehicle looking through the left turn: the same clamp
    d = 3.0
    lane = w.place("N_in_0", float(length[w.link("N_in_0")]) - d, 8.0, route=("N_in", "E_out"))
    w.veh.committed[lane] = True
    assert _lead(w, lane)[:2] == (i, pytest.approx(d + 0.78 - C.VEHICLE_LENGTH))
    w.veh.pos[i] = z_in + 0.5  # inside the merge zone: the whole rear is on its path
    assert _lead(w, i)[:2] == (bus, pytest.approx(float(length[left]) - z_in - 0.5 + rear))
    w.veh.pos[i] = 0.78
    w.veh.lock_conn[bus] = -1  # unknown origin: conservative, the whole rear
    assert _lead(w, i)[:2] == (bus, pytest.approx(float(length[left]) - 0.78 + rear))
    w.veh.lock_conn[bus] = left  # hangs back over the follower's own connector: whole rear
    assert _lead(w, i)[:2] == (bus, pytest.approx(float(length[left]) - 0.78 + rear))
    w2 = World(w.net)
    bus2 = w2.place("E_out_0", 1.6, 2.6, route=("E_out",), vtype="bus")
    w2.veh.lock_conn[bus2] = right
    lane2 = w2.place("N_in_0", float(length[w.link("N_in_0")]) - d, 8.0, route=("N_in", "E_out"))
    w2.veh.committed[lane2] = True
    gap = d + float(length[left]) + z_in - float(length[left])
    assert _lead(w2, lane2)[:2] == (bus2, pytest.approx(gap))


def test_diverge_zone(junction: World) -> None:
    w = junction
    net = w.net
    a, b = w.link("W_in_0->E_out_0"), w.link("W_in_0->N_out_0")
    k = next(
        k
        for k in range(net.n_conflicts)
        if {int(net.conf_a[k]), int(net.conf_b[k])} == {a, b}
        and net.conf_kind[k] == ConflictKind.diverging
    )
    d_sep = float((net.conf_zone_b if net.conf_b[k] == b else net.conf_zone_a)[k][1])
    lane = float(net.link_length[w.link("W_in_0")])
    i = w.place("W_in_0", lane - 4.0, 8.0, route=("W_in", "E_out"))
    j = w.place("W_in_0->N_out_0", d_sep - 1.0 + 5.0, 3.0, route=("W_in", "N_out"))
    assert _lead(w, i)[:2] == (j, pytest.approx(4.0 + d_sep - 1.0))
    w.veh.pos[j] = d_sep + 1.0 + 5.0  # rear past d_sep: separated
    assert _lead(w, i)[0] == -1
    # connector vehicles: the nearest vehicle ahead on a diverging sibling within d_sep
    w.veh.pos[j] = d_sep - 1.0 + 5.0
    c = w.place("W_in_0->E_out_0", 1.0, 8.0, route=("W_in", "E_out"))
    assert _lead(w, c)[:2] == (j, pytest.approx(d_sep - 1.0 - 1.0))


def test_rear_over_the_lane_end_blocks_any_lane_vehicle(junction: World) -> None:
    w = junction
    lane = float(w.net.link_length[w.link("W_in_0")])
    j = w.place("W_in_0->N_out_0", 2.0, 1.0, route=("W_in", "N_out"))  # rear at -3
    i = w.place("W_in_0", lane - 10.0, 8.0, route=("W_in",))  # on its last road, no plan
    assert _lead(w, i)[:2] == (j, pytest.approx(10.0 - 3.0))


def test_lookahead_sees_a_rear_hanging_back_over_an_empty_lane(two_junctions: World) -> None:
    """Review repro: a bus longer than its connector keeps its rear on the lane before it,
    so a vehicle looking into that lane must not see it as free up to its end."""
    w = two_junctions
    net, length = w.net, w.net.link_length
    lane, conn = w.link("J1_J2_0"), w.link("J1_J2_0->J2_S_0")
    into = w.link("W_J1_0->J1_J2_0")
    assert length[conn] < 12.0  # shorter than a bus
    bus_at = float(length[conn]) - 1.0
    rear_on_lane = float(length[lane]) + bus_at - 12.0
    route = ("W_J1", "J1_J2", "J2_E")
    bus = w.place("J1_J2_0->J2_S_0", bus_at, 0.0, route=("J1_J2", "J2_S"), vtype="bus")
    car = w.place("W_J1_0->J1_J2_0", 2.0, 13.0, route=route)  # looks into J1_J2_0
    assert _lead(w, car)[:2] == (bus, pytest.approx(float(length[into]) - 2.0 + rear_on_lane))
    assert w.leaders().lane_rear[lane] == pytest.approx(rear_on_lane)
    # a committed lane vehicle looking through its planned connector
    w2 = World(net)
    bus2 = w2.place("J1_J2_0->J2_S_0", bus_at, 0.0, route=("J1_J2", "J2_S"), vtype="bus")
    d = 3.0
    lane_car = w2.place("W_J1_0", float(length[0]) - d, 13.0, route=route, committed=True)
    assert w2.veh.next_conn[lane_car] == into
    gap = d + float(length[into]) + rear_on_lane
    assert _lead(w2, lane_car)[:2] == (bus2, pytest.approx(gap))


def test_no_running_vehicles(junction: World) -> None:
    lead = junction.leaders()
    assert lead.order.size == lead.leader.size == 0
    assert (lead.tail == -1).all()


# --------------------------------------------------------------------------- reference
def _reference(w: World) -> dict[int, tuple[int, float, float]]:
    """O(N^2) leader search written from the F.1 step 4 text: {handle: (leader, gap, stop)}."""
    net, veh = w.net, w.veh
    n_lanes, length = net.n_lanes, net.link_length
    diverge: dict[tuple[int, int], float] = {}
    merge: dict[tuple[int, int], float] = {}  # (a, b) -> start of the merge zone on a
    for k in range(net.n_conflicts):
        a, b = int(net.conf_a[k]), int(net.conf_b[k])
        if net.conf_kind[k] == ConflictKind.diverging:
            diverge[a, b] = float(net.conf_zone_b[k][1])
            diverge[b, a] = float(net.conf_zone_a[k][1])
        elif net.conf_kind[k] == ConflictKind.merging:
            merge[a, b] = float(net.conf_zone_a[k][0])
            merge[b, a] = float(net.conf_zone_b[k][0])
    run = w.run().tolist()
    remaining = dict(zip(run, w.remaining().tolist(), strict=True))
    out: dict[int, tuple[int, float, float]] = {}
    for i in run:
        li, pi, vi = int(veh.link[i]), float(veh.pos[i]), float(veh.speed[i])
        t = int(veh.type_idx[i])
        types = w.types
        reach = max(
            C.LOOKAHEAD_MIN_DISTANCE,
            vi**2 / (2 * types.decel[t]) + vi * types.headway[t] + types.min_gap[t],
        )
        d_end = float(length[li]) - pi
        nc = int(veh.next_conn[i])
        on_lane = li < n_lanes
        # (link, offset, connector leading into it or -1, i's front on that connector)
        path: list[tuple[int, float, int, float]] = []
        stop = math.inf
        if not on_lane:
            to = int(net.conn_to_lane[li - n_lanes])
            path = [(to, d_end, li, pi)]
            if remaining[i] >= 2:
                stop = d_end + float(length[to])
        elif veh.committed[i] and nc >= 0:
            to = int(net.conn_to_lane[nc - n_lanes])
            path = [(nc, d_end, -1, 0.0), (to, d_end + float(length[nc]), nc, -d_end)]
            if remaining[i] >= 2:
                stop = d_end + float(length[nc] + length[to])
        ptr = net.lane_out_ptr
        out_conns = set(net.lane_out_conn[ptr[li] : ptr[li + 1]].tolist()) if on_lane else set()
        best: tuple[float, int] | None = None
        for j in run:
            if j == i:
                continue
            lj, pj, lenj = int(veh.link[j]), float(veh.pos[j]), float(veh.length[j])
            rj = pj - lenj
            gaps: list[float] = []
            if lj == li and (pj, int(veh.uid[j])) > (pi, int(veh.uid[i])):
                gaps.append(rj - pi)
            for lk, off, via, front in path:
                if lj != lk or off >= reach:
                    continue
                start = merge.get((via, int(veh.lock_conn[j])))
                if rj < 0 and start is not None and front <= start:
                    # hangs back over a merging sibling: on via's path only in the zone
                    gaps.append(off + max(rj, start - float(length[via])))
                else:
                    gaps.append(off + rj)
            if lj >= n_lanes and rj < 0:  # rear hanging back over its from-lane's end
                back = int(net.conn_from_lane[lj - n_lanes])
                gaps += [
                    off + float(length[lk]) + rj
                    for lk, off, _, _ in path
                    if lk == back and off < reach
                ]
            if on_lane and lj in out_conns:
                limit = math.inf if lj == nc else diverge.get((nc, lj), 0.0)
                if rj <= limit:
                    gaps.append(d_end + rj)
            if not on_lane and (li, lj) in diverge and pj > pi and rj <= diverge[li, lj]:
                gaps.append(rj - pi)
            if (
                not on_lane
                and (li, lj) in merge
                and float(length[lj]) - pj < d_end
                and pi > merge[li, lj]  # i's front inside its merge zone
            ):
                gaps.append(d_end - (float(length[lj]) - pj) - lenj)
            for g in gaps:
                if best is None or (g, int(veh.uid[j])) < (best[0], int(veh.uid[best[1]])):
                    best = (g, j)
        out[i] = (-1, math.inf, stop) if best is None else (best[1], best[0], stop)
    return out


@st.composite
def _traffic(draw: st.DrawFn, net: CompiledNetwork) -> World:
    w = World(net)
    successors: dict[int, list[int]] = {}
    for m in range(net.n_movements):
        successors.setdefault(int(net.mov_from_road[m]), []).append(int(net.mov_to_road[m]))
    links = draw(st.lists(st.integers(0, net.n_links - 1), min_size=1, max_size=8, unique=True))
    for lk in links:
        on_lane = lk < net.n_lanes
        if on_lane:
            roads = [int(net.link_road[lk])]
        else:
            mov = int(net.link_movement[lk])
            roads = [int(net.mov_from_road[mov]), int(net.mov_to_road[mov])]
        while successors.get(roads[-1]) and draw(st.booleans()):
            roads.append(draw(st.sampled_from(successors[roads[-1]])))
        route = [net.road_ids[r] for r in roads]
        front = -1.0
        for _ in range(draw(st.integers(1, 4))):
            vtype = draw(st.sampled_from(["car", "bus"]))
            if front < 0:
                front = draw(st.integers(0, 8)) * 0.5
            else:  # upstream first: this vehicle's rear stays ahead of the previous front
                front += float(w.types.length[w.types.index[vtype]]) + draw(st.integers(0, 60)) / 2
            if front > net.link_length[lk]:
                break
            h = w.place(
                net.link_ids[lk], front, draw(st.integers(0, 30)) * 1.0, route=route, vtype=vtype
            )
            if on_lane and w.veh.next_conn[h] >= 0:
                w.veh.committed[h] = draw(st.booleans())
            into = (np.flatnonzero(net.conn_to_lane == lk) + net.n_lanes).tolist()
            if on_lane and into and draw(st.booleans()):  # still holds the lock it came by
                w.veh.lock_conn[h] = draw(st.sampled_from(into))
    return w


def _check_against_reference(w: World) -> None:
    net, veh, length = w.net, w.veh, w.net.link_length
    run = w.run().tolist()
    merging = SiblingGroups.build(net).merging
    for i in run:  # merge comparisons across links: no near-ties
        li = int(veh.link[i])
        if li < net.n_lanes:
            continue
        for e in range(merging.ptr[li - net.n_lanes], merging.ptr[li - net.n_lanes + 1]):
            other = int(merging.other[e])
            for j in run:
                if veh.link[j] == other:
                    gap = (length[other] - veh.pos[j]) - (length[li] - veh.pos[i])
                    assume(abs(gap) > 1e-6)
    lead = w.leaders()
    ref = _reference(w)
    for k, h in enumerate(run):
        leader, gap, stop = ref[h]
        assert lead.leader[k] == leader, (h, ref)
        assert lead.gap[k] == pytest.approx(gap, abs=1e-9)
        assert lead.stop_gap[k] == pytest.approx(stop)
        assert lead.leader_speed[k] == (float(veh.speed[leader]) if leader >= 0 else 0.0)
        assert lead.order[k] == run[int(lead.perm[k])]


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_junction_matches_the_quadratic_reference(
    junction_net: CompiledNetwork, data: st.DataObject
) -> None:
    _check_against_reference(data.draw(_traffic(junction_net)))


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_corridor_matches_the_quadratic_reference(
    corridor_net: CompiledNetwork, data: st.DataObject
) -> None:
    _check_against_reference(data.draw(_traffic(corridor_net)))
