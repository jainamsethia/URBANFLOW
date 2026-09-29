"""Admission at unsignalised stop lines (plan F.3) on synthetic states."""

from __future__ import annotations

import math

import pytest

from urbanflow.core import constants as C
from urbanflow.engine.intersections import eta, reservations

from .conftest import World

WE = ("W_in", "E_out")  # W -> E straight
NS = ("N_in", "S_out")  # N -> S straight, crosses W -> E


def _lane_end(w: World, lane: str) -> float:
    return float(w.net.link_length[w.link(lane)])


def test_eta_formula() -> None:
    assert eta(0.0, 5.0, 1.5, 10.0) == 0.0
    assert eta(-3.0, 5.0, 1.5, 10.0) == 0.0
    assert eta(20.0, 10.0, 1.5, 8.0) == pytest.approx(20.0 / 8.0)  # already above v_c
    assert eta(3.0, 0.0, 1.5, 10.0) == pytest.approx(2.0)  # accelerating: sqrt(2x/a)
    assert eta(50.0, 0.0, 2.0, 10.0) == pytest.approx(10.0 / 2.0 + 25.0 / 10.0)  # then cruise
    assert eta(5.0, 0.0, 1.0, 0.0) == math.inf
    # early <= late for the same vehicle (F.3 bounds)
    v, a, vc = 4.0, 1.5, 12.0
    for x in (1.0, 10.0, 80.0):
        assert eta(x, v, a, max(v, vc)) <= eta(x, v, a * C.ETA_END_ACCEL_FACTOR, vc)


def test_reservations(junction: World) -> None:
    w = junction
    lane = _lane_end(w, "E_in_0")
    w.place("E_in_0", lane - 5.0, 5.0, route=("E_in", "W_out"), committed=True)
    w.place("E_in_0->W_out_0", 3.0, 5.0, route=("E_in", "W_out"), vtype="bus")
    w.place("N_in_0", lane - 5.0, 5.0, route=("N_in", "W_out"))  # not committed
    res = reservations(w.net, w.veh, w.types, w.run())
    bus = w.types.index["bus"]
    expected = C.VEHICLE_LENGTH + C.IDM_MIN_GAP + w.types.length[bus] + w.types.min_gap[bus]
    assert res[w.link("W_out_0")] == pytest.approx(expected)
    assert res.sum() == pytest.approx(expected)


def test_free_approach_outside_the_decision_zone(junction: World) -> None:
    h = junction.place("W_in_0", 10.0, 10.0, route=WE)
    adm, _ = junction.admit()
    assert adm.commits == 0 and not junction.veh.committed[h] and not junction.veh.held[h]
    assert math.isinf(adm.obstacle_gap[0]) and math.isinf(adm.cap_gap[0])


def test_commit_on_an_empty_junction(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", _lane_end(w, "W_in_0") - 10.0, 10.0, route=WE)
    adm, reserved = w.admit(next_seq=7)
    assert (adm.commits, adm.forced, adm.next_seq) == (1, 0, 8)
    assert w.veh.committed[h] and w.veh.commit_seq[h] == 7 and not w.veh.forced[h]
    assert reserved[w.link("E_out_0")] == pytest.approx(C.VEHICLE_LENGTH + C.IDM_MIN_GAP)
    assert math.isinf(adm.obstacle_gap[0]) and not w.veh.held[h]


def test_exit_space_blocks_the_box(junction: World) -> None:
    w = junction
    d = 10.0
    h = w.place("W_in_0", _lane_end(w, "W_in_0") - d, 5.0, route=WE)
    tail = w.place("E_out_0", 8.0, 0.0, route=("E_out",))  # rear 3 m < len + s0 = 7 m
    adm, _ = w.admit()
    assert adm.commits == 0 and not w.veh.committed[h] and w.veh.held[h]
    assert w.at(adm.obstacle_gap, h) == pytest.approx(d + C.IDM_MIN_GAP - C.STOP_LINE_CLEARANCE)
    assert w.at(adm.cap_gap, h) == pytest.approx(d)
    # enough room once the tail has moved on, unless committed vehicles reserved it
    w.veh.pos[tail] = 12.0
    adm, _ = w.admit()
    assert adm.commits == 1 and w.veh.committed[h] and not w.veh.held[h]
    w.veh.committed[h] = False
    w.place("S_in_0", _lane_end(w, "S_in_0") - 3.0, 1.0, route=("S_in", "E_out"), committed=True)
    adm, reserved = w.admit()
    assert not w.veh.committed[h] and w.veh.held[h]
    assert reserved[w.link("E_out_0")] == pytest.approx(C.VEHICLE_LENGTH + C.IDM_MIN_GAP)


def test_conflict_windows(junction: World) -> None:
    w = junction
    lane = _lane_end(w, "W_in_0")
    h = w.place("W_in_0", lane - 10.0, 10.0, route=WE)
    foe = w.place("N_in_0", lane - 5.0, 10.0, route=NS, committed=True)
    adm, _ = w.admit()
    assert not w.veh.committed[h] and w.veh.held[h] and adm.commits == 0
    # the same committed foe far upstream: disjoint windows
    w.veh.pos[foe] = 20.0
    w.veh.speed[foe] = 3.0
    adm, _ = w.admit()
    assert w.veh.committed[h] and adm.commits == 1


def test_foe_on_the_connector_until_it_clears_the_zone(junction: World) -> None:
    w = junction
    lane = _lane_end(w, "W_in_0")
    h = w.place("W_in_0", lane - 10.0, 10.0, route=WE)
    foe = w.place("N_in_0->S_out_0", 4.0, 8.0, route=NS)
    w.admit()
    assert not w.veh.committed[h]
    w.veh.pos[foe] = float(w.net.link_length[w.link("N_in_0->S_out_0")])  # rear still inside
    w.admit()
    assert not w.veh.committed[h]
    w.veh.link[foe] = w.link("S_out_0")  # gone: on its exit lane
    w.veh.committed[foe] = False
    w.veh.next_conn[foe] = -1
    w.veh.route_cursor[foe] = 1
    w.veh.pos[foe] = 10.0
    w.admit()
    assert w.veh.committed[h]


def test_foe_that_left_the_connector_until_its_rear_clears_the_zone(junction: World) -> None:
    """Review repro: j's front is on N_out_0 but its rear is still in the crossing zone of
    W_in_0->N_out_0 (``lock_conn``), so i on E_in_0 must not be admitted across it."""
    w = junction
    left = w.link("W_in_0->N_out_0")
    j = w.place("N_out_0", 1.0, 0.8, route=("N_out",))  # rear 4 m back on the connector
    w.veh.lock_conn[j] = left  # set on connector entry (engine/advance.py)
    i = w.place("E_in_0", _lane_end(w, "E_in_0") - 0.3, 0.0, route=("E_in", "W_out"))
    zone = next(
        z for z in w.junctions.conflicts[w.veh.next_conn[i] - w.net.n_lanes] if z[0] == left
    )
    rear = float(w.net.link_length[left]) + 1.0 - C.VEHICLE_LENGTH
    assert zone[3] <= rear < zone[4]  # inside the foe zone [fz_in, fz_out)
    w.admit()
    assert not w.veh.committed[i] and w.veh.held[i]
    w.veh.lock_conn[j] = -1  # a vehicle that came from another connector is no foe
    w.admit()
    assert w.veh.committed[i]
    w.veh.committed[i] = False
    w.veh.lock_conn[j] = left
    w.veh.pos[j] = C.VEHICLE_LENGTH + 0.5  # rear on the lane: the zone is clear
    w.admit()
    assert w.veh.committed[i]


def test_exit_space_counts_a_rear_hanging_back_over_the_target_lane(
    two_junctions: World,
) -> None:
    """Review repro: J1_J2_0 has no vehicle with its front on it, but a bus on its
    outgoing connector still occupies its last metres, so there is no room to admit i."""
    w = two_junctions
    route = ("W_J1", "J1_J2", "J2_E")
    i = w.place("W_J1_0", _lane_end(w, "W_J1_0") - 5.0, 5.0, route=route)
    bus = w.place("J1_J2_0->J2_S_0", 3.0, 0.0, route=("J1_J2", "J2_S"), vtype="bus")
    free = _lane_end(w, "J1_J2_0") + 3.0 - 12.0
    assert 0 < free < C.VEHICLE_LENGTH + C.IDM_MIN_GAP <= _lane_end(w, "J1_J2_0")
    w.admit()
    assert not w.veh.committed[i] and w.veh.held[i]
    w.veh.pos[bus] = 5.0  # 2 m further on: now there is room
    w.admit()
    assert w.veh.committed[i]


def test_first_come_first_served_at_equal_rank(junction: World) -> None:
    w = junction
    lane = _lane_end(w, "W_in_0")
    early = w.place("N_in_0", lane - 20.0, 10.0, route=NS)  # d/v = 2.0
    late = w.place("W_in_0", lane - 25.0, 10.0, route=WE)  # d/v = 2.5
    adm, _ = w.admit()
    assert w.veh.committed[early] and w.veh.commit_seq[early] == 0
    assert not w.veh.committed[late] and w.veh.held[late]
    assert adm.commits == 1


def test_rank_orders_the_admission_loop(priority: World) -> None:
    w = priority
    lane = _lane_end(w, "W_in_0")
    minor = w.place("N_in_0", lane - 10.0, 10.0, route=NS)  # rank 1, arrives first
    major = w.place("W_in_0", lane - 25.0, 10.0, route=WE)  # rank 3
    w.admit()
    assert w.veh.committed[major] and w.veh.commit_seq[major] == 0
    assert not w.veh.committed[minor] and w.veh.held[minor]


def _minor_vs_approaching_major(w: World) -> tuple[int, int]:
    lane = _lane_end(w, "W_in_0")
    major = w.place("W_in_0", lane - 70.0, 14.0, route=WE)  # outside its decision zone
    minor = w.place("N_in_0", lane - 5.0, 2.0, route=NS)
    return minor, major


def test_minor_yields_to_an_approaching_major(priority: World, junction: World) -> None:
    minor, major = _minor_vs_approaching_major(priority)
    decision = 14.0**2 / (2 * C.IDM_DECEL) + 14.0 + C.DECISION_MARGIN
    assert decision < 70.0
    priority.admit()
    assert not priority.veh.committed[major]  # free approach
    assert not priority.veh.committed[minor] and priority.veh.held[minor]
    # uncontrolled: equal ranks, an uncommitted vehicle is never a foe
    minor, major = _minor_vs_approaching_major(junction)
    junction.admit()
    assert junction.veh.committed[minor] and not junction.veh.committed[major]


def test_major_queued_behind_another_movement_is_not_a_foe(priority: World) -> None:
    w = priority
    lane = _lane_end(w, "W_in_0")
    # a major far-side turner (rank 2) waits at the line for space on N_out...
    turner = w.place("W_in_0", lane - 0.5, 0.0, route=("W_in", "N_out"))
    w.place("N_out_0", 6.0, 0.0, route=("N_out",))
    # ...so the major straight queued behind it cannot reach the line first
    straight = w.place("W_in_0", lane - 8.0, 0.0, route=WE)
    # the minor right turn S -> E merges with W -> E but does not conflict with W -> N
    minor = w.place("S_in_0", _lane_end(w, "S_in_0") - 5.0, 2.0, route=("S_in", "E_out"))
    w.admit()
    assert w.veh.held[turner] and not w.veh.committed[straight]
    assert w.veh.committed[minor]
    # with the turner gone the straight is the lane's first uncommitted vehicle: a foe
    w2 = World(w.net)
    w2.place("W_in_0", lane - 8.0, 0.0, route=WE)
    minor = w2.place("S_in_0", _lane_end(w2, "S_in_0") - 5.0, 2.0, route=("S_in", "E_out"))
    w2.admit()
    assert not w2.veh.committed[minor] and w2.veh.held[minor]


def test_force_commit_when_it_cannot_stop(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", _lane_end(w, "W_in_0") - 5.0, 12.0, route=WE)
    w.place("E_out_0", 6.0, 0.0, route=("E_out",))  # exit blocked
    assert 2 * 5.0 * C.IDM_EMERGENCY_DECEL < 12.0**2
    adm, reserved = w.admit()
    assert (adm.commits, adm.forced) == (1, 1)
    assert w.veh.committed[h] and w.veh.forced[h] and not w.veh.held[h]
    assert reserved[w.link("E_out_0")] == pytest.approx(C.VEHICLE_LENGTH + C.IDM_MIN_GAP)


def test_lane_end_obstacle_without_a_planned_connector(corridor_world: World) -> None:
    w = corridor_world
    d = 40.0
    h = w.place("J1_J2_2", _lane_end(w, "J1_J2_2") - d, 10.0, route=("J1_J2", "J2_N"))
    assert w.veh.next_conn[h] == -1  # lane 2 cannot reach J2_N: a lane change is needed
    adm, _ = w.admit()
    assert w.veh.held[h] and adm.commits == 0
    assert w.at(adm.obstacle_gap, h) == pytest.approx(d + C.IDM_MIN_GAP - C.STOP_LINE_CLEARANCE)
    assert w.at(adm.cap_gap, h) == pytest.approx(d)


def test_only_the_first_uncommitted_vehicle_is_a_candidate(junction: World) -> None:
    w = junction
    lane = _lane_end(w, "W_in_0")
    front = w.place("W_in_0", lane - 3.0, 0.0, route=WE)
    back = w.place("W_in_0", lane - 12.0, 5.0, route=WE)  # D = 16.25 m
    tail = w.place("E_out_0", 8.0, 0.0, route=("E_out",))
    adm, _ = w.admit()
    assert w.veh.held[front] and not w.veh.held[back]
    assert math.isinf(w.at(adm.obstacle_gap, back))
    w.veh.pos[tail] = 100.0
    adm, _ = w.admit()
    assert w.veh.committed[front] and not w.veh.committed[back] and adm.commits == 1
    adm, _ = w.admit()  # next step: the one behind a committed vehicle is the candidate
    assert w.veh.committed[back]


def test_last_road_vehicles_are_never_candidates(junction: World) -> None:
    w = junction
    h = w.place("E_out_0", _lane_end(w, "E_out_0") - 1.0, 10.0, route=("E_out",))
    adm, _ = w.admit()
    assert adm.commits == 0 and not w.veh.held[h] and math.isinf(adm.obstacle_gap[0])


def test_empty(junction: World) -> None:
    adm, _ = junction.admit(next_seq=3)
    assert adm.obstacle_gap.size == 0 and adm.next_seq == 3
