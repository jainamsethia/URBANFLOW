"""Admission at stop lines (plan F.3, B.2 #25) and the F.1 lookahead rule on synthetic states."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urbanflow.core import constants as C
from urbanflow.core.types import SignalState
from urbanflow.engine.intersections import eta, reservations, signal_lookahead
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.scenario import ScenarioBuilder

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


def test_exit_space_credits_the_last_vehicles_emergency_stopping_distance(
    junction: World,
) -> None:
    """B.2 #25: ``free(to)`` = rear of the last body on ``to`` + ``v^2 / (2 b_emerg)``, the
    distance it still travels under emergency braking; a moving platoon does not look
    parked (review finding: static rears held each queued vehicle ~14 m too long)."""
    w = junction
    lane = w.link("E_out_0")
    h = w.place("W_in_0", _lane_end(w, "W_in_0") - 10.0, 5.0, route=WE)
    tail = w.place("E_out_0", 8.0, 0.0, route=("E_out",))  # rear 3 m < len + s0 = 7 m
    need = C.VEHICLE_LENGTH + C.IDM_MIN_GAP
    b_emerg = float(w.types.emergency_decel[w.veh.type_idx[tail]])
    v_edge = math.sqrt(2 * b_emerg * (need - 3.0))  # free = 7 m exactly
    for v, admitted in ((0.0, False), (v_edge - 0.01, False), (v_edge + 0.01, True)):
        w.veh.committed[h] = False
        w.veh.speed[tail] = v
        assert w.leaders().lane_free[lane] == pytest.approx(3.0 + v * v / (2 * b_emerg))
        w.admit()
        assert bool(w.veh.committed[h]) is admitted, v
    # an empty lane is free to its end; reservations still count against a moving tail
    assert World(w.net).leaders().lane_free[lane] == pytest.approx(_lane_end(w, "E_out_0"))
    w.veh.committed[h] = False
    w.place("S_in_0", _lane_end(w, "S_in_0") - 3.0, 1.0, route=("S_in", "E_out"), committed=True)
    w.admit()
    assert not w.veh.committed[h] and w.veh.held[h]


def test_exit_space_bound_uses_an_overhanging_connector_body(two_junctions: World) -> None:
    """The overhanging bus of the P2 repro, now moving: its emergency stopping distance
    frees the space (rear -5.4 m on the lane end, v^2/(2 b_emerg) = 12 m at 12 m/s)."""
    w = two_junctions
    route = ("W_J1", "J1_J2", "J2_E")
    i = w.place("W_J1_0", _lane_end(w, "W_J1_0") - 5.0, 5.0, route=route)
    bus = w.place("J1_J2_0->J2_S_0", 3.0, 0.0, route=("J1_J2", "J2_S"), vtype="bus")
    w.admit()
    assert not w.veh.committed[i]
    w.veh.speed[bus] = 12.0
    b_emerg = float(w.types.emergency_decel[w.veh.type_idx[bus]])
    rear = _lane_end(w, "J1_J2_0") + 3.0 - float(w.veh.length[bus])
    free = w.leaders().lane_free[w.link("J1_J2_0")]
    assert free == pytest.approx(rear + 144.0 / (2 * b_emerg)) and free >= 7.0
    w.admit()
    assert w.veh.committed[i]


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
    w.veh.lock_conn[j] = left  # its zone lock (engine/intersections.py::grant_zones)
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


# --------------------------------------------------------------------------- signals (F.3 step 1)
EW_GREEN = {"W_in->E_out": "G", "E_in->W_out": "G", "E_in->S_out": "g", "W_in->N_out": "g"}
V = 13.9


def test_red_stops_even_outside_the_decision_zone(signalised: World) -> None:
    w = signalised
    h = w.place("W_in_0", _lane_end(w, "W_in_0") - 150.0, V, route=WE)
    adm, _ = w.admit(signals={})
    assert w.veh.held[h] and not w.veh.committed[h] and adm.commits == 0
    assert w.at(adm.obstacle_gap, h) == pytest.approx(150.0 + C.IDM_MIN_GAP - 0.5)
    adm, _ = w.admit(signals={"W_in->E_out": "G"})  # green: free approach
    assert not w.veh.held[h] and math.isinf(w.at(adm.obstacle_gap, h))


def test_red_runner_that_cannot_stop_is_force_committed(signalised: World) -> None:
    w = signalised
    h = w.place("W_in_0", _lane_end(w, "W_in_0") - 5.0, V, route=WE)
    assert V * V / (2 * 5.0) > C.IDM_EMERGENCY_DECEL
    adm, reserved = w.admit(signals={})
    assert (adm.commits, adm.forced, adm.red_runs) == (1, 0, 1)
    assert w.veh.committed[h] and w.veh.forced[h] and not w.veh.held[h]
    assert reserved[w.link("E_out_0")] == pytest.approx(C.VEHICLE_LENGTH + C.IDM_MIN_GAP)


def test_yellow_dilemma(signalised: World) -> None:
    """AT-11 on a synthetic state: 10 m proceeds, 60 m at 13.9 m/s stops."""
    w = signalised
    lane = _lane_end(w, "W_in_0")
    near = w.place("W_in_0", lane - 10.0, V, route=WE)
    far = w.place("E_in_0", lane - 60.0, V, route=("E_in", "W_out"))
    assert V * V / (2 * 60.0) <= C.YELLOW_MAX_DECEL < V * V / (2 * 10.0)
    adm, _ = w.admit(signals={"W_in->E_out": "y", "E_in->W_out": "y"})
    assert w.veh.committed[near] and not w.veh.forced[near] and not w.veh.held[near]
    assert w.veh.held[far] and not w.veh.committed[far]
    assert (adm.commits, adm.forced, adm.red_runs) == (1, 0, 0)


def test_yellow_dilemma_with_a_blocked_exit(signalised: World) -> None:
    w = signalised
    lane = _lane_end(w, "W_in_0")
    w.place("E_out_0", 6.0, 0.0, route=("E_out",))  # no room behind it
    # 20 m: too close for a comfortable stop (4.8 m/s^2) but can still stop (<= 6)
    h = w.place("W_in_0", lane - 20.0, V, route=WE)
    adm, _ = w.admit(signals={"W_in->E_out": "y"})
    assert w.veh.held[h] and not w.veh.committed[h] and adm.commits == 0
    # 12 m: cannot stop any more -> force-commit, counted in forced_commits
    w.veh.pos[h] = lane - 12.0
    adm, _ = w.admit(signals={"W_in->E_out": "y"})
    assert (adm.commits, adm.forced, adm.red_runs) == (1, 1, 0) and w.veh.forced[h]


def test_yellow_dilemma_skips_the_decision_zone(signalised: World) -> None:
    """A hard-braking vehicle (b = 6) at 20 m/s is outside its decision zone at 62 m but
    inside the dilemma zone (400/124 = 3.2 m/s^2 > 3): it decides at yellow onset."""
    w = signalised
    d, v = 62.0, 20.0
    h = w.place("W_in_0", _lane_end(w, "W_in_0") - d, v, route=WE, vtype="hard")
    assert d > v * v / (2 * 6.0) + v * 1.0 + C.DECISION_MARGIN
    w.admit(signals={"W_in->E_out": "G"})
    assert not w.veh.committed[h] and not w.veh.held[h]  # green: free approach
    w.admit(signals={"W_in->E_out": "y"})
    assert w.veh.committed[h] and not w.veh.forced[h]


def test_committed_vehicles_never_recheck_the_light(signalised: World) -> None:
    w = signalised
    h = w.place("W_in_0", _lane_end(w, "W_in_0") - 30.0, 5.0, route=WE, committed=True)
    adm, _ = w.admit(signals={})
    assert w.veh.committed[h] and not w.veh.held[h] and adm.red_runs == 0


def test_permissive_yields_to_the_opposing_protected_movement(signalised: World) -> None:
    w = signalised
    lane = _lane_end(w, "E_in_0")
    left = w.place("E_in_0", lane - 0.5, 0.0, route=("E_in", "S_out"))  # g, at the line
    straight = w.place("W_in_0", lane - 60.0, 12.0, route=WE)  # G, uncommitted, approaching
    w.admit(signals=EW_GREEN)
    assert not w.veh.committed[left] and w.veh.held[left]
    assert not w.veh.committed[straight]  # outside its decision zone
    # the protected vehicle far away (ETA ~16 s): the permissive turn fits before it
    w.veh.pos[straight] = lane - 190.0
    w.admit(signals=EW_GREEN)
    assert w.veh.committed[left]
    # both at the stop line: G commits first (rank), g must wait for it
    w2 = World(w.net)
    left = w2.place("E_in_0", lane - 0.5, 0.0, route=("E_in", "S_out"))
    straight = w2.place("W_in_0", lane - 0.5, 0.0, route=WE)
    w2.admit(signals=EW_GREEN)
    assert w2.veh.committed[straight] and w2.veh.commit_seq[straight] == 0
    assert not w2.veh.committed[left] and w2.veh.held[left]


# --------------------------------------------------------------------------- F.1 step 4 rule
@pytest.fixture(scope="module")
def series_net() -> CompiledNetwork:
    """W -> J1 (uncontrolled) -> J2 (signalized) -> E; J1_J2 is 30 m long (1 lane)."""
    b = ScenarioBuilder("series")
    b.boundary("W", (-200.0, 0.0))
    b.intersection("J1", (0.0, 0.0), kind="uncontrolled")
    b.intersection("J2", (40.0, 0.0), kind="signalized")
    b.boundary("E", (240.0, 0.0))
    b.road("W_J1", "W", "J1")
    b.road("J1_J2", "J1", "J2")
    b.road("J2_E", "J2", "E")
    return compile_network(b.build())


def test_lookahead_stop_line_is_an_idm_obstacle_at_r_and_y(series_net: CompiledNetwork) -> None:
    w = World(series_net)
    route = ("W_J1", "J1_J2", "J2_E")
    conn = w.place("W_J1_0->J1_J2_0", 2.0, 10.0, route=route)  # on J1's connector
    lane = w.place("W_J1_0", _lane_end(w, "W_J1_0") - 3.0, 10.0, route=route, committed=True)
    far = w.place("W_J1_0", 50.0, 10.0, route=route)  # uncommitted: its lookahead ends at J1
    leaders = w.leaders()
    stop = leaders.stop_gap
    length = {k: float(w.net.link_length[w.link(k)]) for k in ("W_J1_0->J1_J2_0", "J1_J2_0")}
    assert w.at(stop, conn) == pytest.approx(length["W_J1_0->J1_J2_0"] - 2.0 + length["J1_J2_0"])
    assert w.at(stop, lane) == pytest.approx(3.0 + sum(length.values()))
    assert math.isinf(w.at(stop, far))
    state = np.full(w.net.n_movements, SignalState.G.code, dtype=np.uint8)
    j2 = w.net.mov_index["J1_J2->J2_E"]
    run = w.run()
    for code, obstacle in (
        (SignalState.G, False),
        (SignalState.g, False),
        (SignalState.y, True),
        (SignalState.r, True),
    ):
        state[j2] = code.code
        gaps = signal_lookahead(w.net, w.veh, w.types, run, stop, w.routes, w.junctions, state)
        for h in (conn, lane):
            expected = w.at(stop, h) + C.IDM_MIN_GAP - C.STOP_LINE_CLEARANCE
            assert w.at(gaps, h) == (pytest.approx(expected) if obstacle else math.inf), code
        assert math.isinf(w.at(gaps, far))


def test_lookahead_obstacle_needs_the_stop_test_of_admission(series_net: CompiledNetwork) -> None:
    """B.2 #25 (review repro): a y stop line is an IDM obstacle only if v^2/(2d) <=
    YELLOW_MAX_DECEL (else the vehicle drives through, as admission's dilemma rule lets
    it), an r line only if v^2/(2d) <= b_emerg; both always stay G.2 cap obstacles."""
    w = World(series_net)
    route = ("W_J1", "J1_J2", "J2_E")
    h = w.place("W_J1_0->J1_J2_0", 2.0, 10.0, route=route)
    d = w.at(w.leaders().stop_gap, h)
    obstacle = d + C.IDM_MIN_GAP - C.STOP_LINE_CLEARANCE
    state = np.full(w.net.n_movements, SignalState.G.code, dtype=np.uint8)
    j2 = w.net.mov_index["J1_J2->J2_E"]

    def gap(light: SignalState, v: float) -> float:
        w.veh.speed[h] = v
        state[j2] = light.code
        leaders = w.leaders()
        assert w.at(leaders.stop_gap, h) == pytest.approx(d)  # the cap obstacle stays
        args = (w.net, w.veh, w.types, w.run(), leaders.stop_gap, w.routes, w.junctions, state)
        return w.at(signal_lookahead(*args), h)

    v_y = math.sqrt(2 * d * C.YELLOW_MAX_DECEL)
    v_r = math.sqrt(2 * d * C.IDM_EMERGENCY_DECEL)
    assert gap(SignalState.y, 0.99 * v_y) == pytest.approx(obstacle)
    assert math.isinf(gap(SignalState.y, 1.01 * v_y))  # drives through the yellow
    assert gap(SignalState.r, 1.01 * v_y) == pytest.approx(obstacle)  # red: stop if able
    assert gap(SignalState.r, 0.99 * v_r) == pytest.approx(obstacle)
    assert math.isinf(gap(SignalState.r, 1.01 * v_r))  # cannot stop: admission decides
    assert math.isinf(gap(SignalState.G, 0.5 * v_y))


# --------------------------------------------------------------------------- end-of-green clearing
LEFT = ("E_in", "S_out")  # g in EW_GREEN, opposing W_in -> E_out
EW_YELLOW = {"W_in->E_out": "y", "E_in->W_out": "y", "E_in->S_out": "y", "W_in->N_out": "y"}


def _sneaker(w: World, *, d: float = 0.5) -> int:
    """A left turner stopped ``d`` m before the line."""
    return w.place("E_in_0", _lane_end(w, "E_in_0") - d, 0.0, route=LEFT)


def _eligible(w: World, *handles: int) -> np.ndarray:
    """Per-lane end-of-green eligibility as the engine keeps it in the yellow: the uids of
    ``handles``, held at their line in the last green step (-1 elsewhere)."""
    out = np.full(w.net.n_lanes, -1, dtype=np.int64)
    for h in handles:
        out[w.veh.link[h]] = w.veh.uid[h]
    return out


def test_a_sneaker_commits_in_the_yellow_that_ends_its_g(signalised: World) -> None:
    """F.3 end-of-green clearing: exit space only, no gap check; after every other
    candidate of the step, so the opposing dilemma vehicle commits first."""
    w = signalised
    lane = _lane_end(w, "W_in_0")
    i = _sneaker(w)
    j = w.place("W_in_0", lane - 20.0, V, route=WE)  # dilemma: 4.8 m/s^2 > 3 to stop
    sneakers = _eligible(w, i)
    adm, reserved = w.admit(signals=EW_YELLOW, permissive=["E_in->S_out"], sneakers=sneakers)
    assert w.veh.committed[j] and w.veh.committed[i] and not w.veh.forced[i]
    assert w.veh.commit_seq[j] < w.veh.commit_seq[i]
    assert (adm.commits, adm.sneakers, adm.forced) == (2, 1, 0)
    assert sneakers[w.link("E_in_0")] == -1 and not w.veh.held[i]  # consumed
    assert reserved[w.link("S_out_0")] == pytest.approx(C.VEHICLE_LENGTH + C.IDM_MIN_GAP)
    # the same state in green: the gap check holds it (j arrives in ~1.4 s), and being
    # held at the line on a g movement makes it eligible for the coming yellow
    w2 = World(w.net)
    i = _sneaker(w2)
    w2.place("W_in_0", lane - 20.0, V, route=WE, committed=True)
    sneakers = np.full(w.net.n_lanes, -1, dtype=np.int64)
    w2.admit(signals=EW_GREEN, permissive=[], sneakers=sneakers)
    assert not w2.veh.committed[i] and w2.veh.held[i]
    assert sneakers.tolist() == _eligible(w2, i).tolist()
    adm, _ = w2.admit(signals=EW_YELLOW, permissive=["E_in->S_out"], sneakers=sneakers)
    assert w2.veh.committed[i] and adm.sneakers == 1


def test_a_vehicle_that_reaches_the_line_in_the_yellow_does_not_sneak(signalised: World) -> None:
    """P4 review repro (B.2 #25: it must have waited at the line since before the yellow):
    a left turner still approaching at the end of the green is not eligible; it brakes for
    the yellow (2.56 m/s, 4.07 m out: held), and once stopped at the line it stays held."""
    w = signalised
    i = w.place("E_in_0", _lane_end(w, "E_in_0") - 60.0, 10.0, route=LEFT)
    sneakers = np.full(w.net.n_lanes, -1, dtype=np.int64)
    w.admit(signals=EW_GREEN, permissive=[], sneakers=sneakers)  # free approach (d > D)
    assert not w.veh.held[i] and (sneakers == -1).all()
    w.veh.pos[i], w.veh.speed[i] = _lane_end(w, "E_in_0") - 4.07, 2.56
    for _ in range(2):  # braking in the yellow, then stopped at the line
        adm, _ = w.admit(signals=EW_YELLOW, permissive=["E_in->S_out"], sneakers=sneakers)
        assert not w.veh.committed[i] and w.veh.held[i] and adm.sneakers == 0
        assert (sneakers == -1).all()
        w.veh.pos[i], w.veh.speed[i] = _lane_end(w, "E_in_0") - 0.5, 0.0


@pytest.mark.parametrize(
    ("case", "signals", "permissive"),
    [
        ("not held before", EW_YELLOW, ["E_in->S_out"]),
        ("another vehicle eligible", EW_YELLOW, ["E_in->S_out"]),
        ("away from the line", EW_YELLOW, ["E_in->S_out"]),
        ("exit blocked", EW_YELLOW, ["E_in->S_out"]),
        ("protected yellow", EW_YELLOW, []),
        ("red", {}, ["E_in->S_out"]),
    ],
)
def test_no_sneaking_otherwise(
    signalised: World, case: str, signals: dict[str, str], permissive: list[str]
) -> None:
    w = signalised
    i = _sneaker(w, d=10.0 if case == "away from the line" else 0.5)
    sneakers = _eligible(w, i)
    if case == "not held before":
        sneakers[:] = -1
    if case == "another vehicle eligible":
        sneakers[w.link("E_in_0")] += 1
    if case == "exit blocked":
        w.place("S_out_0", 4.0, 0.0, route=("S_out",))
    before = sneakers.copy()
    adm, _ = w.admit(signals=signals, permissive=permissive, sneakers=sneakers)
    assert not w.veh.committed[i] and w.veh.held[i] and adm.sneakers == 0
    assert sneakers.tolist() == before.tolist()  # still eligible (blocked exit: next step)


def test_at_most_the_quota_per_lane_and_yellow(signalised: World) -> None:
    w = signalised
    first = _sneaker(w)
    sneakers = _eligible(w, first)
    w.admit(signals=EW_YELLOW, permissive=["E_in->S_out"], sneakers=sneakers)
    assert w.veh.committed[first]
    second = _sneaker(w, d=0.4)  # the next one reaches the line in the same yellow
    w.veh.pos[first] = 3.0
    w.veh.link[first] = w.link("E_in_0->S_out_0")
    w.admit(signals=EW_YELLOW, permissive=["E_in->S_out"], sneakers=sneakers)
    assert not w.veh.committed[second] and w.veh.held[second]
    assert C.SNEAKERS_PER_PHASE == 1  # one eligible uid per lane, consumed by the commit
    # the next green holds it at the line (opposing traffic), so the next yellow serves it
    w.place("W_in_0", _lane_end(w, "W_in_0") - 20.0, V, route=WE, committed=True)
    sneakers[:] = -1  # the engine's reset outside the yellow
    w.admit(signals=EW_GREEN, permissive=[], sneakers=sneakers)
    assert not w.veh.committed[second] and sneakers[w.link("E_in_0")] == w.veh.uid[second]
    w.admit(signals=EW_YELLOW, permissive=["E_in->S_out"], sneakers=sneakers)
    assert w.veh.committed[second]
