"""Zone locks inside intersections (plan F.3 "Zone locks", E.4 lock columns) on synthetic states.

Geometry of the 1-lane junction used below (zones ``[z_in, z_out]`` on the first connector,
then on the foe): W->E (10.4 m) crosses N->S at [2.1, 5.1] / [5.3, 8.3] and S->N at
[5.3, 8.3] / [2.1, 5.1]; W->S (a 5.65 m right turn) merges with N->S only among those.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from urbanflow.core import constants as C
from urbanflow.engine import Engine
from urbanflow.engine.intersections import release_locks
from urbanflow.scenario import ScenarioBuilder

from .conftest import World

WE, NS, SN, WS = ("W_in", "E_out"), ("N_in", "S_out"), ("S_in", "N_out"), ("W_in", "S_out")
V = 5.0  # m/s; request distance v^2/(2b) + v dt + a dt^2/2 + 5 m = 17 m at dt 1


def _lane(w: World) -> float:
    return float(w.net.link_length[w.link("W_in_0")])


def _zone(w: World, conn: str, foe: str) -> tuple[float, float, float, float]:
    """``(z_in, z_out, foe z_in, foe z_out)`` of the conflict of ``conn`` with ``foe``."""
    for z in w.junctions.conflicts[w.link(conn) - w.net.n_lanes]:
        if z[0] == w.link(foe):
            return z[1], z[2], z[3], z[4]
    raise AssertionError(f"{conn} does not conflict with {foe}")


def _commit(w: World, h: int, seq: int, *, forced: bool = False) -> int:
    w.veh.committed[h] = True
    w.veh.commit_seq[h] = seq
    w.veh.forced[h] = forced
    return h


def _owner(w: World, conn: str, pos: float, route: tuple[str, ...], seq: int = 0) -> int:
    """A vehicle on ``conn`` holding its lock (granted earlier)."""
    h = _commit(w, w.place(conn, pos, V, route=route), seq)
    w.veh.granted[h] = True
    w.veh.lock_conn[h] = w.link(conn)
    return h


# --------------------------------------------------------------------------- requests
def test_request_and_grant_on_an_empty_junction(junction: World) -> None:
    w = junction
    near = _commit(w, w.place("W_in_0", _lane(w) - 3.0, V, route=WE), 0)
    far = _commit(w, w.place("S_in_0", _lane(w) - 60.0, V, route=SN), 1)
    g = w.grant()
    assert w.veh.granted[near] and w.veh.lock_conn[near] == w.link("W_in_0->E_out_0")
    assert not w.veh.held[near] and np.isinf(w.at(g.obstacle_gap, near))
    # 62 m to its first zone > 17 m: no request yet, so neither granted nor held
    assert not w.veh.granted[far] and w.veh.lock_conn[far] == -1 and not w.veh.held[far]
    assert (g.granted, g.forced, g.revoked, g.conflicts) == (1, 0, 0, 0)
    assert np.isinf(w.at(g.obstacle_gap, far)) and w.at(g.hold_link, far) == -1


def test_a_connector_without_crossing_or_merging_zones_is_granted_at_commit(
    two_junctions: World,
) -> None:
    w = two_junctions
    route = ("W_J1", "J1_J2", "J2_E")
    assert not w.junctions.conflicts[w.link("W_J1_0->J1_J2_0") - w.net.n_lanes]
    h = _commit(w, w.place("W_J1_0", 20.0, V, route=route), 0)  # far from the line
    w.grant()
    assert w.veh.granted[h] and w.veh.lock_conn[h] == -1 and not w.veh.held[h]


def test_rule_a_a_locked_zone_holds_the_request_at_its_first_zone(junction: World) -> None:
    w = junction
    owner = _owner(w, "N_in_0->S_out_0", 4.0, NS)
    i = _commit(w, w.place("W_in_0", _lane(w) - 3.0, V, route=WE), 1)
    z_in = _zone(w, "W_in_0->E_out_0", "N_in_0->S_out_0")[0]
    g = w.grant()
    assert not w.veh.granted[i] and w.veh.held[i] and w.veh.lock_conn[i] == -1
    dist = 3.0 + z_in
    assert w.at(g.obstacle_gap, i) == pytest.approx(dist + C.IDM_MIN_GAP - C.STOP_LINE_CLEARANCE)
    assert w.at(g.cap_gap, i) == pytest.approx(dist)
    assert w.at(g.hold_link, i) == w.link("W_in_0->E_out_0")
    assert w.at(g.hold_pos, i) == pytest.approx(z_in)
    assert w.veh.granted[owner] and not w.veh.held[owner]


def test_each_zone_is_released_when_the_owners_rear_passes_its_z_out(junction: World) -> None:
    """Rear coordinates are mapped back across the connector end (E.4 lock_conn): the owner
    on S_out_0 at ``p`` has its rear at ``p - len + L_c`` on N->S."""
    w = junction
    ns = w.link("N_in_0->S_out_0")
    owner = _owner(w, "N_in_0->S_out_0", 4.0, NS)
    i = _commit(w, w.place("W_in_0", _lane(w) - 3.0, V, route=WE), 1)
    fz_out = _zone(w, "W_in_0->E_out_0", "N_in_0->S_out_0")[3]
    w.veh.link[owner], w.veh.route_cursor[owner] = w.link("S_out_0"), 1  # on its exit lane
    w.veh.committed[owner] = w.veh.granted[owner] = False  # cleared on lane entry
    length = float(w.net.link_length[ns])
    w.veh.pos[owner] = fz_out - length + C.VEHICLE_LENGTH - 0.1  # rear 0.1 m before z_out
    w.grant()
    assert not w.veh.granted[i]
    w.veh.pos[owner] += 0.2  # rear 0.1 m past z_out: this zone is free
    w.grant()
    assert w.veh.granted[i] and w.veh.lock_conn[owner] == ns  # its later zones stay locked


def test_all_or_nothing(junction: World) -> None:
    """A lock on a later zone of the set blocks the whole request; the vehicle waits at its
    first zone, not at the locked one."""
    w = junction
    _owner(w, "S_in_0->N_out_0", 1.0, SN)  # locks W->E's zone [5.3, 8.3]
    i = _commit(w, w.place("W_in_0", _lane(w) - 3.0, V, route=WE), 1)
    first = w.junctions.conflicts[w.link("W_in_0->E_out_0") - w.net.n_lanes][0][1]
    locked = _zone(w, "W_in_0->E_out_0", "S_in_0->N_out_0")[0]
    assert first < locked
    g = w.grant()
    assert not w.veh.granted[i] and w.at(g.hold_pos, i) == pytest.approx(first)


def test_rule_b_an_earlier_committed_ungranted_foe_arriving_first(junction: World) -> None:
    w = junction
    lane = _lane(w)
    _owner(w, "E_in_0->W_out_0", 1.0, ("E_in", "W_out"))  # blocks j only (E->W vs N->S)
    j = _commit(w, w.place("N_in_0", lane - 5.0, V, route=NS), 3)
    i = _commit(w, w.place("W_in_0", lane - 3.0, V, route=WE), 5)
    w.grant()
    assert not w.veh.granted[j] and w.veh.held[j]  # rule (a)
    assert not w.veh.granted[i] and w.veh.held[i]  # rule (b): j is earlier and arrives first
    w.veh.commit_seq[j] = 7  # committed after i: no longer i's concern
    w.grant()
    assert w.veh.granted[i] and not w.veh.granted[j]
    # an earlier foe that arrives only after i has cleared does not hold i back
    w2 = World(w.net)
    j = _commit(w2, w2.place("N_in_0", lane - 60.0, V, route=NS), 3)
    i = _commit(w2, w2.place("W_in_0", lane - 3.0, V, route=WE), 5)
    w2.grant()
    assert w2.veh.granted[i] and not w2.veh.granted[j] and not w2.veh.held[j]


def test_rule_c_a_committed_but_ungranted_leader_from_the_same_lane(junction: World) -> None:
    w = junction
    lane = _lane(w)
    k = _owner(w, "S_in_0->N_out_0", 1.0, SN)  # blocks W->E, not the right turn W->S
    leader = _commit(w, w.place("W_in_0", lane - 1.0, 0.0, route=WE), 1)
    i = _commit(w, w.place("W_in_0", lane - 9.0, 3.0, route=WS), 2)
    assert w.at(w.leaders().leader, i) == leader
    w.grant()
    assert not w.veh.granted[leader] and not w.veh.granted[i] and w.veh.held[i]
    w.veh.link[k], w.veh.pos[k] = w.link("N_out_0"), 20.0  # k has left: the leader gets
    w.veh.route_cursor[k] = 1  # its grant, then i
    w.grant()
    assert w.veh.granted[leader] and w.veh.granted[i]


# --------------------------------------------------------------------------- force grants
def test_force_committed_vehicles_are_granted_and_revoke_owners_that_can_stop(
    junction: World,
) -> None:
    w = junction
    lane = _lane(w)
    owner = _commit(w, w.place("N_in_0", lane - 8.0, V, route=NS), 0)  # granted, not entered
    w.veh.granted[owner], w.veh.lock_conn[owner] = True, w.link("N_in_0->S_out_0")
    i = _commit(w, w.place("W_in_0", lane - 20.0, 10.0, route=WE), 1, forced=True)
    g = w.grant()
    assert w.veh.granted[i] and w.veh.forced[i] and w.veh.lock_conn[i] == w.link("W_in_0->E_out_0")
    assert not w.veh.granted[owner] and w.veh.lock_conn[owner] == -1 and w.veh.held[owner]
    dist = 8.0 + w.junctions.conflicts[w.link("N_in_0->S_out_0") - w.net.n_lanes][0][1]
    assert w.at(g.cap_gap, owner) == pytest.approx(dist)
    assert (g.revoked, g.conflicts, g.forced) == (1, 0, 1)
    g = w.grant()  # the revoked owner requests again: i's lock now holds it back
    assert not w.veh.granted[owner] and w.veh.held[owner] and g.revoked == 0


def test_owners_inside_a_zone_or_forced_are_not_revoked(junction: World) -> None:
    w = junction
    lane = _lane(w)
    inside = _owner(w, "N_in_0->S_out_0", 4.0, NS)  # inside its first zone [2.1, 5.1]
    i = _commit(w, w.place("W_in_0", lane - 20.0, 10.0, route=WE), 1, forced=True)
    g = w.grant()
    assert w.veh.granted[i] and w.veh.granted[inside] and (g.revoked, g.conflicts) == (0, 1)
    w2 = World(w.net)
    owner = _commit(w2, w2.place("N_in_0", lane - 8.0, V, route=NS), 0, forced=True)
    w2.veh.granted[owner], w2.veh.lock_conn[owner] = True, w2.link("N_in_0->S_out_0")
    i = _commit(w2, w2.place("W_in_0", lane - 20.0, 10.0, route=WE), 1, forced=True)
    g = w2.grant()
    assert w2.veh.granted[owner] and (g.revoked, g.conflicts) == (0, 1)


def test_a_vehicle_that_cannot_stop_before_its_zone_is_granted_forced(junction: World) -> None:
    w = junction
    _owner(w, "N_in_0->S_out_0", 4.0, NS)
    i = _commit(w, w.place("W_in_0", _lane(w) - 1.0, 12.0, route=WE), 1)
    dist = 1.0 + _zone(w, "W_in_0->E_out_0", "N_in_0->S_out_0")[0]
    assert 2 * dist * C.IDM_EMERGENCY_DECEL < 12.0**2
    g = w.grant()
    assert w.veh.granted[i] and w.veh.forced[i] and not w.veh.held[i]
    assert g.forced == 1 and g.conflicts == 1  # the owner is inside its zone


def test_force_grants_cascade_to_ungranted_leaders_of_the_same_lane(junction: World) -> None:
    w = junction
    lane = _lane(w)
    k = _owner(w, "S_in_0->N_out_0", 1.0, SN)  # holds the leader back (rule a)
    w.veh.speed[k] = 2.0  # it can still stop 1.1 m before its first zone
    leader = _commit(w, w.place("W_in_0", lane - 1.0, 0.0, route=WE), 1)
    w.grant()
    assert not w.veh.granted[leader]
    i = _commit(w, w.place("W_in_0", lane - 9.0, 3.0, route=WS), 2, forced=True)
    g = w.grant()
    assert w.veh.granted[i] and w.veh.granted[leader] and w.veh.forced[leader]
    assert g.forced == 2 and w.at(g.hold_link, leader) == -1
    # the cascade grant revoked k: it has not entered a zone and can stop
    assert not w.veh.granted[k] and w.veh.lock_conn[k] == -1 and g.revoked == 1


def test_the_previous_connectors_lock_defers_the_next_one(junction: World) -> None:
    """Rule (d): a vehicle still holding the lock of the connector behind it is not granted
    the next set; a force grant is made but takes the new lock only once the old one is
    released (overlaps counted then)."""
    w = junction
    other = w.link("E_in_0->W_out_0")
    i = _commit(w, w.place("W_in_0", _lane(w) - 3.0, V, route=WE), 1)
    w.veh.lock_conn[i] = other  # hand-set: not on its path any more
    w.grant()
    assert not w.veh.granted[i] and w.veh.held[i]
    w.veh.forced[i] = True
    _owner(w, "N_in_0->S_out_0", 4.0, NS)
    g = w.grant()
    assert w.veh.granted[i] and w.veh.lock_conn[i] == other and g.conflicts == 0
    release_locks(w.net, w.veh, w.junctions, w.run())  # off the old path: released
    assert w.veh.lock_conn[i] == -1
    g = w.grant()
    assert w.veh.lock_conn[i] == w.link("W_in_0->E_out_0") and g.conflicts == 1


# --------------------------------------------------------------------------- release
def test_release_when_the_rear_passes_the_last_z_out(junction: World) -> None:
    w, veh = junction, junction.veh
    conn = w.link("W_in_0->E_out_0")
    length, end = float(w.net.link_length[conn]), float(w.junctions.lock_end[conn - w.net.n_lanes])
    assert end == pytest.approx(length)  # its last zone is the merge at the connector end
    approach = _commit(w, w.place("W_in_0", _lane(w) - 3.0, V, route=WE), 0)
    on_conn = _owner(w, "W_in_0->E_out_0", 8.0, WE)
    out = w.place("E_out_0", 4.0, V, route=WE)  # rear at 4 - 5 + L_c on the connector
    gone = w.place("E_out_0", 6.0, V, route=WE)
    committed_again = w.place("E_out_0", 6.0 + 2 * C.VEHICLE_LENGTH, V, route=WE)
    hs = [approach, on_conn, out, gone, committed_again]
    veh.lock_conn[hs] = conn
    veh.forced[hs] = True
    veh.committed[committed_again] = True
    release_locks(w.net, veh, w.junctions, w.run())
    assert veh.lock_conn[hs].tolist() == [conn, conn, conn, -1, -1]
    assert veh.forced[hs].tolist() == [True, True, True, False, True]


def test_removal_releases_locks(make_engine: Callable[..., Engine]) -> None:
    b = ScenarioBuilder("lock")
    b.intersection("J", (0.0, 0.0), kind="uncontrolled")
    for name, (dx, dy) in {"W": (-1, 0), "E": (1, 0), "N": (0, 1), "S": (0, -1)}.items():
        b.boundary(name, (dx * 100.0, dy * 100.0))
        b.road(f"{name}_in", name, "J")
        b.road(f"{name}_out", "J", name)
    b.trip("we", 0.0, route=["W_in", "E_out"])
    e = make_engine(b.build())
    veh = e.vehicles
    while "we" not in veh.id_to_handle or veh.lock_conn[veh.id_to_handle["we"]] < 0:
        e.step()
        assert e.time < 60.0
    h = veh.id_to_handle["we"]
    e.commands.remove_vehicle("we")
    assert veh.lock_conn[h] == -1
