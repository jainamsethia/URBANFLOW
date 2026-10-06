"""Every invariant detects a hand-corrupted state (plan F.7)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest

from urbanflow import generate
from urbanflow.core.errors import InvariantViolation, SimulationError
from urbanflow.core.types import SignalState, VehicleStatus
from urbanflow.engine import Engine, check_always, check_debug
from urbanflow.engine.advance import Advance
from urbanflow.engine.invariants import Holds
from urbanflow.scenario import ScenarioBuilder


@pytest.fixture
def engine(
    make_engine: Callable[..., Engine], junction_builder: Callable[..., ScenarioBuilder]
) -> Engine:
    b = junction_builder()
    b.flow("we", route=["W_in", "E_out"], period=3.0)
    b.flow("ns", route=["N_in", "S_out"], period=4.0)
    e = make_engine(b.build())
    for _ in range(40):
        e.step()
        if (e.vehicles.link[e.vehicles.running()] >= e.network.n_lanes).any():
            break
    return e


def _on(e: Engine, *, connector: bool) -> int:
    run = e.vehicles.running()
    on_conn = e.vehicles.link[run] >= e.network.n_lanes
    return int(run[on_conn if connector else ~on_conn][0])


def _pair_on_a_lane(e: Engine) -> tuple[int, int]:
    """(follower, leader) on the same lane."""
    veh = e.vehicles
    run = veh.running()
    for lane in np.unique(veh.link[run]):
        hs = run[veh.link[run] == lane]
        if hs.size >= 2 and lane < e.network.n_lanes:
            hs = hs[np.argsort(veh.pos[hs])]
            return int(hs[0]), int(hs[1])
    raise AssertionError("no lane with two vehicles")


def _violates(rule: str, check: Callable[[], None], uid: int | None = None) -> None:
    with pytest.raises(InvariantViolation) as info:
        check()
    assert info.value.rule == rule
    if uid is not None:
        assert uid in info.value.uids


def test_clean_state_passes(engine: Engine) -> None:
    run = engine.vehicles.running()
    assert run.size and (engine.vehicles.link[run] >= engine.network.n_lanes).any()
    check_always(engine.vehicles, run, engine.step_count)
    check_debug(engine)


def test_i1_finite(engine: Engine, caplog: pytest.LogCaptureFixture) -> None:
    veh = engine.vehicles
    h = _on(engine, connector=False)
    veh.pos[h] = np.nan
    with caplog.at_level(logging.ERROR, logger="urbanflow.engine"):
        _violates("I1", lambda: check_always(veh, veh.running(), 5), int(veh.uid[h]))
    assert "invariant I1 violated at step 5" in caplog.text
    assert issubclass(InvariantViolation, SimulationError)


def test_i2_non_negative_speed(engine: Engine) -> None:
    veh = engine.vehicles
    h = _on(engine, connector=False)
    veh.speed[h] = -0.5
    _violates("I2", lambda: check_always(veh, veh.running(), 5), int(veh.uid[h]))


def test_i3_on_link(engine: Engine) -> None:
    veh = engine.vehicles
    h = _on(engine, connector=True)
    veh.pos[h] = engine.network.link_length[veh.link[h]] + 0.01
    _violates("I3", lambda: check_debug(engine), int(veh.uid[h]))


def test_i4_no_overlap(engine: Engine) -> None:
    veh = engine.vehicles
    follower, leader = _pair_on_a_lane(engine)
    veh.pos[follower] = veh.pos[leader] - 1.0
    _violates("I4", lambda: check_debug(engine), int(veh.uid[follower]))


def test_i6_conservation(engine: Engine) -> None:
    engine.generated += 1
    _violates("I6", lambda: check_debug(engine))
    engine.generated -= 1
    h = _on(engine, connector=False)
    engine.vehicles.link[h] = engine.network.n_links
    _violates("I6", lambda: check_debug(engine), int(engine.vehicles.uid[h]))


def test_i6_statuses_match_the_backlog(engine: Engine) -> None:
    vid = engine.commands.add_vehicle(route=["W_in", "E_out"])
    check_debug(engine)
    engine.queues.clear()  # the vehicle waits but no queue holds it
    _violates("I6", lambda: check_debug(engine))
    assert engine.vehicles.handle_of(vid) >= 0


def test_i7_route_consistency(engine: Engine) -> None:
    veh = engine.vehicles
    h = _on(engine, connector=False)
    veh.route_cursor[h] += 1
    _violates("I7", lambda: check_debug(engine), int(veh.uid[h]))
    veh.route_cursor[h] -= 1
    c = _on(engine, connector=True)
    veh.route_cursor[c] = 1  # a connector needs route[cursor + 1]
    _violates("I7", lambda: check_debug(engine), int(veh.uid[c]))


def test_i8_no_ungated_crossing(engine: Engine) -> None:
    veh = engine.vehicles
    c = _on(engine, connector=True)
    veh.committed[c] = False
    _violates("I8", lambda: check_debug(engine), int(veh.uid[c]))
    veh.committed[c] = True
    h = _on(engine, connector=False)
    crossing = Advance(
        dx=np.zeros(0),
        arrived=np.zeros(0, dtype=np.intp),
        crossed=np.array([h]),
        crossed_committed=np.array([False]),
        violations=0,
    )
    _violates("I8", lambda: check_debug(engine, crossing), int(veh.uid[h]))


def test_i11_identity(engine: Engine) -> None:
    veh = engine.vehicles
    run = veh.running()
    a, b = int(run[0]), int(run[1])
    uid_b = int(veh.uid[b])
    veh.uid[b] = veh.uid[a]
    _violates("I11", lambda: check_debug(engine))
    veh.uid[b] = uid_b
    vid = veh.ids[a]
    veh.ids[a] = "someone-else"
    _violates("I11", lambda: check_debug(engine))
    veh.ids[a] = vid
    veh.status[a] = VehicleStatus.arrived.code  # arrived but never freed
    _violates("I11", lambda: check_debug(engine))
    veh.status[a] = VehicleStatus.running.code
    check_debug(engine)


@pytest.fixture
def signalled(make_engine: Callable[..., Engine]) -> Engine:
    e = make_engine(generate("single_intersection", lanes=1))
    for _ in range(10):
        e.step()
    return e


def test_i10_signal_state_machine(signalled: Engine) -> None:
    e = signalled
    sig = e.signals
    check_debug(e)
    green = int(np.flatnonzero(sig.movement_state == SignalState.G.code)[0])
    j = int(e.network.mov_intersection[green])
    sig.movement_state[green] = SignalState.r.code  # G -> r without yellow
    _violates("I10", lambda: check_debug(e))
    sig.forced[j] = True  # ...unless a forced set_phase caused it
    check_debug(e)
    sig.forced[j] = False
    sig.yellow_zero[j] = True  # ...or the program has no yellow
    check_debug(e)
    sig.yellow_zero[j] = False
    sig.movement_state[green] = SignalState.y.code  # G -> y is the legal path
    check_debug(e)
    sig.movement_state[green] = 7  # not a signal state
    _violates("I10", lambda: check_debug(e))
    sig.movement_state[green] = SignalState.G.code
    check_debug(e)


def test_i10_passes_a_whole_signalised_run(signalled: Engine) -> None:
    for _ in range(200):  # debug checks run every step: several full cycles, no violation
        signalled.step()
    assert signalled.red_runs == 0


# --------------------------------------------------------------------------- I5 and I9
@pytest.fixture
def pair(
    make_engine: Callable[..., Engine], junction_builder: Callable[..., ScenarioBuilder]
) -> Engine:
    """Two vehicles inserted on W_in_0 (to E_out) and N_in_0 (to S_out), nothing else."""
    b = junction_builder()
    b.trip("we", 0.0, route=["W_in", "E_out"])
    b.trip("ns", 0.0, route=["N_in", "S_out"])
    e = make_engine(b.build())
    e.step()
    e.holds = None  # the hand-made states below are not the step's holds
    return e


def _onto(e: Engine, vid: str, link: str, pos: float) -> int:
    """Move ``vid`` onto a connector of its route (committed) or its exit lane."""
    net, veh = e.network, e.vehicles
    h = veh.id_to_handle[vid]
    lk = net.link_index[link]
    veh.link[h], veh.pos[h] = lk, pos
    on_conn = lk >= net.n_lanes
    veh.committed[h] = on_conn
    veh.next_conn[h] = lk if on_conn else -1
    veh.route_cursor[h] = 0 if on_conn else 1
    return h


def test_i5_zone_exclusivity(pair: Engine) -> None:
    """W->E crosses N->S at [2.1, 5.1] on W->E and [5.3, 8.3] on N->S."""
    e, veh = pair, pair.vehicles
    a = _onto(e, "we", "W_in_0->E_out_0", 4.0)  # body [-1, 4]
    b = _onto(e, "ns", "N_in_0->S_out_0", 7.0)  # body [2, 7]
    _violates("I5", lambda: check_debug(e), int(veh.uid[a]))
    veh.forced[b] = True  # a forced grant's overlap is counted in zone_conflicts instead
    check_debug(e)
    veh.forced[b] = False
    veh.pos[a] = 1.5  # before the zone: fine
    check_debug(e)
    # P4 review repro: held at the zone, capped exactly at z_in; round-off leaves the front
    # 1e-15 m past it. Within POSITION_EPS it is not inside (as for every other invariant)
    we, ns = e.network.link_index["W_in_0->E_out_0"], e.network.link_index["N_in_0->S_out_0"]
    z_in = next(z[1] for z in e.junctions.conflicts[we - e.network.n_lanes] if z[0] == ns)
    veh.pos[a] = z_in + 1e-12
    check_debug(e)
    veh.pos[a] = z_in + 1e-3
    _violates("I5", lambda: check_debug(e), int(veh.uid[a]))
    veh.pos[a] = 4.0
    # on its exit lane, a body is mapped back onto the connector it has locked
    b = _onto(e, "ns", "S_out_0", 1.0)  # rear 1 - 5 + 10.4 = 6.4 on N->S
    check_debug(e)  # no lock: the rear has passed every zone (lock_conn is released)
    veh.lock_conn[b] = e.network.link_index["N_in_0->S_out_0"]
    _violates("I5", lambda: check_debug(e), int(veh.uid[b]))


def test_i9_held_vehicles_stay_behind_their_obstacle(pair: Engine) -> None:
    e, veh, net = pair, pair.vehicles, pair.network
    conn = net.link_index["W_in_0->E_out_0"]
    a = _onto(e, "we", "W_in_0->E_out_0", 1.0)
    h = np.array([a])

    def hold(link: int, pos: float, exempt: bool = False) -> None:
        e.holds = Holds(h, np.array([link]), np.array([pos]), np.array([exempt]))

    hold(conn, 2.1)  # held at its first zone
    check_debug(e)
    veh.pos[a] = 2.5
    _violates("I9", lambda: check_debug(e), int(veh.uid[a]))
    hold(conn, 2.1, exempt=True)  # the obstacle appeared when it could not stop any more
    check_debug(e)
    lane = net.link_index["W_in_0"]
    hold(lane, float(net.link_length[lane]))  # held at the stop line, found beyond it
    _violates("I9", lambda: check_debug(e), int(veh.uid[a]))
    _onto(e, "we", "E_out_0", 3.0)
    hold(conn, 2.1)  # held at a zone, found on the exit lane
    _violates("I9", lambda: check_debug(e), int(veh.uid[a]))


def test_i5_and_i9_pass_a_saturated_uncontrolled_run(
    make_engine: Callable[..., Engine], junction_builder: Callable[..., ScenarioBuilder]
) -> None:
    b = junction_builder()
    for i, (a, z) in enumerate((("W", "E"), ("N", "S"), ("E", "N"), ("S", "W"))):
        b.flow(f"f{i}", route=[f"{a}_in", f"{z}_out"], period=4.0)
    e = make_engine(b.build())
    held = 0
    for _ in range(300):  # debug checks every step
        e.step()
        assert e.holds is not None
        held += int(e.holds.handles.size)
    assert held > 0 and e.arrived > 100
