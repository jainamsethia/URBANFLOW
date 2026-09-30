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
