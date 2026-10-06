"""Signalised intersections end to end, at the engine level (plan F.3 signal step, H.2-H.5).

The facade-level acceptance tests (``tests/acceptance``, AT-08 to AT-11) own the AT
scenarios; this file checks what they cannot see: the ``red_runs``, ``forced_commits`` and
safety-cap counters, the yellow dilemma branch of F.3 step 1 inside the pipeline, the F.1
lookahead rule of B.2 #25, and the I10 exemptions. Every engine here runs with
``debug_checks=True`` (I3, I4, I6-I8, I10, I11 every step).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from urbanflow import bundled, generate
from urbanflow.core import constants as C
from urbanflow.core.events import EventType, decode_phase_aux
from urbanflow.core.types import SignalState, Stage
from urbanflow.engine import Engine
from urbanflow.network.conflicts import ConflictKind
from urbanflow.scenario import Scenario, ScenarioBuilder

pytestmark = pytest.mark.integration

MakeEngine = Callable[..., Engine]
Builder = Callable[..., ScenarioBuilder]
DET = {"speed_factor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}}
EXITED = EventType.vehicle_exited_link.code
PHASE = EventType.phase_changed.code
J = 0
GREEN, YELLOW, ALL_RED = Stage.green, Stage.yellow, Stage.all_red


def _stage(e: Engine) -> tuple[Stage, int]:
    return Stage.from_code(int(e.signals.stage[J])), int(e.signals.phase[J])


def _red_phase(e: Engine, j: int, movement: str) -> int:
    prog = e.signals.program(j)
    k = prog.movements.tolist().index(e.network.mov_index[movement])
    return next(p for p in range(prog.n_phases) if prog.phase_state[p, k] == SignalState.r.code)


# --------------------------------------------------------------------------- counters
def test_red_hold_and_discharge_run_no_red_and_force_nobody(make_engine: MakeEngine) -> None:
    """A W -> E stream held at red for 60 s by a manual hold, then released (AT-09/10's
    setting): nobody runs the red, nobody is force-committed, the G.2 cap never fails."""
    b = ScenarioBuilder("queue", dt=1.0)
    b.intersection("J", (0.0, 0.0), kind="signalized")
    for name, (dx, dy) in {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}.items():
        b.boundary(name, (dx * 400.0, dy * 400.0))
        b.road(f"{name}_in", name, "J")
        b.road(f"{name}_out", "J", name)
    b.flow("we", route=["W_in", "E_out"], rate=1800.0, count=40)
    e = make_engine(b.build())
    we = e.network.mov_index["W_in->E_out"]
    ns = _red_phase(e, J, "W_in->E_out")
    e.commands.hold_phase("J", ns)
    red_steps = 0
    while red_steps < 60:
        e.step()
        red_steps += int(e.signals.movement_state[we] == SignalState.r.code)
    lane = e.network.link_index["W_in_0"]
    run = e.vehicles.running()
    assert np.count_nonzero(e.vehicles.link[run] == lane) >= 20  # a queue formed
    assert (e.red_runs, e.forced_commits, e.safety_cap_violations) == (0, 0, 0)
    e.commands.hold_phase("J", 1 - ns)
    while e.vehicles.running().size or e.backlog:
        e.step()
        assert e.time < 1000.0, "the queue did not clear"
    assert e.arrived == 40 and e.teleported == 0
    assert (e.red_runs, e.forced_commits, e.safety_cap_violations) == (0, 0, 0)


def test_yellow_dilemma_branch_in_the_pipeline(
    make_engine: MakeEngine, junction_builder: Builder
) -> None:
    """F.3 step 1 end to end. An uncommitted 13.9 m/s vehicle 10 m from the line cannot
    arise in green (it commits ~56 m out), so the state is set by hand: at yellow onset
    "near" (needs 9.7 m/s^2 > YELLOW_MAX_DECEL) is a dilemma vehicle and commits without
    being forced; "far" (60 m, 1.6 m/s^2) stops 0.5 m before the line."""
    b = junction_builder(kind="signalized", lanes=2)
    e = make_engine(b.build(), controllers={J: "external"})
    net, veh, sig = e.network, e.vehicles, e.signals
    e.commands.add_vehicle(route=["W_in", "E_out"], id="near", depart_lane=0)
    e.commands.add_vehicle(route=["W_in", "E_out"], id="far", depart_lane=1)
    e.step()
    while sig.green_elapsed[J] < sig.programs[0].min_green[0]:
        e.step()
    assert _stage(e) == (GREEN, 0)
    near, far = veh.handle_of("near"), veh.handle_of("far")
    lanes = {"near": net.link_index["W_in_0"], "far": net.link_index["W_in_1"]}
    length = float(net.link_length[lanes["near"]])
    for h, d in ((near, 10.0), (far, 60.0)):  # both uncommitted, at 13.9 m/s
        veh.pos[h], veh.speed[h], veh.committed[h] = length - d, 13.9, False
    e.commands.request_phase("J", 1)  # the yellow starts in the next step
    crossed: list[str] = []
    far_gap = []
    for _ in range(25):
        e.step()
        ev = e.events
        crossed += [veh.ids[h] for h in ev.handle[ev.type == EXITED].tolist()]
        if veh.link[far] == lanes["far"]:
            far_gap.append(length - float(veh.pos[far]))
    assert "near" in crossed and "far" not in crossed
    assert veh.speed[far] < 1e-6 and 0.4 <= far_gap[-1] <= 0.6
    assert (e.red_runs, e.forced_commits, e.safety_cap_violations) == (0, 0, 0)


# --------------------------------------------------------------------------- F.1 lookahead
@pytest.mark.parametrize("gap_at_onset", [21.8, 24.6, 27.3])
def test_no_hard_braking_for_a_yellow_beyond_a_short_link(
    make_engine: MakeEngine, gap_at_onset: float
) -> None:
    """Review repro (B.2 #25): W -> J1 (uncontrolled) -> J2 (signalised) with a ~22 m link
    between them; J2's yellow starts while a 13.89 m/s car is on J1's connector,
    ``gap_at_onset`` m before J2's line (stopping would need 3.5-4.4 m/s^2, above
    YELLOW_MAX_DECEL). The lookahead no longer brakes it at b_emerg: it drives on, commits
    as a dilemma vehicle and crosses J2 in the yellow."""
    b = ScenarioBuilder("short_link", dt=0.2)
    b.vehicle_type("det", **DET)
    b.boundary("W", (-300.0, 0.0))
    b.intersection("J1", (0.0, 0.0), kind="uncontrolled")
    b.intersection("J2", (32.0, 0.0), kind="signalized")
    b.boundary("E", (332.0, 0.0))
    b.boundary("N", (32.0, 300.0))
    b.boundary("S", (32.0, -300.0))
    b.road("W_J1", "W", "J1")
    b.road("J1_J2", "J1", "J2")
    b.road("J2_E", "J2", "E")
    b.road("N_J2", "N", "J2")
    b.road("J2_S", "J2", "S")
    b.trip("car", 0.0, route=["W_J1", "J1_J2", "J2_E"], vehicle_type="det")
    e = make_engine(b.build(), dt=0.2)
    net, veh, sig = e.network, e.vehicles, e.signals
    j2 = net.int_index["J2"]
    e.commands.set_controller("J2", "external")
    movement = net.mov_index["J1_J2->J2_E"]
    red = _red_phase(e, j2, "J1_J2->J2_E")
    link = net.link_index["J1_J2_0"]
    h, stop = -1, 0.0
    while True:  # until the car is on J1's connector, gap_at_onset m before J2's line
        e.step()
        assert e.time < 60.0
        h = veh.id_to_handle.get("car", -1)
        if h < 0 or veh.link[h] < net.n_lanes:
            continue
        stop = float(net.link_length[veh.link[h]] - veh.pos[h] + net.link_length[link])
        if stop <= gap_at_onset:
            break
    assert stop > gap_at_onset - 13.89 * 0.2 and veh.speed[h] == pytest.approx(13.89, abs=0.01)
    assert veh.speed[h] ** 2 / (2 * stop) > C.YELLOW_MAX_DECEL
    e.commands.request_phase("J2", red)  # the yellow starts in the next step
    states, accel = [], []
    while veh.route_cursor[h] < 2:  # until it enters J2_E
        e.step()
        states.append(SignalState.from_code(int(sig.movement_state[movement])))
        accel.append(float(veh.accel[h]))
        assert e.time < 60.0
    assert states[0] is SignalState.y and states[-1] is SignalState.y  # crossed in the yellow
    assert min(accel) > -C.IDM_DECEL  # no hard braking, let alone b_emerg
    assert (e.red_runs, e.forced_commits, e.safety_cap_violations) == (0, 0, 0)


# --------------------------------------------------------------------------- I10 exemptions
def test_i10_exempts_yellow_zero_and_forced_jumps(make_engine: MakeEngine) -> None:
    """G -> r without a yellow in between is an I10 violation unless the program has
    yellow = 0 or a forced ``set_phase`` caused it; the debug checks run every step."""
    e = make_engine(generate("single_intersection", yellow=0.0, demand_rate=600))
    before = e.signals.movement_state.copy()
    direct = 0
    for _ in range(200):
        e.step()
        now = e.signals.movement_state
        direct += int(
            np.count_nonzero((before >= SignalState.g.code) & (now == SignalState.r.code))
        )
        before = now.copy()
    assert direct > 0 and e.signals.yellow_zero[J]
    e = make_engine(Scenario.load(bundled("single_intersection")))
    for _ in range(10):
        e.step()
    e.commands.set_phase("J", 1)  # losing movements go G -> r at once
    e.step()
    assert _stage(e) == (GREEN, 1) and not e.signals.yellow_zero[J]


# --------------------------------------------------------------------------- g vs G
def _left_vs_straight(
    make_engine: MakeEngine, junction_builder: Builder, kind: str, t_straight: float
) -> tuple[dict[str, float], list[bool]]:
    """A permissive left turn E -> S (arrives first) and the opposing straight W -> E."""
    b = junction_builder(kind=kind)
    b.vehicle_type("det", **DET)
    b.trip("left", 0.0, route=["E_in", "S_out"], vehicle_type="det")
    b.trip("straight", t_straight, route=["W_in", "E_out"], vehicle_type="det")
    e = make_engine(b.build(), dt=0.5)
    net, veh = e.network, e.vehicles
    conns = {
        "left": net.link_index["E_in_0->S_out_0"],
        "straight": net.link_index["W_in_0->E_out_0"],
    }
    k = next(
        k
        for k in range(net.n_conflicts)
        if {int(net.conf_a[k]), int(net.conf_b[k])} == set(conns.values())
    )
    assert net.conf_kind[k] == ConflictKind.crossing
    zone = {
        n: (net.conf_zone_a if net.conf_a[k] == c else net.conf_zone_b)[k] for n, c in conns.items()
    }
    travel: dict[str, float] = {}
    both: list[bool] = []
    for _ in range(int((t_straight + 60.0) / 0.5)):  # both have arrived by then
        e.step()
        inside = []
        for name, c in conns.items():
            h = veh.id_to_handle.get(name)
            if h is None or not veh.active[h]:
                continue
            lk, pos = int(veh.link[h]), float(veh.pos[h])
            if lk == net.conn_to_lane[c - net.n_lanes]:
                pos += float(net.link_length[c])
            elif lk != c:
                continue
            inside.append(pos - float(veh.length[h]) < zone[name][1] and pos > zone[name][0])
        both.append(len(inside) == 2 and all(inside))
        ev = e.events
        arrived = ev.type == EventType.vehicle_arrived.code
        for h, t in zip(ev.handle[arrived].tolist(), ev.time[arrived].tolist(), strict=True):
            travel[str(veh.ids[h])] = t - float(veh.insert_time[h])
    assert set(travel) == {"left", "straight"} and e.red_runs == 0
    return travel, both


def test_permissive_g_yields_to_the_opposing_protected_G(
    make_engine: MakeEngine, junction_builder: Builder
) -> None:
    # the free reference runs one 68 s cycle (30/3/1, two phases) later: the same signal
    # state (W -> E green on arrival), but the left turner is long gone
    free, _ = _left_vs_straight(make_engine, junction_builder, "signalized", 6.0 + 68.0)
    signal, both = _left_vs_straight(make_engine, junction_builder, "signalized", 6.0)
    assert signal["straight"] == pytest.approx(free["straight"], abs=0.1)  # G is not delayed
    assert signal["left"] > free["left"] + 3.0  # g waits for it
    assert not any(both)
    # the same timing at an uncontrolled junction: first come, first served
    free, _ = _left_vs_straight(make_engine, junction_builder, "uncontrolled", 60.0)
    fcfs, both = _left_vs_straight(make_engine, junction_builder, "uncontrolled", 6.0)
    assert fcfs["left"] == pytest.approx(free["left"], abs=0.1)
    assert fcfs["straight"] > free["straight"] + 3.0 and not any(both)


# --------------------------------------------------------------------------- 1800 s run
def test_signalised_single_intersection_1800_s(make_engine: MakeEngine) -> None:
    """The bundled scenario: 0 invariant violations (debug checks), 0 red runs, and two
    engines with the same seed are identical after every step."""
    scenario = Scenario.load(bundled("single_intersection"))
    a, b = make_engine(scenario, seed=7), make_engine(scenario, seed=7)
    changes = []
    for _ in range(1800):
        a.step()
        b.step()
        ra, rb = a.vehicles.running(), b.vehicles.running()
        assert np.array_equal(ra, rb)
        for col in ("uid", "link", "pos", "speed", "committed", "held"):
            assert np.array_equal(getattr(a.vehicles, col)[ra], getattr(b.vehicles, col)[rb]), col
        assert np.array_equal(a.signals.movement_state, b.signals.movement_state)
        assert np.array_equal(a.events.type, b.events.type)
        ev = a.events
        changes += [decode_phase_aux(x) for x in ev.aux[ev.type == PHASE].tolist()]
        assert (ev.link[ev.type == PHASE] == -1).all()
    assert a.red_runs == 0 and a.arrived > 500
    # a green starts every 34 s of the 30/3/1 cycle: at 34, 68, ..., 1768 s (not forced)
    assert changes == [(1, False), (0, False)] * 26
    assert a.step_count == 1800 and (a.generated, a.arrived) == (b.generated, b.arrived)


# --------------------------------------------------------------------------- end of green
def test_end_of_green_clearing_serves_a_starving_permissive_left(
    make_engine: MakeEngine, junction_builder: Builder
) -> None:
    """B.2 #25 limitation: opposing W -> E traffic every 3 s never leaves the gap a stopped
    E -> S left turner needs, so it waits at the line through the green; in the yellow
    that ends its g it commits (at most one per lane and yellow) and clears on its zone
    locks. Debug checks (I5, I9) run every step."""
    b = junction_builder(kind="signalized", duration=600.0)
    b.signal("J", template="two_phase", green=30.0, yellow=3.0, all_red=1.0)
    b.vehicle_type("det", **DET)
    b.flow("we", route=["W_in", "E_out"], period=3.0, vehicle_type="det")
    b.flow("left", route=["E_in", "S_out"], period=80.0, begin=5.0, vehicle_type="det")
    e = make_engine(b.build())
    veh, net = e.vehicles, e.network
    lane = net.link_index["E_in_0"]
    states: list[tuple[Stage, int]] = []
    eligible = -1  # uid held at the line in the last green step
    for _ in range(600):
        on_lane = veh.running()[veh.link[veh.running()] == lane]
        waiting = set(on_lane[~veh.committed[on_lane]].tolist())
        e.step()
        committed = {h for h in waiting if veh.active[h] and veh.committed[h]}
        stage = Stage.from_code(int(e.signals.stage[J]))
        states.append((stage, len(committed)))
        if stage is YELLOW:  # only the vehicle waiting since before the yellow sneaks
            assert {int(veh.uid[h]) for h in committed} <= {eligible}
        else:
            eligible = int(e.sneakers[lane])
            if eligible >= 0:  # held at its line in this green step
                h = veh.id_to_handle[veh.uid_to_id[eligible]]
                assert veh.held[h] and veh.link[h] == lane
    yellow_commits = [n for stage, n in states if stage is YELLOW]
    assert sum(yellow_commits) >= 6 and max(yellow_commits) <= C.SNEAKERS_PER_PHASE
    assert e.teleported == 0 and e.zone_conflicts == 0 and e.red_runs == 0
    # 8 left turners (one per 80 s, fewer than the 68 s cycles): all but the last served
    still = [vid for vid in veh.id_to_handle if vid.startswith("left.")]
    assert len(still) <= 1, still
