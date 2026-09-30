"""The signal runtime state machine (plan H.2): timing, requests, events, snapshots."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urbanflow.core.events import PHASE_FORCED_BIT, EventBuffer, EventType, decode_phase_aux
from urbanflow.core.types import SIGNAL_CODE_UNSIGNALISED, SignalState, Stage
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.scenario import ScenarioBuilder
from urbanflow.signals import SignalProgram, SignalRuntime, build_programs

from .conftest import EW, EW_LEFT, EW_PLUS, NS, program_of, runtime_for

R, Y, g, G = (s.code for s in SignalState)
GREEN, YELLOW, ALL_RED = Stage.green, Stage.yellow, Stage.all_red


def _run(
    rt: SignalRuntime, j: int, steps: int, dt: float = 1.0, start: int = 0
) -> tuple[list[tuple[Stage, int]], list[tuple[int, int, int, float]]]:
    """Advance ``steps`` times; per step the (stage, phase) applied, and every
    ``phase_changed`` event as (step, aux, link, time)."""
    states, events = [], []
    buf = EventBuffer()
    for n in range(start, start + steps):
        buf.clear()
        rt.advance(dt, buf, step=n + 1, time=n * dt)
        states.append((Stage.from_code(int(rt.stage[j])), int(rt.phase[j])))
        for kind, a, lk, t, ix in zip(
            buf.type.tolist(),
            buf.aux.tolist(),
            buf.link.tolist(),
            buf.time.tolist(),
            buf.intersection.tolist(),
            strict=True,
        ):
            assert kind == EventType.phase_changed.code and ix == j
            events.append((n + 1, a, lk, t))
    return states, events


def _stages(states: list[tuple[Stage, int]]) -> list[tuple[Stage, int, int]]:
    """Run-length encoding: (stage, phase, number of steps)."""
    out: list[tuple[Stage, int, int]] = []
    for st in states:
        if out and out[-1][:2] == st:
            out[-1] = (*st, out[-1][2] + 1)
        else:
            out.append((*st, 1))
    return out


@pytest.fixture(scope="module")
def two_nodes() -> tuple[CompiledNetwork, SignalProgram]:
    """W -> J1 (signalized, one phase) -> J2 (uncontrolled) -> E."""
    b = ScenarioBuilder("two_nodes")
    b.boundary("W", (-200.0, 0.0))
    b.intersection("J1", (0.0, 0.0), kind="signalized")
    b.intersection("J2", (100.0, 0.0), kind="uncontrolled")
    b.boundary("E", (300.0, 0.0))
    b.road("W_J1", "W", "J1")
    b.road("J1_J2", "J1", "J2")
    b.road("J2_E", "J2", "E")
    scenario = b.build()
    net = compile_network(scenario)
    (prog,) = build_programs(net, scenario.resolved.network)
    return net, prog


def test_initial_state_and_unsignalised_codes(
    two_nodes: tuple[CompiledNetwork, SignalProgram],
) -> None:
    net, prog = two_nodes
    rt = runtime_for(net, prog)
    j1, j2 = net.int_index["J1"], net.int_index["J2"]
    assert [p.intersection for p in rt.programs] == [j1]
    assert rt.movement_state[net.mov_index["W_J1->J1_J2"]] == G
    assert rt.movement_state[net.mov_index["J1_J2->J2_E"]] == SIGNAL_CODE_UNSIGNALISED
    snap = rt.snapshot()
    assert (snap.phase[j1], snap.green[j1], snap.target[j1], snap.stage[j1]) == (0, 0, -1, 0)
    assert (snap.phase[j2], snap.green[j2], snap.target[j2]) == (-1, -1, -1)
    assert math.isnan(snap.min_green[j2]) and math.isnan(snap.stage_elapsed[j2])
    assert snap.min_green[j1] == 5.0 and snap.max_green[j1] == 60.0
    assert snap.stage_elapsed.dtype == np.float32 and snap.phase.dtype == np.int16
    assert snap.movement_state.dtype == np.uint8 and snap.stage.dtype == np.uint8
    with pytest.raises(Exception, match="not signalised"):
        rt.program(j2)


def test_min_green_is_honoured(ew_ns: tuple[CompiledNetwork, SignalProgram]) -> None:
    net, prog = ew_ns
    j = prog.intersection
    rt = runtime_for(net, prog)
    rt.request(j, 1)  # before any green was applied: waits for min_green = 5 s
    states, events = _run(rt, j, 12)
    assert _stages(states) == [(GREEN, 0, 5), (YELLOW, 0, 3), (ALL_RED, 0, 1), (GREEN, 1, 3)]
    assert events == [(10, 1, -1, 9.0)]  # green 1 starts in the 10th step, at t = 9 s
    assert rt.green_elapsed[j] == 3.0 and rt.stage_elapsed[j] == 3.0


@pytest.mark.parametrize(
    ("dt", "min_green", "yellow", "all_red"),
    [(1.0, 5.0, 3.0, 1.0), (0.5, 5.0, 3.0, 1.0), (0.4, 5.0, 2.5, 0.7), (0.3, 4.1, 3.0, 1.3)],
)
def test_stage_durations_are_quantised_to_steps(
    dt: float, min_green: float, yellow: float, all_red: float
) -> None:
    net, prog = program_of(
        [{"green": EW, "duration": 30}, {"green": NS, "duration": 30}],
        min_green=min_green,
        yellow=yellow,
        all_red=all_red,
    )
    j = prog.intersection
    rt = runtime_for(net, prog)
    rt.request(j, 1)

    def n(d: float) -> int:
        return math.ceil(d / dt - 1e-9)

    states, events = _run(rt, j, n(min_green) + n(yellow) + n(all_red) + 2, dt)
    assert _stages(states) == [
        (GREEN, 0, n(min_green)),
        (YELLOW, 0, n(yellow)),
        (ALL_RED, 0, n(all_red)),
        (GREEN, 1, 2),
    ]
    assert len(events) == 1 and events[0][1] == 1


def test_requests_during_a_transition_are_queued_and_the_last_wins() -> None:
    net, prog = program_of(
        [{"green": EW}, {"green": NS}, {"green": EW_LEFT}], min_green=2, yellow=2, all_red=1
    )
    j = prog.intersection
    rt = runtime_for(net, prog)
    _run(rt, j, 2)  # min green served
    rt.request(j, 1)
    states, _ = _run(rt, j, 1, start=2)
    assert states == [(YELLOW, 0)] and rt.target[j] == 1
    rt.request(j, 2)
    rt.request(j, 0)  # replaces 2
    states, events = _run(rt, j, 7, start=3)
    # the transition keeps its target; the queued request applies after min green of 1
    assert _stages(states) == [
        (YELLOW, 0, 1),
        (ALL_RED, 0, 1),
        (GREEN, 1, 2),
        (YELLOW, 1, 2),
        (ALL_RED, 1, 1),
    ]
    assert rt.target[j] == 0 and [e[1] for e in events] == [1]


def test_requesting_the_current_phase_is_a_no_op_and_cancels(
    ew_ns: tuple[CompiledNetwork, SignalProgram],
) -> None:
    net, prog = ew_ns
    j = prog.intersection
    rt = runtime_for(net, prog)
    rt.request(j, 1)
    rt.request(j, 0)  # last wins: stay in 0
    states, events = _run(rt, j, 8)
    assert states == [(GREEN, 0)] * 8 and events == [] and rt.pending[j] == -1


def test_zero_length_stages_are_skipped_in_the_same_call() -> None:
    phases = [{"green": EW}, {"green": NS}]
    for yellow, all_red, expected in (
        (0.0, 2.0, [(GREEN, 0, 5), (ALL_RED, 0, 2), (GREEN, 1, 2)]),
        (2.0, 0.0, [(GREEN, 0, 5), (YELLOW, 0, 2), (GREEN, 1, 2)]),
        (0.0, 0.0, [(GREEN, 0, 5), (GREEN, 1, 4)]),
    ):
        net, prog = program_of(phases, yellow=yellow, all_red=all_red)
        j = prog.intersection
        rt = runtime_for(net, prog)
        rt.request(j, 1)
        states, events = _run(rt, j, 9)
        assert _stages(states) == expected, (yellow, all_red)
        assert len(events) == 1


def test_immediate_switch_and_g_to_G_changes() -> None:
    net, prog = program_of([{"green": EW}, {"green": EW_PLUS}], min_green=0)
    j = prog.intersection
    k = {net.mov_ids[m]: m for m in prog.movements.tolist()}
    rt = runtime_for(net, prog)
    rt.request(j, 1)
    states, events = _run(rt, j, 1)
    assert states == [(GREEN, 1)] and events == [(1, 1, -1, 0.0)]  # no yellow, no all-red
    assert rt.movement_state[k["W_in->N_out"]] == G and rt.movement_state[k["W_in->S_out"]] == G
    # EW -> EW_LEFT: the common movements switch G <-> g when the yellow starts
    net, prog = program_of([{"green": EW}, {"green": EW_LEFT}], min_green=0)
    rt = runtime_for(net, prog)
    k = {net.mov_ids[m]: m for m in prog.movements.tolist()}
    rt.request(j, 1)
    _run(rt, j, 1)
    ms = rt.movement_state
    assert rt.stage[j] == YELLOW.code
    assert (ms[k["E_in->W_out"]], ms[k["W_in->N_out"]], ms[k["E_in->S_out"]]) == (g, G, Y)


def test_movement_states_through_a_transition(
    ew_ns: tuple[CompiledNetwork, SignalProgram],
) -> None:
    net, prog = ew_ns
    j = prog.intersection
    rt = runtime_for(net, prog)
    k = {net.mov_ids[m]: m for m in prog.movements.tolist()}
    _run(rt, j, 5)
    assert rt.state_string(j).count("G") == 2 and rt.state_string(j).count("g") == 2
    rt.request(j, 1)
    seen = []
    for n in range(5, 10):
        _run(rt, j, 1, start=n)
        seen.append(
            (int(rt.movement_state[k["W_in->E_out"]]), int(rt.movement_state[k["N_in->S_out"]]))
        )
        assert rt.state_string(j) == "".join(
            "rygG"[c] for c in rt.movement_state[prog.movements].tolist()
        )
    assert seen == [(Y, R), (Y, R), (Y, R), (R, R), (R, G)]
    assert set(rt.state_string(j)) == {"G", "r"}


def test_force_jumps_now_and_emits_a_forced_event_next_step(
    ew_ns: tuple[CompiledNetwork, SignalProgram],
) -> None:
    net, prog = ew_ns
    j = prog.intersection
    rt = runtime_for(net, prog)
    _run(rt, j, 3)
    rt.request(j, 1)  # pending, min green not reached
    rt.force(j, 1)
    assert (rt.phase[j], rt.stage[j], rt.pending[j], rt.stage_elapsed[j]) == (1, 0, -1, 0.0)
    assert np.array_equal(rt.movement_state[prog.movements], prog.phase_state[1])
    assert rt.forced[j]
    states, events = _run(rt, j, 2, start=3)
    # B.2 #25: link -1, aux = phase | PHASE_FORCED_BIT (bit 16)
    assert events == [(4, 1 | PHASE_FORCED_BIT, -1, 3.0)] and states == [(GREEN, 1)] * 2
    assert PHASE_FORCED_BIT == 1 << 16 and decode_phase_aux(events[0][1]) == (1, True)
    rt.end_step()
    assert not rt.forced[j] and np.array_equal(rt.last_state, rt.movement_state)


def test_place_sets_the_initial_state_inside_a_transition(
    ew_ns: tuple[CompiledNetwork, SignalProgram],
) -> None:
    net, prog = ew_ns
    j = prog.intersection
    rt = runtime_for(net, prog)
    rt.place(j, 0, stage=YELLOW, target=1, stage_elapsed=2.0, green_elapsed=32.0)
    tr = prog.transition(0, 1)
    assert np.array_equal(rt.movement_state[prog.movements], tr.yellow)
    assert (rt.target[j], rt.green_elapsed[j]) == (1, 32.0)
    states, events = _run(rt, j, 3)
    assert states == [(YELLOW, 0), (ALL_RED, 0), (GREEN, 1)] and events[0][:2] == (3, 1)
    rt.place(j, 1, stage_elapsed=4.0)
    assert (rt.phase[j], rt.stage[j], rt.green_elapsed[j]) == (1, GREEN.code, 4.0)
    with pytest.raises(ValueError, match="has no yellow stage"):
        rt.place(j, 1, stage=YELLOW, target=1)
    with pytest.raises(Exception, match="phase 5 does not exist"):
        rt.request(j, 5)


def test_snapshot_reports_the_target_as_green_during_a_transition(
    ew_ns: tuple[CompiledNetwork, SignalProgram],
) -> None:
    net, prog = ew_ns
    j = prog.intersection
    rt = runtime_for(net, prog)
    _run(rt, j, 5)
    rt.request(j, 1)
    _run(rt, j, 2, start=5)
    snap = rt.snapshot()
    assert (snap.phase[j], snap.green[j], snap.target[j], snap.stage[j]) == (0, 1, 1, 1)
    assert (snap.stage_elapsed[j], snap.green_elapsed[j]) == (2.0, 7.0)
    rt.reset()
    assert (rt.phase[j], rt.target[j], rt.stage[j], rt.green_elapsed[j]) == (0, -1, 0, 0.0)
    assert snap.phase[j] == 0 and snap.target[j] == 1  # snapshots are copies
