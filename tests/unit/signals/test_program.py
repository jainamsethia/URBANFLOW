"""Signal programs: compilation from the spec, transitions and the conflict matrix (H.1)."""

from __future__ import annotations

import numpy as np
import pytest

from urbanflow import bundled, generate
from urbanflow.core.errors import NotFoundError
from urbanflow.core.types import SignalState
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.network.conflicts import ConflictKind
from urbanflow.scenario import Scenario
from urbanflow.signals import SignalProgram, build_programs

from .conftest import EW, EW_LEFT, EW_PLUS, NS, program_of

R, Y, g, G = (s.code for s in SignalState)


def _state(prog: SignalProgram, net: CompiledNetwork, phase: int) -> dict[str, int]:
    return {
        net.mov_ids[m]: int(c)
        for m, c in zip(prog.movements.tolist(), prog.phase_state[phase].tolist(), strict=True)
    }


def test_from_spec(ew_ns: tuple[CompiledNetwork, SignalProgram]) -> None:
    net, prog = ew_ns
    j = net.int_index["J"]
    assert prog.intersection == j
    assert prog.movements.tolist() == np.flatnonzero(net.mov_intersection == j).tolist()
    assert prog.phase_ids == ("EW", "NS") and (prog.n_phases, prog.n_movements) == (2, 12)
    assert prog.phase_state.dtype == np.uint8 and prog.phase_state.shape == (2, 12)
    code = {"G": G, "g": g}
    for p, green in enumerate((EW, NS)):
        state = _state(prog, net, p)
        assert {m: c for m, c in state.items() if c} == {m: code[s] for m, s in green.items()}
    assert prog.duration.tolist() == [30.0, 30.0]
    assert prog.min_green.tolist() == [5.0, 5.0] and prog.max_green.tolist() == [60.0, 60.0]
    assert (prog.yellow, prog.all_red, prog.initial_phase) == (3.0, 1.0, 0)
    assert prog.controller.type == "fixed_time"
    assert not prog.phase_state.flags.writeable and not prog.conflict_mm.flags.writeable


def test_per_phase_green_bounds_inherit_from_the_signal() -> None:
    _, prog = program_of(
        [
            {"green": EW, "duration": 20, "min_green": 8},
            {"green": NS, "duration": 40, "max_green": 90},
        ],
        min_green=4,
        max_green=50,
        yellow=2.5,
        all_red=0,
        initial_phase=1,
    )
    assert prog.phase_ids == ("p0", "p1")
    assert prog.min_green.tolist() == [8.0, 4.0] and prog.max_green.tolist() == [50.0, 90.0]
    assert prog.duration.tolist() == [20.0, 40.0]
    assert (prog.yellow, prog.all_red, prog.initial_phase) == (2.5, 0.0, 1)


def test_conflict_matrix(ew_ns: tuple[CompiledNetwork, SignalProgram]) -> None:
    net, prog = ew_ns
    k = {net.mov_ids[m]: i for i, m in enumerate(prog.movements.tolist())}
    cm = prog.conflict_mm
    assert cm.dtype == bool and np.array_equal(cm, cm.T)
    assert cm[k["W_in->E_out"], k["N_in->S_out"]]  # crossing straights
    assert cm[k["W_in->E_out"], k["N_in->E_out"]]  # merging into E_out
    assert cm[k["W_in->E_out"], k["E_in->S_out"]]  # opposing left turn crosses
    assert not cm[k["W_in->E_out"], k["E_in->W_out"]]  # opposing straights never meet
    assert not cm[k["W_in->E_out"], k["W_in->S_out"]]  # diverging only
    # brute force over the connector conflict table
    expected = np.zeros_like(cm)
    for a, b, kind in zip(net.conf_a, net.conf_b, net.conf_kind, strict=True):
        if kind in (ConflictKind.crossing, ConflictKind.merging):
            ma, mb = k[net.mov_ids[net.link_movement[a]]], k[net.mov_ids[net.link_movement[b]]]
            expected[ma, mb] = expected[mb, ma] = True
    assert np.array_equal(cm, expected)


def test_transition_with_intergreen(ew_ns: tuple[CompiledNetwork, SignalProgram]) -> None:
    net, prog = ew_ns
    tr = prog.transition(0, 1)
    ids = [net.mov_ids[m] for m in prog.movements.tolist()]

    def named(mask: np.ndarray) -> set[str]:
        return {m for m, x in zip(ids, mask.tolist(), strict=True) if x}

    assert named(tr.losing) == set(EW) and named(tr.gaining) == set(NS)
    assert named(tr.common) == set() and not tr.immediate
    yellow = {m: int(s) for m, s in zip(ids, tr.yellow.tolist(), strict=True) if s}
    assert yellow == dict.fromkeys(EW, Y)
    assert not tr.all_red.any()  # losing r, gaining still r
    assert tr.yellow.dtype == np.uint8 and tr.all_red.dtype == np.uint8


def test_g_to_G_switches_at_the_start_of_the_transition() -> None:
    net, prog = program_of([{"green": EW}, {"green": EW_LEFT}])
    k = {net.mov_ids[m]: i for i, m in enumerate(prog.movements.tolist())}
    tr = prog.transition(0, 1)
    assert tr.losing[k["E_in->S_out"]] and tr.losing.sum() == 1 and not tr.gaining.any()
    assert tr.common[[k["W_in->E_out"], k["E_in->W_out"], k["W_in->N_out"]]].all()
    for states in (tr.yellow, tr.all_red):  # common movements take their q state at once
        assert states[k["E_in->W_out"]] == g and states[k["W_in->N_out"]] == G
        assert states[k["W_in->E_out"]] == G
    assert tr.yellow[k["E_in->S_out"]] == Y and tr.all_red[k["E_in->S_out"]] == R


def test_immediate_switch_when_nothing_loses_green() -> None:
    net, prog = program_of([{"green": EW}, {"green": EW_PLUS}])
    tr = prog.transition(0, 1)
    assert tr.immediate and not tr.losing.any()
    k = {net.mov_ids[m]: i for i, m in enumerate(prog.movements.tolist())}
    assert tr.gaining[k["W_in->S_out"]] and tr.gaining.sum() == 1
    assert not prog.transition(1, 0).immediate  # EW_PLUS -> EW loses W_in->S_out
    assert prog.transition(0, 0).immediate


def test_phase_index(ew_ns: tuple[CompiledNetwork, SignalProgram]) -> None:
    _, prog = ew_ns
    assert prog.phase_index("NS") == 1 and prog.phase_index(0) == 0
    with pytest.raises(NotFoundError, match='unknown phase "NSS" \\(did you mean "NS"\\?\\)'):
        prog.phase_index("NSS")
    with pytest.raises(NotFoundError, match=r"phase 2 does not exist \(2 phases, 0-1\)"):
        prog.phase_index(2)


def test_build_programs_follows_int_program() -> None:
    scenario = Scenario.load(bundled("single_intersection"))
    net = compile_network(scenario)
    (prog,) = build_programs(net, scenario.resolved.network)
    assert net.int_program[prog.intersection] == 0 and prog.phase_ids == ("p0", "p1")
    unsignalised = generate("single_intersection", kind="priority")
    assert build_programs(compile_network(unsignalised), unsignalised.resolved.network) == ()
