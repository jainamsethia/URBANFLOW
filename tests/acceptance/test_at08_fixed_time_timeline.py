"""AT-08: a fixed-time 30/3/1 program seen through ``sim.signals`` (plan section V, H.2, H.4).

The bundled junction runs two phases, 30 s green each, 3 s yellow and 1 s all-red, with
traffic and every invariant checked. Step by step, ``sim.signals["J"]`` shows G for 30 s,
then y on the movements losing green for 3 s, then all-red for 1 s, then the next phase.
"""

from __future__ import annotations

import pytest

from urbanflow import Scenario, Simulation, bundled
from urbanflow.core.types import Stage

pytestmark = pytest.mark.acceptance

GREEN, YELLOW, ALL_RED = 30, 3, 1  # s = steps at dt = 1


def _expected(sim: Simulation) -> list[tuple[Stage, int, int, float, str]]:
    """Per step of one 68 s cycle: (stage, phase, target, remaining, state string)."""
    order = list(sim.signals["J"].movement_states)
    phases = sim.signals.phases("J")
    out = []
    for p, phase in enumerate(phases):
        q = (p + 1) % len(phases)
        green = "".join(phase.green.get(m, "r") for m in order)
        yellow = "".join("y" if m in phase.green else "r" for m in order)
        out += [(Stage.green, p, -1, GREEN - k, green) for k in range(1, GREEN + 1)]
        out += [(Stage.yellow, p, q, YELLOW - k, yellow) for k in range(1, YELLOW + 1)]
        out += [(Stage.all_red, p, q, ALL_RED - k, "r" * len(order)) for k in range(1, ALL_RED + 1)]
    return out


def test_at08_fixed_time_timeline() -> None:
    sim = Simulation(Scenario.load(bundled("single_intersection")), debug_checks=True)
    cycle = _expected(sim)
    assert len(cycle) == 2 * (GREEN + YELLOW + ALL_RED) == 68
    assert cycle[0][4] == "GgGrrrrrrGgG" and cycle[GREEN][4] == "yyyrrrrrryyy"
    shown = []
    for _ in range(3 * len(cycle)):
        sim.step()
        view = sim.signals["J"]
        shown.append((view.stage, view.phase_index, view.target, view.remaining, view.state_string))
        assert view.cycle == 68.0 and view.controller == "fixed_time" and not view.held
        assert view.phase_id == f"p{view.phase_index}"
    assert shown == cycle * 3
    assert sim.get_results().summary["vehicles.arrived"] > 0  # with traffic throughout
