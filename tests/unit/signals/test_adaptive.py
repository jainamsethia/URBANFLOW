"""max_pressure and actuated controllers (plan H.4)."""

from __future__ import annotations

import numpy as np
import pytest

from urbanflow import Simulation, generate
from urbanflow.signals.controllers import max_pressure_choice


def test_max_pressure_choice_hand_computed() -> None:
    x_in = np.array([10, 4, 0])
    x_out = np.array([2, 6, 0])
    phase_conn = np.array([[True, False, False], [False, True, True]])
    assert max_pressure_choice(x_in, x_out, phase_conn, current=1) == 0  # 8 vs -2
    assert max_pressure_choice(x_in, x_out, phase_conn, current=0, exclude=0) == 1


def test_max_pressure_ties_keep_current_then_lowest() -> None:
    phase_conn = np.eye(3, dtype=bool)
    x = np.array([3, 3, 3])
    zero = np.zeros(3, dtype=int)
    assert max_pressure_choice(x, zero, phase_conn, current=2) == 2
    assert max_pressure_choice(x, zero, phase_conn, current=2, exclude=2) == 0


def _ns_green_share(controller: str) -> float:
    """Fraction of steps (after the first 60 s) in which N_in->S_out is green."""
    scen = generate("single_intersection", approach_rates={"E": 1e-3, "W": 1e-3}, duration=600)
    sim = Simulation(scen, controllers={"*": controller}, seed=3, debug_checks=True)
    green = 0
    total = 0
    while not sim.done:
        sim.step()
        if sim.time > 60:
            total += 1
            green += sim.signals["J"].movement_states["N_in->S_out"] in ("G", "g")
    return green / total


@pytest.mark.parametrize("controller", ["actuated", "max_pressure"])
def test_adaptive_controllers_serve_the_loaded_approach(controller: str) -> None:
    fixed = _ns_green_share("fixed_time")
    adaptive = _ns_green_share(controller)
    assert fixed == pytest.approx(0.5, abs=0.1)
    assert adaptive > 0.85  # no east-west demand: the green stays (or rests) north-south


def test_controllers_change_results_but_not_demand() -> None:
    scen = generate("single_intersection", duration=900)
    results = {
        c: Simulation(scen, controllers={"*": c}, seed=1).run()
        for c in ("fixed_time", "actuated", "max_pressure")
    }
    generated = {r.summary["vehicles.generated"] for r in results.values()}
    assert len(generated) == 1  # common random numbers: identical demand
    for name, r in results.items():
        assert r.controllers["J"]["type"] == name
