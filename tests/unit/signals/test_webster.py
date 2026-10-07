from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from urbanflow import Simulation, generate
from urbanflow.core import constants as C
from urbanflow.signals.controllers import Webster, WebsterParams, webster_timing


def test_textbook_example() -> None:
    # Y = 0.5, L = 10 s: C0 = (1.5*10 + 5) / 0.5 = 40 s; greens split (C - L) = 30 s by y.
    cycle, green = webster_timing(np.array([0.3, 0.2]), 10.0, np.array([5.0, 5.0]), (30, 180))
    assert cycle == pytest.approx(40.0)
    assert green.tolist() == pytest.approx([18.0, 12.0])


def test_no_demand_and_oversaturation() -> None:
    cycle, green = webster_timing(np.zeros(2), 8.0, np.array([5.0, 5.0]), (30, 180))
    assert cycle == 30 and green.tolist() == [11.0, 11.0]
    cycle, green = webster_timing(np.array([0.6, 0.4]), 8.0, np.array([5.0, 5.0]), (30, 180))
    assert cycle == 180 and green.tolist() == pytest.approx([103.2, 68.8])
    _, green = webster_timing(np.array([0.3, 0.001]), 8.0, np.array([5.0, 7.0]), (30, 180))
    assert green[1] == 7.0  # min-green floor


def test_bad_bounds() -> None:
    with pytest.raises(ValidationError):
        WebsterParams(cycle_bounds=(90, 60))


def test_greens_follow_the_demand() -> None:
    scenario = generate(
        "single_intersection", approach_rates={"N": 700, "S": 700, "E": 200, "W": 200}
    )
    sim = Simulation(scenario, controllers={"*": "webster"}, seed=0)
    ctrl = sim._engine.controllers[next(iter(sim._engine.controllers))]
    assert isinstance(ctrl, Webster) and ctrl.timing is not None
    ew, ns = ctrl.timing.duration.tolist()  # p0 serves E/W, p1 serves N/S
    # lane flows 350 vs 100 veh/h: (30 - 8) * (0.194, 0.056) / 0.25 -> 17.1 s, floor 5 s
    assert ns == pytest.approx(22 * 350 / 450) and ew == 5.0
    assert ctrl.cycle == C.WEBSTER_CYCLE_MIN
    view = sim.signals["J"]
    assert view.controller == "webster" and view.remaining == 5.0
    assert sim.run(duration=900).summary["vehicles.arrived"] > 0


def test_webster_beats_even_split_on_unbalanced_demand() -> None:
    scenario = generate(
        "single_intersection",
        approach_rates={"N": 800, "S": 800, "E": 200, "W": 200},
        green=30,
        duration=1800,
    )

    def delay(controller: str) -> float:
        sim = Simulation(scenario, controllers={"*": controller}, seed=1)
        return float(sim.run().summary["delay.mean"])

    assert delay("webster") < delay("fixed_time")
