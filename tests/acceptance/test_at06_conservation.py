"""AT-06: conservation holds every step of the AT-05 run; 0 invariant violations (V)."""

from __future__ import annotations

import pytest

from urbanflow import Simulation, generate

pytestmark = pytest.mark.acceptance


def test_at06_conservation() -> None:
    scenario = generate("single_intersection", kind="uncontrolled")
    # debug_checks: invariants I3, I4, I6 (conservation), I7, I8, I11 raise every step
    sim = Simulation(scenario, seed=0, duration=600, debug_checks=True)
    while not sim.done:
        sim.step()
        s = sim.metrics.summary()
        assert s["vehicles.generated"] == (
            s["vehicles.backlog"]
            + s["vehicles.en_route"]
            + s["vehicles.arrived"]
            + s["vehicles.removed"]
        )
    result = sim.get_results()
    assert result.steps == 600 and not result.interrupted
    assert result.summary["vehicles.arrived"] > 0
    assert result.summary["vehicles.teleported"] == 0
