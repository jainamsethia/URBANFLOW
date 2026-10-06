"""The bundled single_intersection over its full hour (B.2 #25 permissive-left limitation).

Before end-of-green clearing and zone locks (P3), seeds 0 and 7 at dt 1 ended with a
backlog of 102 / 157 vehicles and 4 watchdog teleports each: permissive left turners in
lanes shared with straights starved behind the opposing flow. Now nobody is teleported
and the backlog stays near zero. Public API; about 4 s per seed.
"""

from __future__ import annotations

import pytest

import urbanflow
from urbanflow import Simulation


@pytest.mark.parametrize("seed", [0, 7])
def test_bundled_single_intersection_hour_has_no_teleports(seed: int) -> None:
    scenario = urbanflow.Scenario.load(urbanflow.bundled("single_intersection"))
    result = Simulation(scenario, seed=seed, dt=1.0).run()
    s = result.summary
    assert result.sim_time == 3600.0
    assert s["vehicles.teleported"] == 0
    assert s["vehicles.backlog"] <= 20  # was 102 / 157
    assert s["vehicles.arrived"] > 2250  # was 2155 / 2147 (demand: 2400 veh/h)
