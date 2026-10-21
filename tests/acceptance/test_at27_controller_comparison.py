"""AT-27: grid_4x4 x {fixed_time, actuated, max_pressure, webster} x 3 seeds all complete and
give a comparison table; no controller is asserted to win (plan V)."""

from __future__ import annotations

import math

import pytest

import urbanflow
from urbanflow.experiments.compare import compare_controllers

pytestmark = pytest.mark.acceptance

CONTROLLERS = ["fixed_time", "actuated", "max_pressure", "webster"]


def test_four_controllers_three_seeds() -> None:
    scenario = urbanflow.Scenario.load(urbanflow.bundled("grid_4x4"))
    out = compare_controllers(scenario, CONTROLLERS, seeds=3, duration=300, max_workers=1)
    assert out["controllers"] == CONTROLLERS and out["seeds"] == [0, 1, 2]
    travel = next(m for m in out["metrics"] if m["metric"] == "travel_time.mean")
    for c in CONTROLLERS:
        stats = travel["by_controller"][c]
        assert len(stats["values"]) == 3 and all(math.isfinite(v) for v in stats["values"])
    assert set(travel["vs_baseline"]) == set(CONTROLLERS[1:])
    assert all(travel["vs_baseline"][c]["n_pairs"] == 3 for c in CONTROLLERS[1:])
