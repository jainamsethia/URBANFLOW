"""AT-38: variants of one scenario and seed run concurrently in worker processes end in the
same state as the same simulations run alone (plan V)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import urbanflow
from urbanflow import Simulation
from urbanflow.experiments.runner import ExperimentRunner
from urbanflow.experiments.spec import ExperimentStatus, parse_experiment
from urbanflow.experiments.store import ExperimentStore

pytestmark = pytest.mark.acceptance

CONTROLLERS = {"scenario": None, "mp": "max_pressure", "act": "actuated"}


def test_concurrent_runs_equal_solo_runs(tmp_path: Path) -> None:
    store = ExperimentStore.open(f"sqlite:///{(tmp_path / 'uf.db').as_posix()}")
    runner = ExperimentRunner(tmp_path, store, max_workers=3)
    variants = [
        {"name": n} if c is None else {"name": n, "policy": {"kind": "controller", "controller": c}}
        for n, c in CONTROLLERS.items()
    ]
    spec = parse_experiment(
        {
            "name": "at38",
            "scenario": {"bundled": "grid_3x3"},
            "base_config": {"duration": 120},
            "variants": variants,
            "seeds": [4],
        }
    )
    eid = runner.submit(spec)
    assert runner.run(eid) is ExperimentStatus.completed
    scenario = urbanflow.Scenario.load(urbanflow.bundled("grid_3x3"))
    for run in store.list_runs(eid):
        result = json.loads((Path(run.artifacts_dir) / "result.json").read_text())
        controller = CONTROLLERS[run.variant]
        solo = Simulation(
            scenario,
            controllers=None if controller is None else {"*": controller},
            seed=4,
            duration=120,
        )
        assert solo.run().state_digest == result["state_digest"], run.variant
