"""AT-36: an experiment of 3 variants x 2 seeds on 2 workers completes; a run whose worker
process dies (killed whenever it starts) is marked failed with ``error.txt`` and the others
complete (plan V)."""

from __future__ import annotations

import contextlib
import os
import signal
from pathlib import Path

import pytest

from urbanflow.experiments.runner import WORKER_DIED, ExperimentRunner, ProgressEvent
from urbanflow.experiments.spec import ExperimentStatus, RunStatus, parse_experiment
from urbanflow.experiments.store import ExperimentStore

pytestmark = pytest.mark.acceptance


def test_crashed_run_is_failed_and_the_others_complete(tmp_path: Path) -> None:
    store = ExperimentStore.open(f"sqlite:///{(tmp_path / '.urbanflow' / 'uf.db').as_posix()}")
    runner = ExperimentRunner(tmp_path, store, max_workers=2)
    spec = parse_experiment(
        {
            "name": "at36",
            "scenario": {"bundled": "single_intersection"},
            "base_config": {"duration": 120},
            "variants": [
                {"name": "fixed"},
                {"name": "mp", "policy": {"kind": "controller", "controller": "max_pressure"}},
                {  # long enough to still be running when the test kills its worker
                    "name": "victim",
                    "scenario": {"bundled": "grid_4x4"},
                    "config": {"duration": 1800},
                },
            ],
            "seeds": {"count": 2},
        }
    )
    eid = runner.submit(spec)
    victim = f"{eid}-victim-s1"

    def crash(event: ProgressEvent) -> None:  # the worker of the victim dies every time
        if event.kind == "started" and event.run_id == victim and event.pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(event.pid, signal.SIGTERM)

    assert runner.run(eid, on_progress=crash) is ExperimentStatus.partial
    runs = {r.id: r for r in store.list_runs(eid)}
    assert len(runs) == 6
    dead = runs.pop(victim)
    assert dead.status is RunStatus.failed and dead.error == WORKER_DIED
    assert (Path(dead.artifacts_dir) / "error.txt").is_file()
    for r in runs.values():
        assert r.status is RunStatus.completed, r
        assert (Path(r.artifacts_dir) / "summary.json").is_file()
    assert store.get_experiment(eid).counts == {"completed": 5, "failed": 1}
