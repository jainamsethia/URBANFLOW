"""The experiment runner with real worker processes: artifacts, resume, failures, cancel
(plan Q.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from urbanflow.experiments.compare import compare_experiment
from urbanflow.experiments.runner import ExperimentRunner, ProgressEvent
from urbanflow.experiments.spec import ExperimentStatus, RunStatus, parse_experiment
from urbanflow.experiments.store import ExperimentStore

pytestmark = pytest.mark.integration


def _spec(variants: list[dict[str, Any]], seeds: list[int], **extra: Any) -> Any:
    return parse_experiment(
        {
            "name": "tiny",
            "scenario": {"bundled": "single_intersection"},
            "base_config": {"duration": 60},
            "variants": variants,
            "seeds": seeds,
            **extra,
        }
    )


@pytest.fixture
def runner(tmp_path: Path) -> ExperimentRunner:
    store = ExperimentStore.open(f"sqlite:///{(tmp_path / 'uf.db').as_posix()}")
    return ExperimentRunner(tmp_path, store, max_workers=2)


FIXED = {"name": "fixed"}
PRESSURE = {"name": "mp", "policy": {"kind": "controller", "controller": "max_pressure"}}


def test_runs_complete_with_artifacts(runner: ExperimentRunner) -> None:
    events: list[ProgressEvent] = []
    eid = runner.submit(_spec([FIXED, PRESSURE], [0, 1]))
    assert runner.run(eid, on_progress=events.append) is ExperimentStatus.completed
    runs = runner.store.list_runs(eid)
    assert [r.status for r in runs] == [RunStatus.completed] * 4
    for r in runs:
        out = Path(r.artifacts_dir)
        assert {"spec.json", "scenario.json", "env.json", "result.json", "summary.json"} <= {
            p.name for p in out.iterdir()
        }
        assert any((out / "tables").glob("*.csv"))
        assert json.loads((out / "summary.json").read_text())["seed"] == r.seed
        assert r.summary is not None and r.summary["vehicles.arrived"] > 0
    assert {e.kind for e in events} >= {"started", "finished"}
    report = compare_experiment(runner.store, eid)
    travel = next(m for m in report["metrics"] if m["metric"] == "travel_time.mean")
    assert travel["vs_baseline"]["mp"]["n_pairs"] == 2 and report["missing"] == []


def test_resume_reruns_only_unfinished_runs(runner: ExperimentRunner) -> None:
    eid = runner.submit(_spec([FIXED], [0, 1]))
    runner.run(eid)
    first = runner.store.list_runs(eid)[0]
    runner.store.mark_run(first.id, RunStatus.failed, error="pretend")
    started: list[str] = []
    runner.run(eid, on_progress=lambda e: started.append(e.run_id) if e.kind == "started" else None)
    assert started == [first.id]
    assert runner.store.get_experiment(eid).status is ExperimentStatus.completed


def test_a_failing_run_does_not_stop_the_others(runner: ExperimentRunner) -> None:
    broken = {"name": "nope", "policy": {"kind": "controller", "controller": "no_such_controller"}}
    eid = runner.submit(_spec([FIXED, broken], [0]))
    assert runner.run(eid) is ExperimentStatus.partial
    ok, bad = runner.store.list_runs(eid)
    assert ok.status is RunStatus.completed and bad.status is RunStatus.failed
    assert "no_such_controller" in (bad.error or "")
    assert "Traceback" in (Path(bad.artifacts_dir) / "error.txt").read_text()


def test_cancel_stops_pending_runs(runner: ExperimentRunner) -> None:
    eid = runner.submit(_spec([FIXED], [0, 1, 2, 3, 4, 5]))
    runner.max_workers = 1

    def stop(event: ProgressEvent) -> None:
        if event.kind == "finished":
            runner.cancel()

    assert runner.run(eid, on_progress=stop) is ExperimentStatus.cancelled
    statuses = [r.status for r in runner.store.list_runs(eid)]
    assert statuses[0] is RunStatus.completed
    assert RunStatus.cancelled in statuses and RunStatus.pending not in statuses
