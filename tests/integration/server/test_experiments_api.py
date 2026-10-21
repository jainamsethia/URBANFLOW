"""Experiments over REST: create, queue, run, cancel, compare, export, artifacts (plan M.1)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urbanflow.core.settings import load_settings
from urbanflow.experiments.spec import ExperimentStatus, parse_experiment
from urbanflow.experiments.store import ExperimentStore
from urbanflow.server.app import create_app

pytestmark = pytest.mark.integration

API = "/api/v1"


def _spec(**extra: Any) -> dict[str, Any]:
    return {
        "name": "api-demo",
        "scenario": {"bundled": "single_intersection"},
        "base_config": {"duration": 60},
        "variants": [
            {"name": "fixed"},
            {"name": "mp", "policy": {"kind": "controller", "controller": "max_pressure"}},
        ],
        "seeds": [0, 1],
        **extra,
    }


@pytest.fixture
def client(workspace: Path) -> Iterator[TestClient]:
    with TestClient(create_app(load_settings(max_workers=2))) as c:
        yield c


def _wait(client: TestClient, eid: str, done: set[str], timeout: float = 180) -> dict[str, Any]:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        doc = client.get(f"{API}/experiments/{eid}").json()
        if doc["status"] in done:
            return doc
        time.sleep(0.3)
    raise AssertionError(f"experiment {eid} still {doc['status']}")


def test_experiment_lifecycle(client: TestClient) -> None:
    r = client.post(f"{API}/experiments", json={"spec": _spec()})
    assert r.status_code == 201, r.text
    exp = r.json()
    eid = exp["id"]
    assert exp["status"] == "draft" and exp["progress"]["total"] == 4
    assert [run["status"] for run in exp["runs"]] == ["pending"] * 4
    listed = client.get(f"{API}/experiments").json()
    assert listed["total"] == 1 and listed["items"][0]["variants"] == ["fixed", "mp"]

    assert client.post(f"{API}/experiments/{eid}/run").status_code == 202
    doc = _wait(client, eid, {"completed", "failed", "partial"})
    assert doc["status"] == "completed" and doc["progress"]["completed"] == 4
    assert client.post(f"{API}/experiments/{eid}/cancel").status_code == 409  # not active

    report = client.get(f"{API}/experiments/{eid}/compare", params={"metric": "delay.mean"}).json()
    assert [m["metric"] for m in report["metrics"]] == ["delay.mean"]
    assert report["metrics"][0]["vs_baseline"]["mp"]["n_pairs"] == 2
    ts = client.get(f"{API}/experiments/{eid}/timeseries", params={"column": "active"}).json()
    assert [s["variant"] for s in ts["series"]] == ["fixed", "mp"]
    fixed = ts["series"][0]
    assert fixed["runs"] == 2 and len(fixed["time"]) == len(fixed["mean"]) > 1
    assert all(
        lo is None or lo <= m <= hi
        for lo, m, hi in zip(fixed["lo"], fixed["mean"], fixed["hi"], strict=True)
    )
    bad = client.get(f"{API}/experiments/{eid}/timeseries", params={"column": "time"})
    assert bad.status_code == 422
    csv_text = client.get(f"{API}/experiments/{eid}/export").text
    assert csv_text.startswith("run_id,variant,seed,status") and len(csv_text.splitlines()) == 5
    cmp = client.get(f"{API}/experiments/{eid}/export", params={"what": "comparison"})
    assert "attachment" in cmp.headers["content-disposition"]

    runs = client.get(f"{API}/runs", params={"experiment_id": eid}).json()
    assert runs["total"] == 4
    run_id = runs["items"][0]["id"]
    run = client.get(f"{API}/runs/{run_id}").json()
    assert run["summary"]["vehicles.arrived"] > 0 and run["config"]["seed"] == 0
    names = {a["name"] for a in client.get(f"{API}/runs/{run_id}/artifacts").json()}
    assert {"summary.json", "result.json", "tables/trips.csv"} <= names
    summary = client.get(f"{API}/runs/{run_id}/artifacts/summary.json").json()
    assert summary["seed"] == 0
    for bad in ("../uf.db", "..%2Fsummary.json", "nope.json"):
        assert client.get(f"{API}/runs/{run_id}/artifacts/{bad}").status_code == 404

    run_dir = Path("runs") / run_id  # the workspace is the current directory
    assert (run_dir / "summary.json").is_file()
    assert client.delete(f"{API}/experiments/{eid}", params={"purge": True}).status_code == 204
    assert client.get(f"{API}/experiments/{eid}").status_code == 404
    assert not run_dir.exists()


def test_invalid_specs_are_rejected_with_paths(client: TestClient) -> None:
    r = client.post(f"{API}/experiments", json={"spec": _spec(variants=[{"name": "A B"}])})
    assert r.status_code == 422
    assert r.json()["issues"][0]["path"] == "variants[0].name"
    leaving = _spec(scenario={"path": "../elsewhere.json"})
    r = client.post(f"{API}/experiments", json={"spec": leaving})
    assert r.status_code == 422 and "leaves the workspace" in r.json()["detail"]
    huge = _spec(seeds={"count": 1000}, variants=[{"name": f"v{i}"} for i in range(3)])
    assert client.post(f"{API}/experiments", json={"spec": huge}).status_code == 422


def test_queued_experiments_run_one_at_a_time_and_can_be_cancelled(client: TestClient) -> None:
    long = _spec(name="first", base_config={"duration": 600}, seeds=[0, 1, 2])
    first = client.post(f"{API}/experiments", json={"spec": long, "run": True}).json()["id"]
    second = client.post(f"{API}/experiments", json={"spec": _spec(), "run": True}).json()["id"]
    assert client.get(f"{API}/experiments/{second}").json()["status"] == "queued"
    assert client.delete(f"{API}/experiments/{second}").status_code == 409  # still queued
    assert client.post(f"{API}/experiments/{second}/run").status_code == 409
    r = client.post(f"{API}/experiments/{second}/cancel")
    assert r.status_code == 202 and r.json()["status"] == "cancelled"
    assert {run["status"] for run in r.json()["runs"]} == {"cancelled"}
    assert _wait(client, first, {"completed"})["progress"]["completed"] == 6


def test_a_restart_marks_running_experiments_interrupted(workspace: Path) -> None:
    settings = load_settings(max_workers=1)
    store = ExperimentStore.open(settings.resolved_database_url)
    spec = parse_experiment(_spec())
    store.create_experiment("deadbeef0000", spec, [], status=ExperimentStatus.running)
    store.close()
    with TestClient(create_app(settings)) as c:
        assert c.get(f"{API}/experiments/deadbeef0000").json()["status"] == "interrupted"
