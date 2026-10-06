"""Run directories and environment info (plan Q.3, AC 7.2)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from urbanflow import Simulation, SimulationResult, generate
from urbanflow.core.errors import ConfigError
from urbanflow.experiments import RunArtifacts, environment_info, new_run_id
from urbanflow.experiments import artifacts as artifacts_module
from urbanflow.experiments.artifacts import summary_document


@pytest.fixture(scope="module")
def result() -> SimulationResult:
    sim = Simulation(generate("single_intersection", kind="uncontrolled", duration=60), seed=7)
    return sim.run()


def test_run_id_format() -> None:
    now = datetime(2026, 9, 23, 14, 15, 2, tzinfo=UTC)
    run_id = new_run_id("Grid 3x3!", 7, now=now)
    assert re.fullmatch(r"20260923T141502-grid-3x3-s7-[0-9a-f]{4}", run_id), run_id
    assert new_run_id("***", 0, now=now).startswith("20260923T141502-run-s0-")
    assert len(new_run_id("x" * 200, 1, now=now).split("-")[1]) == 40


def test_create_makes_a_fresh_directory(tmp_path: Path) -> None:
    a = RunArtifacts.create(tmp_path / "runs", name="demo", seed=3)
    b = RunArtifacts.create(tmp_path / "runs", name="demo", seed=3)
    assert a.directory.is_dir() and a.directory.parent == tmp_path / "runs"
    assert a.directory.name == a.run_id and a.run_id != b.run_id
    assert "-demo-s3-" in a.run_id


def test_create_in_an_explicit_directory(tmp_path: Path) -> None:
    out = RunArtifacts.create(tmp_path / "runs", name="demo", seed=0, directory=tmp_path / "out")
    assert out.directory == tmp_path / "out" and out.directory.is_dir()
    assert not (tmp_path / "runs").exists()
    (tmp_path / "out" / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="not empty"):
        RunArtifacts.create(tmp_path / "runs", name="demo", seed=0, directory=tmp_path / "out")


def test_environment_info() -> None:
    info = environment_info("numpy")
    assert {
        "python",
        "platform",
        "machine",
        "cpu_count",
        "cpu_count_physical",
        "ram_bytes",
        "numpy",
        "urbanflow",
        "accel",
    } <= set(info)
    assert info["accel"] == "numpy" and info["ram_bytes"] > 0
    json.dumps(info)


def test_files_and_summary_last(
    tmp_path: Path, result: SimulationResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = generate("single_intersection", kind="uncontrolled", duration=60)
    source = scenario.save(tmp_path / "s.json")
    loaded = type(scenario).load(source)
    run = RunArtifacts.create(tmp_path / "runs", name=loaded.name, seed=7)
    written: list[str] = []
    original = artifacts_module.write_json

    def spy(path: str | Path, obj: object, **kw: object) -> Path:
        written.append(Path(path).name)
        return original(path, obj)

    monkeypatch.setattr(artifacts_module, "write_json", spy)
    run.write_spec({"seed": 7})
    run.write_scenario(loaded)
    run.write_env()
    stamped = run.write_result(result)
    assert written == ["spec.json", "env.json", "summary.json"]  # result.json via save()
    names = sorted(p.name for p in run.directory.iterdir())
    tables = [f"{t}.csv" for t in result.tables]
    assert names == sorted(
        ["env.json", "result.json", "scenario.json", "spec.json", "summary.json", *tables]
    )
    assert run.scenario_path.read_bytes() == source.read_bytes()  # an exact copy
    spec = json.loads(run.spec_path.read_text(encoding="utf-8"))
    assert spec == {"run_id": run.run_id, "seed": 7}
    assert stamped.run_id == run.run_id
    assert SimulationResult.load(run.directory).run_id == run.run_id
    assert run.summary_path.stat().st_mtime_ns >= run.result_path.stat().st_mtime_ns


def test_scenario_without_a_file_is_saved(tmp_path: Path) -> None:
    scenario = generate("single_intersection", kind="uncontrolled")
    run = RunArtifacts.create(tmp_path, name="x", seed=0)
    path = run.write_scenario(scenario)
    assert json.loads(path.read_text(encoding="utf-8"))["meta"]["name"] == scenario.name


def test_summary_document_nests_the_dotted_keys(result: SimulationResult) -> None:
    doc = summary_document(replace(result, run_id="r"))
    assert doc["format"] == "urbanflow.metrics.summary" and doc["run_id"] == "r"
    assert doc["vehicles"]["arrived"] == result.summary["vehicles.arrived"]
    assert set(doc["travel_time"]) == {"mean", "median", "p95", "std"}
    assert doc["throughput_vph"] == result.summary["throughput_vph"]
    assert (doc["seed"], doc["dt"], doc["steps"], doc["duration_s"]) == (7, 1.0, 60, 60.0)
    empty = replace(result, summary={**result.summary, "travel_time.mean": math.nan})
    assert summary_document(empty)["travel_time"]["mean"] is None
    json.dumps(doc, allow_nan=False)
