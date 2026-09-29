"""SimulationResult (plan AA 5.8): serialisation, lossless save/load, table rendering."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import replace
from pathlib import Path

import pytest

from urbanflow import SimulationResult
from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import ConfigError, NotFoundError
from urbanflow.results import RESULT_FILE, provenance

SUMMARY = {
    "vehicles.generated": 2412,
    "vehicles.inserted": 2400,
    "vehicles.arrived": 2180,
    "vehicles.en_route": 220,
    "vehicles.backlog": 12,
    "vehicles.removed": 0,
    "vehicles.teleported": 0,
    "travel_time.mean": 161.4,
    "travel_time.median": 150.25,
    "travel_time.p95": 300.0,
    "travel_time.std": 40.125,
    "throughput_vph": 4360.0,
}


@pytest.fixture
def result() -> SimulationResult:
    return SimulationResult(
        scenario_name="grid-3x3",
        scenario_hash="ab" * 32,
        config=SimulationConfig(dt=0.5, duration=1800, seed=7, metrics={"warmup": 60}),
        seed=7,
        sim_time=1800.0,
        steps=3600,
        wall_time=2.4125,
        interrupted=False,
        summary=SUMMARY,
        event_counts={"vehicle_arrived": 2180, "vehicle_inserted": 2400},
        provenance=provenance(),
    )


def _same(a: SimulationResult, b: SimulationResult) -> bool:
    sa, sb = dict(a.summary), dict(b.summary)
    nan_a = {k for k, v in sa.items() if isinstance(v, float) and math.isnan(v)}
    nan_b = {k for k, v in sb.items() if isinstance(v, float) and math.isnan(v)}
    rest_a = {k: v for k, v in sa.items() if k not in nan_a}
    rest_b = {k: v for k, v in sb.items() if k not in nan_b}
    return nan_a == nan_b and rest_a == rest_b and replace(a, summary={}) == replace(b, summary={})


def test_provenance_keys() -> None:
    info = provenance()
    assert set(info) == {"urbanflow", "numpy", "python", "platform", "machine"}
    assert info["machine"] == platform.machine()
    assert platform.python_version() in info["python"]


def test_to_dict_is_strict_json(result: SimulationResult) -> None:
    partial = replace(result, summary={**SUMMARY, "travel_time.mean": math.nan})
    data = partial.to_dict()
    text = json.dumps(data, allow_nan=False)  # NaN became null
    assert json.loads(text)["summary"]["travel_time.mean"] is None
    assert data["format"] == "urbanflow.result"
    assert data["config"]["dt"] == 0.5 and data["config"]["metrics"]["warmup"] == 60


@pytest.mark.parametrize("mean", [161.4, math.nan, 0.1 + 0.2])
def test_save_load_round_trip(result: SimulationResult, tmp_path: Path, mean: float) -> None:
    original = replace(result, summary={**SUMMARY, "travel_time.mean": mean}, run_id="r1")
    assert original.save(tmp_path / "run") == tmp_path / "run"
    assert (tmp_path / "run" / RESULT_FILE).is_file()
    loaded = SimulationResult.load(tmp_path / "run")
    assert _same(loaded, original)
    assert loaded.config == original.config and loaded.config.metrics.warmup == 60
    assert loaded.run_id == "r1"


def test_load_errors(result: SimulationResult, tmp_path: Path) -> None:
    with pytest.raises(NotFoundError, match=r"result.json missing"):
        SimulationResult.load(tmp_path)
    (tmp_path / RESULT_FILE).write_text('{"format": "other"}', encoding="utf-8")
    with pytest.raises(ConfigError, match="not an UrbanFlow result"):
        SimulationResult.load(tmp_path)
    data = result.to_dict()
    del data["seed"]
    (tmp_path / RESULT_FILE).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="malformed"):
        SimulationResult.load(tmp_path)
    (tmp_path / RESULT_FILE).write_text("[1]", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        SimulationResult.load(tmp_path)


def test_str_is_an_aligned_table(result: SimulationResult) -> None:
    assert str(result).splitlines() == [
        "Results: grid-3x3 (seed 7, 1800 s)",
        "Metric                               Value",
        "Vehicles departed / arrived  2,412 / 2,180",
        "Vehicles en route / waiting       220 / 12",
        "Throughput                     4,360 veh/h",
        "Mean travel time                   161.4 s",
        "Teleports                                0",
        "Wall time                           2.41 s",
    ]


def test_undefined_values_and_interruption(result: SimulationResult) -> None:
    empty = {**SUMMARY, "travel_time.mean": math.nan, "throughput_vph": math.nan}
    partial = replace(result, summary=empty, interrupted=True, sim_time=12.5)
    assert partial.title == "Results: grid-3x3 (seed 7, 12.5 s, interrupted)"
    rows = dict(partial.rows())
    assert rows["Mean travel time"] == "n/a" and rows["Throughput"] == "n/a"
