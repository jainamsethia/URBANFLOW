"""The Scenario object: load/save, content hash, resolved form (plan AA 5.2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from urbanflow.core.errors import NotFoundError, ScenarioValidationError
from urbanflow.scenario import Scenario, ScenarioBuilder
from urbanflow.scenario.schema import scenario_json_schema


def test_load_and_save_round_trip(demo_path: Path, tmp_path: Path) -> None:
    scenario = Scenario.load(demo_path)
    assert scenario.path == demo_path and scenario.issues == ()
    assert scenario.name == "single-intersection-demo"
    saved = scenario.save(tmp_path / "copy.json")
    text = saved.read_text(encoding="utf-8")
    assert '"radius"' not in text  # the declared form keeps derived fields implicit
    again = Scenario.load(saved)
    assert again.spec == scenario.spec and again.content_hash == scenario.content_hash


def test_resolved_save_is_fully_explicit(demo_path: Path, tmp_path: Path) -> None:
    scenario = Scenario.load(demo_path)
    saved = scenario.save(tmp_path / "resolved.json", resolved=True)
    data = json.loads(saved.read_text(encoding="utf-8"))
    junction = data["network"]["intersections"][0]
    assert junction["radius"] == pytest.approx(8.4) and len(junction["movements"]) == 12
    assert "$schema" not in data
    again = Scenario.load(saved)
    assert again.resolved == scenario.resolved
    assert again.content_hash == scenario.content_hash


def test_content_hash_excludes_meta(demo_data: dict[str, Any]) -> None:
    base = Scenario.from_dict(demo_data)
    assert len(base.content_hash) == 64 and base.short_hash == base.content_hash[:12]
    demo_data["meta"] = {"name": "renamed", "description": "other", "tags": []}
    assert Scenario.from_dict(demo_data).content_hash == base.content_hash
    demo_data["network"]["roads"][0]["lanes"][0]["width"] = 3.5
    assert Scenario.from_dict(demo_data).content_hash != base.content_hash


def test_content_hash_ignores_declared_vs_resolved(demo_path: Path, tmp_path: Path) -> None:
    scenario = Scenario.load(demo_path)
    resolved_file = scenario.save(tmp_path / "r.json", resolved=True)
    assert Scenario.load(resolved_file).short_hash == scenario.short_hash
    assert scenario.short_hash == "b758edc3a2b6"  # golden: derivation changes must be deliberate


def test_constructors(demo_data: dict[str, Any], demo_path: Path) -> None:
    text = demo_path.read_text(encoding="utf-8")
    by_text = Scenario.from_json(text)
    by_bytes = Scenario.from_json(text.encode("utf-8"))
    by_spec = Scenario.from_spec(by_text.spec)
    assert by_text.content_hash == by_bytes.content_hash == by_spec.content_hash
    assert Scenario.from_dict(demo_data).content_hash == by_text.content_hash
    assert Scenario.json_schema() == scenario_json_schema()
    assert repr(by_text) == f"Scenario(name='single-intersection-demo', hash={by_text.short_hash})"


def test_errors_name_the_source(demo_data: dict[str, Any], tmp_path: Path) -> None:
    demo_data["demand"]["trips"][0]["vehicle_type"] = "cr"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(demo_data), encoding="utf-8")
    with pytest.raises(ScenarioValidationError) as info:
        Scenario.load(path)
    assert info.value.source == str(path)
    assert info.value.issues[0].path == "demand.trips[0].vehicle_type"
    with pytest.raises(NotFoundError, match="did you mean"):
        Scenario.load(tmp_path / "brokn.json")
    with pytest.raises(ScenarioValidationError) as info:
        Scenario.from_json("{nope")
    assert info.value.issues[0].code == "E000"


def test_warnings_are_kept_and_logged(
    demo_data: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    demo_data["simulation"]["dt"] = 0.1
    with caplog.at_level(logging.WARNING, logger="urbanflow.scenario.validate"):
        scenario = Scenario.from_dict(demo_data, source="mine")
    assert [w.code for w in scenario.issues] == ["W701"]
    (record,) = [r for r in caplog.records if r.name == "urbanflow.scenario.validate"]
    assert record.levelno == logging.WARNING and "mine: simulation.dt: dt=0.1 s" in record.message
    assert (record.__dict__["code"], record.__dict__["path"]) == ("W701", "simulation.dt")


def test_load_logs_the_scenario(demo_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """AB 6.4: "scenario loaded" (INFO) with the name (``scenario``), hash, path, counts and
    duration_ms."""
    with caplog.at_level(logging.INFO, logger="urbanflow.scenario"):
        scenario = Scenario.load(demo_path)
    (record,) = [r for r in caplog.records if r.message == "scenario loaded"]
    assert record.name == "urbanflow.scenario" and record.levelno == logging.INFO
    fields = record.__dict__
    assert (fields["scenario"], fields["hash"], fields["path"]) == (
        scenario.name,
        scenario.short_hash,
        str(demo_path),
    )
    assert fields["counts"] == scenario.summary() and fields["duration_ms"] >= 0


def test_failed_load_is_not_logged_as_loaded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{oops", encoding="utf-8")
    with (
        caplog.at_level(logging.INFO, logger="urbanflow.scenario"),
        pytest.raises(ScenarioValidationError),
    ):
        Scenario.load(path)
    assert "scenario loaded" not in caplog.text


def test_save_always_writes_the_current_version(demo_data: dict[str, Any], tmp_path: Path) -> None:
    """E.7 §1.7: a spec carrying another version still saves as the current one."""
    spec = Scenario.from_dict(demo_data).spec.model_copy(update={"version": "1.7"})
    raw = Scenario(spec)  # unvalidated: validation would report E011
    for resolved in (False, True):
        path = raw.save(tmp_path / f"s{resolved}.json", resolved=resolved)
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == "1.0"
        assert Scenario.load(path).content_hash == Scenario.from_dict(demo_data).content_hash


def test_to_dict_forms(demo_data: dict[str, Any]) -> None:
    scenario = Scenario.from_dict(demo_data)
    assert scenario.to_dict() == demo_data
    resolved = scenario.to_dict(resolved=True)
    assert resolved["network"]["roads"][0]["points"] == [[0.0, 200.0], [0.0, 0.0]]
    assert json.loads(scenario.to_json(resolved=True)) == resolved
    assert "\n" not in scenario.to_json(indent=None)


def test_summary(demo_data: dict[str, Any]) -> None:
    assert Scenario.from_dict(demo_data).summary() == {
        "intersections": 5,
        "roads": 8,
        "lanes": 16,
        "movements": 12,
        "connections": 16,
        "signals": 1,
        "phases": 2,
        "vehicle_types": 5,
        "flows": 4,
        "trips": 1,
        "transit_lines": 1,
    }


def test_edit_and_generate(demo_data: dict[str, Any]) -> None:
    scenario = Scenario.from_dict(demo_data)
    builder = scenario.edit()
    assert isinstance(builder, ScenarioBuilder)
    assert builder.build().content_hash == scenario.content_hash
    generated = Scenario.generate("single_intersection", arms=3)
    assert generated.summary()["intersections"] == 4


def test_unvalidated_construction_resolves(demo_data: dict[str, Any]) -> None:
    spec = Scenario.from_dict(demo_data).spec
    raw = Scenario(spec)
    assert raw.issues == () and raw.resolved.network.intersections[0].radius == pytest.approx(8.4)
