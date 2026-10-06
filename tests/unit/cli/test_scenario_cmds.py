"""Scenario commands: init, validate, generate, schema (plan AC 7.1)."""

from __future__ import annotations

import json
import shutil
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from urbanflow.cli import main as cli
from urbanflow.core.settings import load_settings
from urbanflow.scenario import Scenario, ScenarioBuilder
from urbanflow.scenario.generators import GENERATORS, register_generator
from urbanflow.scenario.io import bundled
from urbanflow.scenario.schema import scenario_json_schema

DEMO = Path(__file__).resolve().parents[2] / "fixtures" / "scenarios" / "demo.json"


def run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    return int(info.value.code or 0)


@pytest.mark.parametrize("command", ["init", "validate", "generate", "schema"])
def test_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert run([command, "--help"]) == 0
    assert "Usage" in capsys.readouterr().out


def test_commands_are_in_the_scenarios_panel(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--help"]) == 0
    out = capsys.readouterr().out
    assert "Scenarios" in out and out.index("init") < out.index("doctor")


# ---------------------------------------------------------------------------- init
def test_init_scaffolds_a_workspace(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["init", "proj"]) == 0
    root = workspace / "proj"
    scenario = root / "scenarios" / "single_intersection.json"
    assert scenario.read_bytes() == bundled("single_intersection").read_bytes()
    assert (root / "runs").is_dir()
    assert (root / ".gitignore").read_text(encoding="utf-8") == "runs/\n.urbanflow/\n"
    assert "urbanflow validate scenarios/single_intersection.json" in (
        root / "README.md"
    ).read_text(encoding="utf-8")
    config = tomllib.loads((root / "urbanflow.toml").read_text(encoding="utf-8"))
    assert config["server"]["port"] == 8000
    assert load_settings(workspace=root).port == 8000
    experiment = json.loads((root / "experiments" / "example.json").read_text(encoding="utf-8"))
    assert experiment["format"] == "urbanflow.experiment"
    assert experiment["scenario"] == {"path": "scenarios/single_intersection.json"}
    out = capsys.readouterr().out
    assert "scenarios/single_intersection.json" in out and "Next steps:" in out
    assert Scenario.load(scenario).issues == ()


def test_init_refuses_a_non_empty_directory(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "notes.txt").write_text("mine", encoding="utf-8")
    assert run(["init"]) == 4
    assert "not empty" in capsys.readouterr().err
    (workspace / "README.md").write_text("keep me", encoding="utf-8")
    assert run(["init", "--force", "--name", "demo"]) == 0
    assert (workspace / "README.md").read_text(encoding="utf-8") == "keep me"
    assert (workspace / "scenarios" / "demo.json").is_file()
    assert "README.md  (exists, kept)" in capsys.readouterr().out


def test_init_uses_the_global_workspace(tmp_path: Path, workspace: Path) -> None:
    target = tmp_path / "elsewhere"
    assert run(["-w", str(target), "init"]) == 0
    assert (target / "urbanflow.toml").is_file()


@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        (["init", "--template", "gird"], 5, "available: grid_3x3, grid_4x4, single_intersection"),
        (["init", "--name", "../evil"], 4, "--name must be a plain file name"),
    ],
)
def test_init_errors(
    argv: list[str], code: int, message: str, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(argv) == code
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------- validate
@pytest.fixture
def broken(workspace: Path) -> Path:
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    green = data["network"]["intersections"][0]["signal"]["phases"][1]["green"]
    green["E_in->W_ot"] = green.pop("E_in->W_out")
    data["demand"]["trips"][0]["vehicle_type"] = "cr"
    data["simulation"]["dt"] = 0.1
    path = workspace / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_validate_ok(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["validate", str(DEMO)]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("OK: single-intersection-demo (hash b758edc3a2b6; 5 intersections,")
    assert out.endswith("16 lanes, 12 movements, 4 flows)")


def test_validate_reports_errors_then_warnings(
    broken: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(["validate", str(broken)]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "Scenario validation failed:",
        "  - network.intersections[0].signal.phases[1]: phase contains unknown movement "
        '"E_in->W_ot" (did you mean "E_in->W_out"?)',
        '  - demand.trips[0].vehicle_type: unknown vehicle type "cr" (did you mean "car"?) '
        "(available: bus, car, city_bus, emergency, truck)",
        "Warnings:",
        '  - network.intersections[0].signal: movement "E_in->W_out" is never green in any phase; '
        "its traffic can never enter",
        "  - simulation.dt: dt=0.1 s is outside the recommended 0.2-1.0 s",
    ]
    assert "Traceback" not in captured.err


def test_validate_strict_and_warnings(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    data["simulation"]["dt"] = 0.1
    path = workspace / "warn.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert run(["validate", str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.err.startswith("Warnings:\n  - simulation.dt:") and captured.out.startswith(
        "OK:"
    )
    assert run(["validate", "--strict", str(path)]) == 3
    assert capsys.readouterr().err.startswith("Scenario validation failed:\n  - simulation.dt:")


def test_validate_json_format(broken: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["validate", "--format", "json", str(DEMO), str(broken)]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    good, bad = report["files"]
    assert good["ok"] and good["name"] == "single-intersection-demo" and len(good["hash"]) == 64
    assert good["counts"]["roads"] == 8
    assert [e["code"] for e in bad["errors"]] == ["E302", "E402"]
    assert {w["code"] for w in bad["warnings"]} == {"W301", "W701"}
    assert set(bad["errors"][0]) == {"code", "path", "message", "severity"}


def test_validate_several_files(broken: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["validate", "--deep", str(DEMO), str(broken)]) == 3
    captured = capsys.readouterr()
    assert f"{DEMO}:" in captured.err and f"{broken}:" in captured.err
    assert captured.out.startswith("OK: single-intersection-demo")


def _protected_left(workspace: Path) -> Path:
    """The demo with N_in->E_out protected (G) while it crosses the protected S_in->N_out."""
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    data["network"]["intersections"][0]["signal"]["phases"][0]["green"]["N_in->E_out"] = "G"
    path = workspace / "w302.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_validate_deep_reports_compile_warnings(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _protected_left(workspace)
    assert run(["validate", str(path)]) == 0  # deep is the default
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        "Warnings:",
        '  - network.intersections[0].signal.phases[0]: movements "N_in->E_out" and '
        '"S_in->N_out" cross but are both protected (G); make one permissive (g)',
    ]
    assert captured.out.startswith("OK:")
    assert run(["validate", "--strict", str(path)]) == 3
    assert run(["validate", "--no-deep", "--strict", str(path)]) == 0
    capsys.readouterr()
    assert run(["validate", "--format", "json", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [w["code"] for w in report["files"][0]["warnings"]] == ["W302"]


def test_validate_deep_reports_compile_errors(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    data["network"]["intersections"][0]["radius"] = 0  # connectors would have zero length
    data["network"]["intersections"][0]["movements"] = [
        {"from_road": "N_in", "to_road": "S_out", "connections": [{"from_lane": 0, "to_lane": 0}]}
    ]
    data["network"]["intersections"][0]["signal"]["phases"] = [{"green": {"N_in->S_out": "G"}}]
    data["demand"] = {}
    path = workspace / "e905.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert run(["validate", "--no-deep", str(path)]) == 0
    capsys.readouterr()
    assert run(["validate", str(path)]) == 3
    err = capsys.readouterr().err
    assert "  - network.intersections[0].movements[0].connections[0]: lane geometry is " in err


def test_validate_debug_adds_a_traceback(broken: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--debug", "validate", str(broken)]) == 3
    assert "Traceback" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "code", "fragment"),
    [
        (None, 5, "scenario file not found"),
        ("{oops", 3, "  - $: invalid JSON:"),
        ('{"format": "cityflow"}', 3, "  - format: not an UrbanFlow scenario"),
    ],
)
def test_validate_load_errors(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    content: str | None,
    code: int,
    fragment: str,
) -> None:
    path = workspace / "s.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert run(["validate", str(path)]) == code
    assert fragment in capsys.readouterr().err


def test_validate_keeps_going_after_a_bad_or_missing_file(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A load error or a missing file is one file's result; the others are still checked."""
    bad = workspace / "bad.json"
    bad.write_bytes(b'{"a": "\xff"}')  # invalid UTF-8: a load-stage (E000) error
    assert run(["validate", str(bad), str(DEMO)]) == 3
    captured = capsys.readouterr()
    assert "  - $: file is not valid UTF-8:" in captured.err
    assert captured.out.startswith("OK: single-intersection-demo")

    missing = workspace / "missing.json"
    assert run(["validate", str(missing), str(bad), str(DEMO)]) == 5
    captured = capsys.readouterr()
    assert f"Error: scenario file not found: {missing}" in captured.err
    assert "not valid UTF-8" in captured.err and captured.out.startswith("OK:")

    assert run(["validate", "--format", "json", str(missing), str(bad), str(DEMO)]) == 5
    captured = capsys.readouterr()
    report = json.loads(captured.out)  # stdout is only the JSON document
    gone, broken, good = report["files"]
    assert report["ok"] is False and good["ok"] and not broken["ok"]
    assert gone["ok"] is False and "scenario file not found" in gone["error"]
    assert [e["code"] for e in broken["errors"]] == ["E000"]


def test_validate_reports_invalid_built_in_overrides(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    data["vehicle_types"].append({"id": "emergency", "speed_factor": {"max": 1.25}})
    path = workspace / "override.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert run(["validate", str(path)]) == 3  # not an internal error (exit 1)
    err = capsys.readouterr().err
    assert "  - vehicle_types[2].speed_factor: speed_factor requires min <= mean <= max" in err
    assert "Internal error" not in err


def test_validate_rejects_unknown_format(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["validate", "--format", "yaml", str(DEMO)]) == 4


# ---------------------------------------------------------------------------- generate
def test_generate_list_and_describe(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["generate", "--list"]) == 0
    out = capsys.readouterr().out
    assert "single_intersection" in out and "builtin" in out and "Source" in out
    assert run(["generate", "single_intersection", "--describe"]) == 0
    out = capsys.readouterr().out
    for name in ("arms", "turn_ratios", "demand_rate", "Literal[3, 4]"):
        assert name in out


def test_generate_writes_into_the_workspace(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(["generate", "single_intersection", "-p", "kind=priority", "-p", "arms=3"]) == 0
    target = workspace / "scenarios" / "single_intersection.json"
    scenario = Scenario.load(target)
    assert scenario.resolved.network.intersections[0].kind == "priority"
    out = capsys.readouterr().out.strip()
    assert out == (
        f"Wrote {Path('scenarios') / 'single_intersection.json'} (4 intersections, 6 roads, "
        f"6 flows; hash {scenario.short_hash})"
    )
    assert run(["generate", "single_intersection"]) == 4
    assert "already exists (use --force to overwrite)" in capsys.readouterr().err
    assert run(["generate", "single_intersection", "--force"]) == 0


def test_generate_params_file_and_dotted_keys(workspace: Path) -> None:
    params = workspace / "p.json"
    params.write_text(json.dumps({"lanes": 3, "turn_ratios": {"far": 0.2}}), encoding="utf-8")
    argv = ["generate", "single_intersection", "--params", str(params), "-o", "x.json"]
    argv += ["-p", "turn_ratios.straight=0.6", "-p", "turn_ratios.near=0.2"]
    assert run(argv) == 0
    info = Scenario.load(workspace / "x.json").spec.meta.generator
    assert info is not None and info.params["lanes"] == 3
    assert info.params["turn_ratios"] == {"far": 0.2, "straight": 0.6, "near": 0.2}


@pytest.mark.parametrize(
    ("argv", "code", "fragment"),
    [
        (["generate", "single_intersection", "-p", "lanes=0", "-o", "x.json"], 3, "params.lanes"),
        (["generate", "single_intersection", "-p", "lanes"], 4, "expected KEY=VALUE"),
        (["generate", "grdi"], 5, 'unknown generator "grdi"'),
        (["generate"], 2, "missing generator NAME"),
        (["generate", "single_intersection", "--params", "nope.json"], 5, "params file not found"),
    ],
)
def test_generate_errors(
    argv: list[str],
    code: int,
    fragment: str,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(argv) == code
    captured = capsys.readouterr()
    assert fragment in captured.err + captured.out
    assert not (workspace / "x.json").exists()


class _Empty(BaseModel):
    pass


def _broken_generator(_p: _Empty) -> Scenario:
    """Returns an invalid scenario (plugin that skips validation)."""
    b = ScenarioBuilder("broken")
    b.boundary("A", (0, 0)).boundary("B", (100, 0)).road("A_B", "A", "B")
    b.trip("t", 0, route=["A_B", "nowhere"])
    return b.build(validate=False)


@pytest.fixture
def broken_generator() -> Iterator[None]:
    register_generator("broken_gen", params=_Empty)(_broken_generator)
    yield
    GENERATORS._items.pop("broken_gen", None)


def test_generate_validates_plugin_output(
    broken_generator: None, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(["generate", "broken_gen", "-o", "b.json"]) == 3
    assert 'demand.trips[0].route[1]: unknown road "nowhere"' in capsys.readouterr().err
    assert not (workspace / "b.json").exists()
    assert run(["generate", "broken_gen", "-o", "b.json", "--no-validate"]) == 0
    assert (workspace / "b.json").is_file()


# ---------------------------------------------------------------------------- schema
def test_schema_to_stdout_and_file(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["schema"]) == 0
    assert json.loads(capsys.readouterr().out) == scenario_json_schema()
    assert run(["schema", "config"]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["$id"] == "urn:urbanflow:config:1.0" and "dt" in config["properties"]
    assert run(["schema", "scenario", "-o", "out/s.json"]) == 0
    assert json.loads((workspace / "out" / "s.json").read_text(encoding="utf-8")) == (
        scenario_json_schema()
    )
    assert run(["schema", "yaml"]) == 2


def test_schema_file_can_be_copied_next_to_a_scenario(workspace: Path) -> None:
    """The exported schema is plain JSON Schema: editors reference it via "$schema"."""
    assert run(["schema", "-o", "scenario.schema.json"]) == 0
    target = workspace / "s.json"
    shutil.copy(DEMO, target)
    data = json.loads(target.read_text(encoding="utf-8"))
    data["$schema"] = "./scenario.schema.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    assert run(["validate", str(target)]) == 0
