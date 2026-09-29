"""``urbanflow run`` end to end (plan AC 7.1, 7.2): output, artifacts, JSON, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urbanflow import SimulationResult, generate
from urbanflow.cli import main as cli
from urbanflow.engine import Engine

pytestmark = pytest.mark.integration


def run(*argv: str) -> int:
    with pytest.raises(SystemExit) as info:
        cli.main(list(argv))
    return int(info.value.code or 0)


@pytest.fixture
def scenario(workspace: Path) -> Path:
    return generate("single_intersection", kind="uncontrolled").save(workspace / "s.json")


def _runs(workspace: Path) -> list[Path]:
    runs = workspace / "runs"
    return sorted(runs.iterdir()) if runs.exists() else []


def test_help_lists_run_in_the_simulation_panel(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--help") == 0
    out = capsys.readouterr().out
    assert "Simulation" in out and "run" in out
    assert run("run", "--help") == 0
    assert "--until" in capsys.readouterr().out


def test_a_60_s_run(workspace: Path, scenario: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("run", "s.json", "--duration", "60", "--seed", "7", "--no-progress") == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith(
        "Loaded single_intersection: 1 intersection, 8 roads, 16 lanes, 16 connectors  (hash "
    )
    assert "Results: single_intersection (seed 7, 60 s)" in out
    assert "Vehicles departed / arrived" in out and "Wall time" in out
    (run_dir,) = _runs(workspace)
    assert f"Run saved: {Path('runs') / run_dir.name}" in out
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "env.json",
        "result.json",
        "scenario.json",
        "spec.json",
        "summary.json",
    ]
    assert (run_dir / "scenario.json").read_bytes() == scenario.read_bytes()
    result = SimulationResult.load(run_dir)
    assert result.run_id == run_dir.name and result.seed == 7 and result.sim_time == 60.0
    assert result.config.duration == 60 and not result.interrupted
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["vehicles"]["generated"] == result.summary["vehicles.generated"] > 0
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    assert spec["run_id"] == run_dir.name and spec["config"]["seed"] == 7
    assert spec["scenario"]["hash"] == result.scenario_hash


def test_json_output(workspace: Path, scenario: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("run", "s.json", "--until", "30", "--json") == 0
    data = json.loads(capsys.readouterr().out)  # nothing but JSON on stdout
    assert data["sim_time"] == 30.0 and data["steps"] == 30
    assert data["summary"]["vehicles.generated"] > 0
    (run_dir,) = _runs(workspace)
    assert data["run_dir"] == str(run_dir) and data["run_id"] == run_dir.name


def test_no_save_and_out(
    workspace: Path, scenario: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("run", "s.json", "--duration", "10", "--no-save") == 0
    assert "Run saved" not in capsys.readouterr().out and _runs(workspace) == []
    assert run("run", "s.json", "--duration", "10", "--out", "results/base") == 0
    assert (workspace / "results" / "base" / "summary.json").is_file()
    assert run("run", "s.json", "--duration", "10", "--out", "results/base") == 4  # not empty


def test_config_layers(workspace: Path, scenario: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (workspace / "urbanflow.toml").write_text(
        "[simulation]\ndt = 0.5\nseed = 3\nduration = 20\n", encoding="utf-8"
    )
    assert run("run", "s.json", "--json", "--no-save") == 0
    config = json.loads(capsys.readouterr().out)["config"]
    assert (config["dt"], config["seed"], config["duration"]) == (0.5, 3, 20)
    other = workspace / "other.toml"
    other.write_text("[simulation]\nseed = 4\nduration = 10\n", encoding="utf-8")
    argv = ["run", "s.json", "--json", "--no-save", "--config", str(other)]
    assert run(*argv, "--set", "seed=5", "--set", "halting_speed=0.2") == 0
    config = json.loads(capsys.readouterr().out)["config"]
    assert (config["dt"], config["seed"], config["halting_speed"]) == (1.0, 5, 0.2)
    assert run(*argv, "--set", "seed=5", "--seed", "6", "--debug-checks") == 0
    config = json.loads(capsys.readouterr().out)["config"]
    assert config["seed"] == 6 and config["debug_checks"] is True


@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        (["missing.json"], 5, "scenario file not found"),
        (["s.json", "--set", "dt=5"], 4, "--set: simulation.dt: must be <= 2 (got 5)"),
        (["s.json", "--set", "oops"], 4, 'invalid assignment "oops"'),
        (["s.json", "--router", "fastest"], 3, 'unknown router "fastest"'),
        (["s.json", "--config", "none.toml"], 5, "config file not found"),
        (["signalized.json"], 6, "signalized intersections"),
    ],
)
def test_exit_codes(
    workspace: Path,
    scenario: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    code: int,
    message: str,
) -> None:
    generate("single_intersection").save(workspace / "signalized.json")
    assert run("run", *argv, "--duration", "5", "--no-save") == code
    assert message in capsys.readouterr().err


def test_invalid_scenario_exits_3(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (workspace / "bad.json").write_text('{"format": "urbanflow.scenario"}', encoding="utf-8")
    assert run("run", "bad.json") == 3
    assert "Scenario validation failed" in capsys.readouterr().err


def test_ctrl_c_saves_partial_results(
    workspace: Path,
    scenario: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_step = Engine.step

    def step(self: Engine) -> None:
        if self.step_count == 25:
            raise KeyboardInterrupt
        real_step(self)

    monkeypatch.setattr(Engine, "step", step)
    assert run("run", "s.json", "--duration", "60") == 130
    captured = capsys.readouterr()
    assert "Interrupted at t=25 s; partial results in" in captured.err
    assert "Results: single_intersection (seed 0, 25 s, interrupted)" in captured.out
    (run_dir,) = _runs(workspace)
    result = SimulationResult.load(run_dir)
    assert result.interrupted and result.steps == 25
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["interrupted"] is True
