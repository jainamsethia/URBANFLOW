"""``urbanflow experiment run/list/show/compare/export`` end to end (plan AC 7.1)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from urbanflow.cli import main as cli
from urbanflow.cli.main import app

pytestmark = pytest.mark.integration

runner = CliRunner()

SPEC = {
    "name": "cli-demo",
    "scenario": {"bundled": "single_intersection"},
    "base_config": {"duration": 60},
    "variants": [
        {"name": "fixed"},
        {"name": "mp", "policy": {"kind": "controller", "controller": "max_pressure"}},
    ],
    "seeds": [0, 1],
}


def _ok(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.stdout


def test_experiment_commands(workspace: Path) -> None:
    spec = workspace / "exp.json"
    spec.write_text(json.dumps(SPEC), encoding="utf-8")
    dry = _ok("experiment", "run", str(spec), "--dry-run")
    assert "4 run(s)" in dry and "dry-run-mp-s1" in dry
    out = _ok("experiment", "run", str(spec), "--workers", "2")
    assert "completed, 4/4 runs" in out and "mp vs fixed" in out
    listed = json.loads(_ok("experiment", "list", "--json"))
    assert [e["name"] for e in listed] == ["cli-demo"] and listed[0]["counts"] == {"completed": 4}
    eid = listed[0]["id"]
    shown = json.loads(_ok("experiment", "show", eid[:4], "--json"))
    assert len(shown["runs"]) == 4 and all(r["status"] == "completed" for r in shown["runs"])
    report = json.loads(_ok("experiment", "compare", "cli-demo", "--format", "json"))
    assert report["variants"] == ["fixed", "mp"] and report["baseline"] == "fixed"
    _ok("experiment", "export", eid, "-o", str(workspace / "export"))
    rows = list(csv.DictReader((workspace / "export" / f"cli-demo-{eid}-runs.csv").open()))
    assert len(rows) == 4 and float(rows[0]["vehicles.arrived"]) > 0
    # a second run of the same spec with --resume has nothing left to do
    again = _ok("experiment", "run", str(spec), "--resume")
    assert eid in again and "completed, 4/4 runs" in again


def test_invalid_experiment_file(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = workspace / "bad.json"
    bad.write_text(json.dumps({**SPEC, "variants": [{"name": "A B"}]}), encoding="utf-8")
    with pytest.raises(SystemExit) as info:
        cli.main(["experiment", "run", str(bad)])
    assert info.value.code == 4
    assert "variants[0].name" in capsys.readouterr().err
