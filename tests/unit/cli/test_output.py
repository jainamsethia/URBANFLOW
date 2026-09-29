"""Console helpers: results on stdout, issues on stderr, JSON mode (plan AC, U.3)."""

from __future__ import annotations

import json

import pytest

from urbanflow.cli.output import emit_json, print_issues, print_summary, print_table
from urbanflow.core.errors import Severity, ValidationIssue

LONG = "x" * 150  # wider than the captured console: must not be wrapped or cut


def test_print_issues_golden_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    issues = [
        ValidationIssue("demand.flows[0].rate", "must be > 0 (got -5.0)", "E004"),
        ValidationIssue("$", f"invalid JSON: {LONG}", "E000"),
    ]
    print_issues(issues)
    warning = ValidationIssue("simulation.dt", "[bold]dt=0.1 s[/]", "W701", Severity.warning)
    print_issues([warning], header="Warnings:")  # markup is printed literally
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Scenario validation failed:\n"
        "  - demand.flows[0].rate: must be > 0 (got -5.0)\n"
        f"  - $: invalid JSON: {LONG}\n"
        "Warnings:\n"
        "  - simulation.dt: [bold]dt=0.1 s[/]\n"
    )


def test_emit_json_prints_only_json_on_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    payload = {"ok": False, "files": [{"path": "a b/ü.json", "errors": [], "n": 1.5}]}
    emit_json(payload)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
    assert captured.err == "" and "\x1b[" not in captured.out  # no colour codes


def test_print_table_folds_instead_of_truncating(capsys: pytest.CaptureFixture[str]) -> None:
    print_table("Generators", ("Name", "Description"), [("grid", LONG)])
    captured = capsys.readouterr()
    assert captured.err == "" and "Generators" in captured.out and "grid" in captured.out
    assert "…" not in captured.out  # no ellipsis (legacy Windows consoles cannot render it)
    assert captured.out.count("x") == len(LONG)  # every character survives, folded


def test_print_summary(capsys: pytest.CaptureFixture[str]) -> None:
    print_summary("Results", {"Vehicles": 12, "Mean delay": "4.2 s"})
    out = capsys.readouterr().out
    assert all(text in out for text in ("Results", "Metric", "Value", "Vehicles", "4.2 s"))
