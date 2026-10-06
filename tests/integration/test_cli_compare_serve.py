from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from urbanflow.cli import main as cli
from urbanflow.cli.main import app

runner = CliRunner()


def test_compare_json(workspace: Path) -> None:
    out = workspace / "cmp.json"
    result = runner.invoke(
        app,
        [
            "compare", "single_intersection", "-c", "fixed_time", "-c", "max_pressure",
            "--seeds", "2", "--duration", "120", "--workers", "1", "--json", "--out", str(out),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["controllers"] == ["fixed_time", "max_pressure"] and len(doc["runs"]) == 4
    assert json.loads(out.read_text(encoding="utf-8"))["baseline"] == "fixed_time"


def _exit(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    return int(info.value.code or 0)


def test_compare_rejects_duplicates(workspace: Path) -> None:
    assert _exit(["compare", "single_intersection", "-c", "a", "-c", "a"]) == 4


def test_serve_refuses_public_host_without_token(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _exit(["serve", "--host", "0.0.0.0"]) == 4
    assert "without an API token" in capsys.readouterr().err
