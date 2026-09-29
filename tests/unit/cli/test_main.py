from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import typer

from urbanflow._version import __version__
from urbanflow.cli import main as cli
from urbanflow.core.errors import NotFoundError, ScenarioValidationError, ValidationIssue


def run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    return int(info.value.code or 0)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"urbanflow {__version__} (")
    assert "numpy" in out


def test_help_lists_panels(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--help"]) == 0
    out = capsys.readouterr().out
    assert "Workbench" in out and "doctor" in out and "--workspace" in out


def test_unknown_command_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["frobnicate"]) == 2


def test_bad_log_level_is_config_error(capsys: pytest.CaptureFixture[str], workspace: Path) -> None:
    assert run(["--log-level", "LOUD", "doctor"]) == 4
    assert "log-level" in capsys.readouterr().err


@pytest.fixture
def boom_app(monkeypatch: pytest.MonkeyPatch) -> typer.Typer:
    """Temporarily register commands that raise, to test the exit-code mapping."""
    app = cli.app

    def validation() -> None:
        raise ScenarioValidationError(
            [ValidationIssue("demand.flows[0].route[1]", 'unknown road "x"', "E502")]
        )

    def missing() -> None:
        raise NotFoundError("scenario file not found: gird.json (did you mean grid.json?)")

    def crash() -> None:
        raise RuntimeError("kaboom")

    before = list(app.registered_commands)
    app.command("x-validation")(validation)
    app.command("x-missing")(missing)
    app.command("x-crash")(crash)
    yield app
    app.registered_commands[:] = before


def test_exit_code_mapping(
    boom_app: typer.Typer, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    assert run(["x-validation"]) == 3
    err = capsys.readouterr().err
    assert (
        "Scenario validation failed:" in err and 'demand.flows[0].route[1]: unknown road "x"' in err
    )
    assert "Traceback" not in err

    assert run(["x-missing"]) == 5
    assert "did you mean grid.json" in capsys.readouterr().err

    assert run(["x-crash"]) == 1
    err = capsys.readouterr().err
    assert "Internal error" in err and "--debug" in err and "Traceback" not in err

    assert run(["--debug", "x-crash"]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_python_dash_m_entry_point() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "urbanflow", "--version"], capture_output=True, text=True
    )
    assert out.returncode == 0 and out.stdout.startswith("urbanflow ")


def test_platform_tag() -> None:
    tag = cli.platform_tag()
    assert "-" in tag and tag.split("-")[0] in {"win", "linux", "macos"} | {sys.platform}
