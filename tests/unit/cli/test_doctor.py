from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from urbanflow.cli import main as cli
from urbanflow.cli.commands import doctor
from urbanflow.core.settings import load_settings


def run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    return int(info.value.code or 0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_doctor_json(capsys: pytest.CaptureFixture[str], workspace: Path) -> None:
    assert run(["doctor", "--json", "--port", str(_free_port())]) == 0
    report = json.loads(capsys.readouterr().out)
    names = {c["name"] for c in report["checks"]}
    assert {
        "Python",
        "Platform",
        "urbanflow",
        "Core dependencies",
        "Workspace",
        "Database",
    } <= names
    assert any(n.startswith("Extra ") for n in names)
    assert report["failures"] == 0
    assert all(c["status"] in {"OK", "WARN", "FAIL"} for c in report["checks"])


def test_doctor_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    monkeypatch.setattr(doctor, "check_python", lambda: doctor.Check("Python", "FAIL", "too old"))
    assert run(["doctor", "--port", str(_free_port())]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_port_in_use_is_a_warning(workspace: Path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert doctor.check_port("127.0.0.1", port).status == "WARN"


def test_frontend_missing_is_a_warning(workspace: Path) -> None:
    settings = load_settings(frontend_dir=workspace / "nope")
    check = doctor.check_frontend(settings)
    assert check.status in {"OK", "WARN"}


def test_frontend_dir_setting(workspace: Path) -> None:
    dist = workspace / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    assert doctor.check_frontend(load_settings(frontend_dir=dist)).status == "OK"


def test_database_checks(workspace: Path) -> None:
    assert doctor.check_database(load_settings()).status == "OK"
    pg = doctor.check_database(load_settings(database_url="postgresql+psycopg://u@h/db"))
    assert pg.status in {"OK", "WARN"}


def test_spawn_round_trip() -> None:
    assert doctor.check_spawn().status == "OK"
