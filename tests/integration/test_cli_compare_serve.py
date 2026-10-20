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


def test_serve_makes_a_token_for_a_public_host(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    import uvicorn
    from fastapi.testclient import TestClient

    apps: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: apps.append(app))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert _exit(["serve", "--host", "0.0.0.0", "--port", str(port)]) == 0
    token = capsys.readouterr().out.split("#token=")[1].split()[0]
    with TestClient(apps[0]) as c:  # type: ignore[arg-type]
        assert c.get("/api/v1/meta").status_code == 401
        c.cookies.set("urbanflow_token", token)
        assert c.get("/api/v1/meta").status_code == 200
