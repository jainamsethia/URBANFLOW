"""AT-01: a fresh environment can run --help and doctor successfully (plan section V)."""

from __future__ import annotations

import json
import socket
import subprocess
import sys

import pytest

pytestmark = pytest.mark.acceptance


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "urbanflow", *args], capture_output=True, text=True, timeout=120
    )


def test_help_shows_command_groups() -> None:
    result = _cli("--help")
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout and "doctor" in result.stdout


def test_doctor_reports_no_failures(tmp_path: pytest.TempPathFactory) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    result = _cli("--workspace", str(tmp_path), "doctor", "--json", "--port", str(port))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["failures"] == 0
