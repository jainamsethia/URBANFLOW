"""AT-03: friendly, path-addressed validation errors without tracebacks (plan section V)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance

DEMO = Path(__file__).resolve().parents[1] / "fixtures" / "scenarios" / "demo.json"
EXPECTED = [
    "Scenario validation failed:",
    '  - network.intersections[0].signal.phases[1]: phase contains unknown movement "E_in->W_ot" '
    '(did you mean "E_in->W_out"?)',
    '  - demand.flows[3].route[1]: route is not connected: no movement from "E_in" to "E_out" '
    'at intersection "J"',
    '  - demand.trips[0].vehicle_type: unknown vehicle type "cr" (did you mean "car"?) '
    "(available: bus, car, city_bus, emergency, truck)",
    "Warnings:",
    '  - network.intersections[0].signal: movement "E_in->W_out" is never green in any phase; '
    "its traffic can never enter",
]


@pytest.fixture
def broken(tmp_path: Path) -> Path:
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    green = data["network"]["intersections"][0]["signal"]["phases"][1]["green"]
    green["E_in->W_ot"] = green.pop("E_in->W_out")  # unknown movement in a phase
    data["demand"]["flows"][3]["route"] = ["E_in", "E_out"]  # disconnected route
    data["demand"]["trips"][0]["vehicle_type"] = "cr"  # bad vehicle type
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _validate(path: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "urbanflow", *flags, "validate", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_friendly_error_list(broken: Path) -> None:
    result = _validate(broken)
    assert result.returncode == 3
    assert result.stdout == ""
    assert result.stderr.splitlines() == EXPECTED
    assert "Traceback" not in result.stderr


def test_debug_adds_a_traceback(broken: Path) -> None:
    result = _validate(broken, "--debug")
    assert result.returncode == 3
    assert "Traceback (most recent call last)" in result.stderr
    assert "ScenarioValidationError" in result.stderr
    assert all(line in result.stderr for line in EXPECTED)
