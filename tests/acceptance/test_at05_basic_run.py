"""AT-05 (user TEST 1): a 600 s run at one intersection (plan section V), uncontrolled
(P2) and signalised (the bundled fixed-time junction, from P3), with ``debug_checks``.

Every vehicle's position is non-decreasing within a link; its speed never exceeds
max(v0 on its current link, v0 on the link it occupied at step start) + 1e-6; vehicles
arrive; the mean travel time is at least the mean free-flow time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from urbanflow import Scenario, Simulation, bundled, generate
from urbanflow.core.types import VehicleStatus

pytestmark = pytest.mark.acceptance

RUNNING = VehicleStatus.running.code
ARRIVED = VehicleStatus.arrived.code


@pytest.mark.parametrize("kind", ["uncontrolled", "signalized"])
@pytest.mark.parametrize("seed", [0, 11])
def test_at05_basic_run(seed: int, kind: str) -> None:
    if kind == "signalized":
        scenario = Scenario.load(bundled("single_intersection"))
    else:
        scenario = generate("single_intersection", kind="uncontrolled")
    sim = Simulation(scenario, seed=seed, duration=600, debug_checks=True)
    free_flow: list[float] = []
    while not sim.done:
        raw = sim.state.raw  # live slot views (read-only)
        slots = np.flatnonzero(raw["active"])
        uid = raw["uid"][slots].copy()
        link, pos, v0 = raw["link"][slots].copy(), raw["pos"][slots].copy(), raw["v0"][slots].copy()
        sim.step()
        raw = sim.state.raw  # the table may have grown
        # a slot freed in this step is reused only in the next one, so it still names `uid`
        assert np.array_equal(raw["uid"][slots], uid)
        still = raw["status"][slots] == RUNNING
        same_link = still & (raw["link"][slots] == link)
        assert np.all(raw["pos"][slots][same_link] >= pos[same_link] - 1e-9)
        bound = np.maximum(v0, raw["v0"][slots]) + 1e-6
        assert np.all(raw["speed"][slots][still] <= bound[still])
        arrived = raw["status"][slots] == ARRIVED
        free_flow += raw["ff_time"][slots][arrived].tolist()
    summary = sim.get_results().summary
    assert summary["vehicles.arrived"] > 0
    assert len(free_flow) == summary["vehicles.arrived"]
    assert summary["travel_time.mean"] >= np.mean(free_flow)


def test_p2_demo_from_the_command_line(tmp_path: Path) -> None:
    """``urbanflow generate ... -p kind=uncontrolled`` then ``urbanflow run --duration 600``."""

    def cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "urbanflow", *args],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=tmp_path,
        )

    generated = cli("generate", "single_intersection", "-p", "kind=uncontrolled", "-o", "s.json")
    assert generated.returncode == 0, generated.stderr
    ran = cli("run", "s.json", "--duration", "600", "--json")
    assert ran.returncode == 0, ran.stderr
    result = json.loads(ran.stdout)
    assert result["sim_time"] == 600.0 and result["summary"]["vehicles.arrived"] > 0
    assert (tmp_path / "runs" / result["run_id"] / "summary.json").is_file()


def test_p3_demo_signalised_run(tmp_path: Path) -> None:
    """``urbanflow run src/urbanflow/scenario/bundled/single_intersection.json --duration 600``
    prints a summary with arrivals, and ``sim.signals["J"]`` shows the current phase."""
    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "urbanflow" / "scenario" / "bundled" / "single_intersection.json"
    ran = subprocess.run(
        [sys.executable, "-m", "urbanflow", "run", str(path), "--duration", "600", "--no-save"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env={**os.environ, "NO_COLOR": "1", "COLUMNS": "120"},
    )
    assert ran.returncode == 0, ran.stderr
    assert "Results: single_intersection (seed 0, 600 s)" in ran.stdout
    row = next(line for line in ran.stdout.splitlines() if "departed / arrived" in line)
    arrived = int(row.split("/")[-1].split()[0].replace(",", ""))
    assert arrived > 0
    sim = Simulation(bundled("single_intersection"), duration=600)
    sim.run(until=45)
    view = sim.signals["J"]
    assert (view.phase_id, view.stage.value, view.state_string) == ("p1", "green", "rrrgGGGGgrrr")
