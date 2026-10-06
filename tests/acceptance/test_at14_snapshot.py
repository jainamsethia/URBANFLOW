"""AT-14: snapshot at t = 300 -> 100 steps -> restore -> 100 steps (plan V, F.5, AD.1 8.9).

The bundled signalised junction with every invariant checked. The digests of the 100 steps
after the restore equal those of the first 100 steps, both for the in-memory snapshot and
for a new simulation restored from the saved file. Restoring onto another scenario raises
``SimulationError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from urbanflow import Scenario, Simulation, SimulationError, Snapshot, bundled, generate

pytestmark = pytest.mark.acceptance

STEPS = 100


def _digests(sim: Simulation) -> list[str]:
    out = []
    for _ in range(STEPS):
        sim.step()
        out.append(sim.state_digest())
    return out


def test_at14_snapshot_restore_round_trip(tmp_path: Path) -> None:
    scenario = Scenario.load(bundled("single_intersection"))
    sim = Simulation(scenario, seed=7, debug_checks=True)
    sim.run(until=300)
    snap = sim.snapshot()
    at_300 = sim.state_digest()
    path = snap.save(tmp_path / "t300.ufs")
    first = _digests(sim)
    assert len(set(first)) == STEPS  # the state moves every step

    sim.restore(snap)
    assert (sim.time, sim.state_digest()) == (300.0, at_300)
    assert _digests(sim) == first

    other = Simulation(scenario, seed=7, debug_checks=True)  # through the file
    other.restore(Snapshot.load(path))
    assert other.state_digest() == at_300
    assert _digests(other) == first

    elsewhere = Simulation(generate("single_intersection", kind="priority"))
    with pytest.raises(SimulationError, match="the snapshot is of another scenario"):
        elsewhere.restore(snap)
    with pytest.raises(SimulationError, match="another scenario"):
        elsewhere.restore(Snapshot.load(path))
