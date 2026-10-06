"""Simulation.snapshot / restore / state_digest and Snapshot files (plan AD.1 8.9, F.5)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from urbanflow import Scenario, Simulation, Snapshot, generate
from urbanflow.core.errors import NotFoundError, SimulationError
from urbanflow.engine.state import EngineState


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return generate("single_intersection", demand_rate=700, duration=900)


def _digests(sim: Simulation, n: int) -> list[str]:
    out = []
    for _ in range(n):
        sim.step()
        out.append(sim.state_digest())
    return out


def test_snapshot_fields_and_digest(scenario: Scenario) -> None:
    sim = Simulation(scenario, seed=3)
    sim.run(until=60)
    snap = sim.snapshot()
    assert isinstance(snap, Snapshot) and isinstance(snap.state, EngineState)
    assert (snap.time, snap.step_count) == (60.0, 60)
    assert snap.scenario_hash == scenario.content_hash and snap.config == sim.config
    assert snap.metrics is not None and "Snapshot(t=60s" in repr(snap)
    digest = sim.state_digest()
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert digest == sim.state_digest()  # reading the digest changes nothing
    assert sim.get_results().state_digest == digest


def test_restore_reproduces_the_run_and_the_summary(scenario: Scenario) -> None:
    sim = Simulation(scenario, seed=3)
    sim.run(until=120)
    snap = sim.snapshot()
    ahead = _digests(sim, 60)
    summary, counts = sim.metrics.summary(), sim.events.counts()
    sim.vehicles.add(route=["W_in", "E_out"])  # undone by the restore
    sim.restore(snap)
    assert sim.step_count == 120 and sim.time == 120.0
    assert _digests(sim, 60) == ahead
    after = sim.metrics.summary()
    assert {k: v for k, v in after.items() if not math.isnan(v)} == {
        k: v for k, v in summary.items() if not math.isnan(v)
    }
    assert sim.events.counts() == counts
    sim.restore(snap)  # snapshots are reusable
    assert _digests(sim, 60) == ahead


def test_file_round_trip_resets_the_summary_with_a_warning(
    scenario: Scenario, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sim = Simulation(scenario, seed=3)
    sim.run(until=90)
    path = sim.snapshot().save(tmp_path / "t90.ufs")
    assert path == tmp_path / "t90.ufs"
    ahead = _digests(sim, 50)
    loaded = Snapshot.load(path)
    assert loaded.metrics is None and loaded.config == sim.config and loaded.time == 90.0
    other = Simulation(scenario, seed=3)  # another instance, same scenario and config
    with caplog.at_level(logging.WARNING, logger="urbanflow.simulation"):
        other.restore(loaded)
    assert "restored a snapshot loaded from a file (t=90 s)" in caplog.text
    summary = other.metrics.summary()
    assert summary["vehicles.inserted"] == 0  # the accumulators restart ...
    assert summary["vehicles.generated"] > 0  # ... the engine counters come with the state
    assert _digests(other, 50) == ahead
    with pytest.raises(NotFoundError, match="no saved state"):
        Snapshot.load(tmp_path / "missing.ufs")


def test_restore_checks_scenario_and_config(scenario: Scenario) -> None:
    snap = Simulation(scenario).snapshot()
    with pytest.raises(SimulationError, match="another scenario"):
        Simulation(generate("single_intersection", kind="priority")).restore(snap)
    with pytest.raises(SimulationError, match="differs in: duration"):
        Simulation(scenario, duration=300).restore(snap)


def test_restore_checks_the_configured_controllers(scenario: Scenario, tmp_path: Path) -> None:
    """P4 review repro: a snapshot of a ``controllers={"J": "external"}`` run restored into a
    fixed-time simulation, which then ran external control while ``get_results()`` and
    ``reset()`` still used fixed_time. The configured controllers are saved and checked."""
    ext = Simulation(scenario, controllers={"J": "external"})
    ext.run(until=30)
    snap = ext.snapshot()
    assert snap.state.meta["configured"]["J"]["type"] == "external"
    message = r'controllers configured at "J" \(external in the snapshot, fixed_time here\)'
    plain = Simulation(scenario)
    with pytest.raises(SimulationError, match=message):
        plain.restore(snap)
    with pytest.raises(SimulationError, match=message):  # file snapshots too
        plain.restore(Snapshot.load(snap.save(tmp_path / "ext.ufs")))
    plain.step()  # nothing was changed
    same = Simulation(scenario, controllers={"J": "external"})
    same.restore(snap)
    assert same.get_results().controllers["J"]["type"] == "external"


def test_restore_clears_the_corrupted_flag_and_the_caches(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = Simulation(scenario)
    sim.run(until=30)
    snap = sim.snapshot()
    speeds = sim.state.speeds  # cached for this step
    engine = sim._engine

    def interrupted() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "step", interrupted)
    with pytest.raises(KeyboardInterrupt):
        sim.step()
    monkeypatch.undo()
    with pytest.raises(SimulationError, match="corrupted"):
        sim.step()
    # P4 review: a half-applied step must not be saved (restoring it would revive it)
    with pytest.raises(SimulationError, match=r"corrupted .*reset\(\) or restore\(\)"):
        sim.snapshot()
    with pytest.raises(SimulationError, match="corrupted"):
        sim.state_digest()
    partial = sim.get_results()  # still valid, without a digest of the broken state
    assert partial.interrupted and partial.state_digest == ""
    sim.restore(snap)
    assert sim.state.speeds is not speeds  # recomputed after the restore
    sim.step()
    assert sim.step_count == 31 and not sim.get_results().interrupted


def test_restore_after_reset_with_another_seed(scenario: Scenario) -> None:
    sim = Simulation(scenario, seed=1)
    sim.run(until=40)
    snap = sim.snapshot()
    ahead = _digests(sim, 20)
    sim.reset(seed=8)
    sim.restore(snap)
    assert sim.seed == 1  # the snapshot's streams and seed
    assert _digests(sim, 20) == ahead


def test_restore_on_a_closed_simulation(scenario: Scenario) -> None:
    sim = Simulation(scenario)
    snap = sim.snapshot()
    sim.close()
    with pytest.raises(SimulationError, match="closed"):
        sim.restore(snap)
