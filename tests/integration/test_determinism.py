"""Determinism through the facade (plan F.6, AA 5.5, AD.1 8.9)."""

from __future__ import annotations

import numpy as np
import pytest

from urbanflow import Scenario, Simulation, generate

pytestmark = pytest.mark.integration

FIELDS = (
    "uids",
    "type_idx",
    "links",
    "positions",
    "speeds",
    "accels",
    "xy",
    "headings",
    "waiting_times",
    "distances",
)
STEPS = 300


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return generate("single_intersection", kind="uncontrolled", demand_rate=900, duration=600)


def _state(sim: Simulation) -> dict[str, np.ndarray]:
    return {name: getattr(sim.state, name) for name in FIELDS}


def _equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return all(a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in FIELDS)


def test_same_seed_same_state_every_step(scenario: Scenario) -> None:
    a, b, c = (
        Simulation(scenario, seed=11),
        Simulation(scenario, seed=11),
        Simulation(scenario, seed=12),
    )
    diverged = False
    for _ in range(STEPS):
        for sim in (a, b, c):
            sim.step()
        assert _equal(_state(a), _state(b))
        assert a.state.ids == b.state.ids
        diverged |= not _equal(_state(a), _state(c))
    assert diverged
    assert a.metrics.summary() == b.metrics.summary()
    assert a.get_results().event_counts == b.get_results().event_counts


def test_reset_with_a_seed_reproduces_from_scratch(scenario: Scenario) -> None:
    fresh = Simulation(scenario, seed=21)
    sim = Simulation(scenario, seed=5)
    sim.run(until=120)
    sim.vehicles.add(route=["E_in", "W_out"])
    sim.reset(seed=21)
    for _ in range(STEPS):
        sim.step()
        fresh.step()
        assert _equal(_state(sim), _state(fresh))
    assert sim.get_results().summary == fresh.get_results().summary
