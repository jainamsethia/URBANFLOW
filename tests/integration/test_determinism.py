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


@pytest.mark.parametrize("kind", ["signalized", "uncontrolled"])
def test_state_digest_is_equal_every_step_for_equal_seeds(kind: str) -> None:
    """F.6: the full engine state (signals, zone locks, RNG streams, queues) matches."""
    scenario = generate("single_intersection", kind=kind, demand_rate=800, duration=400)
    a, b, c = (Simulation(scenario, seed=s, debug_checks=True) for s in (4, 4, 5))
    locks = diverged = 0
    for _ in range(400):
        for sim in (a, b, c):
            sim.step()
        assert a.state_digest() == b.state_digest()
        diverged += a.state_digest() != c.state_digest()
        locks += int((a.state.raw["lock_conn"][a.state.raw["active"]] >= 0).sum())
    assert diverged > 300 and locks > 0  # zone locks were held along the way
    assert a.get_results().state_digest == b.get_results().state_digest
    assert a.get_results().summary == b.get_results().summary
