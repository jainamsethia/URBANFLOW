"""AT-30: every built-in generator, with its defaults, gives a valid scenario with a
deterministic content hash that runs for 300 s with the debug invariants on (plan V)."""

from __future__ import annotations

import pytest

from urbanflow import Simulation, generate
from urbanflow.scenario.generators import list_generators

pytestmark = pytest.mark.acceptance

BUILTIN = [name for name, entry in list_generators() if entry.source == "builtin"]


def test_the_plan_generators_are_built_in() -> None:
    expected = {"single_intersection", "corridor", "grid", "downtown", "high_density"}
    assert expected | {"rush_hour", "emergency", "transit"} <= set(BUILTIN)


@pytest.mark.parametrize("name", BUILTIN)
def test_generator_defaults_run(name: str) -> None:
    scenario = generate(name)
    assert not [i for i in scenario.issues if i.severity.value == "error"]
    assert generate(name).content_hash == scenario.content_hash
    sim = Simulation(scenario, seed=0, debug_checks=True)
    result = sim.run(until=300)
    assert result.summary["vehicles.generated"] > 0
