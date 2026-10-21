"""Transit buses: stopping, dwelling, far-side stops, missed stops and snapshots (plan G.1)."""

from __future__ import annotations

from typing import Any

import pytest

from urbanflow import Simulation
from urbanflow.scenario import Scenario, ScenarioBuilder

DWELL = 20.0


def _bus_line(stops: list[tuple[str, float]], *, dt: float = 1.0, cars: float = 0.0) -> Scenario:
    """One bus on W -> J -> E (two lanes, an uncontrolled junction), optionally with cars."""
    b = ScenarioBuilder("bus", duration=300, dt=dt)
    b.boundary("W", (-400.0, 0.0)).boundary("E", (400.0, 0.0))
    b.intersection("J", (0.0, 0.0), kind="uncontrolled")
    b.road("W_J", "W", "J", lanes=2).road("J_E", "J", "E", lanes=2)
    b.transit_line("L", ["W_J", "J_E"], stops, headway=600.0, end=1.0, dwell=DWELL)
    if cars:
        b.flow("cars", route=["W_J", "J_E"], rate=cars, arrival="poisson")
    return b.build()


def _dwells(sim: Simulation) -> list[tuple[str, float, int]]:
    """Run to the end; (lane, front position, steps) of every dwell of the bus."""
    eng = sim._engine
    out: list[tuple[str, float, int]] = []
    current: list[Any] | None = None
    while not sim.done:
        sim.step()
        veh = eng.vehicles
        h = veh.id_to_handle.get("L.0")
        dwelling = h is not None and veh.dwell_left[h] > 0
        if dwelling and current is None:
            current = [eng.network.link_ids[int(veh.link[h])], float(veh.pos[h]), 0]
            out.append(current)  # type: ignore[arg-type]
        if dwelling and current is not None:
            current[2] += 1
        if not dwelling:
            current = None
    return [tuple(d) for d in out]  # type: ignore[misc]


@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_bus_dwells_at_every_stop(dt: float) -> None:
    # the J_E stop is 30 m past the junction: the bus must brake for it before the junction
    sim = Simulation(_bus_line([("W_J", 150.0), ("J_E", 30.0)], dt=dt), debug_checks=True)
    dwells = _dwells(sim)
    assert [lane for lane, _, _ in dwells] == ["W_J_1", "J_E_1"]  # the curb lanes
    for (_, pos, steps), stop in zip(dwells, (150.0, 30.0), strict=True):
        assert abs(pos - stop) <= 1.0
        assert DWELL - dt <= steps * dt <= DWELL + dt
    summary = sim.get_results().summary
    assert (summary["transit.stops_served"], summary["transit.stops_missed"]) == (2, 0)
    assert summary["vehicles.arrived"] == 1


def test_a_stop_the_bus_cannot_stop_for_is_missed() -> None:
    # inserted at full speed with its front 0.5 m before the stop: it cannot stop there
    sim = Simulation(_bus_line([("W_J", 12.5), ("J_E", 100.0)]))
    dwells = _dwells(sim)
    assert [lane for lane, _, _ in dwells] == ["J_E_1"]
    summary = sim.get_results().summary
    assert (summary["transit.stops_served"], summary["transit.stops_missed"]) == (1, 1)


def test_snapshot_during_a_dwell_round_trips() -> None:
    sim = Simulation(_bus_line([("W_J", 150.0), ("J_E", 60.0)], cars=600.0), seed=3)
    eng = sim._engine
    while not (eng.vehicles.dwell_left > 0).any():
        sim.step()
    snap = sim.snapshot()
    sim.run(until=sim.time + 60)
    digest = sim.state_digest()
    sim.restore(snap)
    sim.run(until=sim.time + 60)
    assert sim.state_digest() == digest
