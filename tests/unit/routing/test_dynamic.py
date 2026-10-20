"""The dynamic router sends traffic around a congested route; it is deterministic."""

from __future__ import annotations

from urbanflow import EventType, Simulation, bundled
from urbanflow.scenario import Scenario, ScenarioBuilder


def _fork(rate: float = 1500.0) -> Scenario:
    # O -> A, then north A -> B -> D (shorter, one lane) or south A -> C -> D, then D -> E
    b = ScenarioBuilder("fork", duration=900)
    b.boundary("O", (-300.0, 0.0)).boundary("E", (900.0, 0.0))
    b.intersection("A", (0.0, 0.0), kind="uncontrolled")
    b.intersection("B", (300.0, 40.0), kind="uncontrolled")
    b.intersection("C", (300.0, -55.0), kind="uncontrolled")
    b.intersection("D", (600.0, 0.0), kind="uncontrolled")
    b.road("O_A", "O", "A", lanes=2)
    for road, a, z in (("A_B", "A", "B"), ("B_D", "B", "D"), ("A_C", "A", "C"), ("C_D", "C", "D")):
        b.road(road, a, z, lanes=1)
    b.road("D_E", "D", "E", lanes=2)
    b.flow("od", origin="O_A", destination="D_E", rate=rate, arrival="poisson")
    return b.build()


def _run(router: str, seed: int = 0) -> tuple[dict[str, float], int, str]:
    sim = Simulation(_fork(), router=router, seed=seed, reroute_period=30)
    south: set[str] = set()
    sim.events.subscribe(
        EventType.vehicle_entered_link,
        lambda ev: south.add(ev.vehicle) if ev.link and ev.link.startswith("A_C") else None,
    )
    summary = sim.run().summary
    return summary, len(south), sim.state_digest()


def test_dynamic_router_uses_the_detour() -> None:
    static, static_south, _ = _run("shortest")
    dynamic, dyn_south, digest = _run("dynamic")
    assert static_south == 0  # everyone takes the shorter north lane and it backs up
    assert dyn_south > 0.25 * dynamic["vehicles.arrived"]  # learnt times send cars south
    assert dynamic["vehicles.backlog"] < static["vehicles.backlog"]
    assert dynamic["vehicles.arrived"] > static["vehicles.arrived"]
    assert _run("dynamic")[2] == digest  # deterministic


def test_no_change_on_a_uniform_grid() -> None:
    # equal alternatives look equal (both include their signal waits): nobody reroutes
    runs = [
        Simulation.from_scenario(bundled("grid_3x3"), router=r, seed=0, duration=600).run()
        for r in ("shortest", "dynamic")
    ]
    assert runs[0].summary["travel_time.mean"] == runs[1].summary["travel_time.mean"]


def test_snapshot_round_trip() -> None:
    sim = Simulation(_fork(), router="dynamic", seed=1, reroute_period=30)
    sim.run(until=200)
    snap = sim.snapshot()
    sim.run(until=400)
    digest = sim.state_digest()
    sim.restore(snap)
    sim.run(until=400)
    assert sim.state_digest() == digest
