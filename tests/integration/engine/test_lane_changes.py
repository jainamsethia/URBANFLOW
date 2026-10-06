"""Lane changes (plan G.4, AT-18 core): MOBIL overtaking, safety, no shadow vehicles."""

from __future__ import annotations

import numpy as np

from urbanflow import ScenarioBuilder, Simulation, bundled
from urbanflow.core.events import EventType


def _two_lane_road(*trips: tuple[str, float, str, int]) -> Simulation:
    b = ScenarioBuilder("two_lanes", duration=600)
    b.vehicle_type(
        "slow", length=8.0, max_speed=4.0, speed_factor={"mean": 1, "std": 0, "min": 1, "max": 1}
    )
    b.vehicle_type("fast", max_speed=20.0, speed_factor={"mean": 1, "std": 0, "min": 1, "max": 1})
    b.boundary("A", (0.0, 0.0)).boundary("B", (1500.0, 0.0))
    b.road("AB", "A", "B", lanes=2, speed_limit=20.0)
    for vid, depart, vtype, lane in trips:
        b.trip(vid, depart, route=["AB"], vehicle_type=vtype, depart_lane=lane)
    return Simulation(b.build(), debug_checks=True)


def test_a_fast_car_overtakes_a_slow_vehicle() -> None:
    sim = _two_lane_road(("slow", 0.0, "slow", 0), ("car", 8.0, "fast", 0))
    changes = 0
    passed = False
    while not sim.done and sim.metrics.summary()["vehicles.arrived"] < 2:
        sim.step()
        changes += int(
            np.count_nonzero(sim._engine.events.type == EventType.vehicle_changed_lane.code)
        )
        v = sim.vehicles
        if "car" in v and "slow" in v and v["car"].position > v["slow"].position:
            passed = True
    assert changes >= 1 and passed
    trips = sim.get_results().tables["trips"]
    order = trips["vehicle_id"].tolist()
    assert order.index("car") < order.index("slow")  # the car arrived first


def test_no_change_into_an_unsafe_gap() -> None:
    """A car beside a faster car on the other lane must not cut in front of it."""
    sim = _two_lane_road(
        ("slow", 0.0, "slow", 0), ("car", 6.0, "fast", 0), ("blocker", 6.0, "fast", 1)
    )
    sim.run()
    s = sim.get_results().summary
    assert s["vehicles.arrived"] == 3
    assert sim._engine.safety_cap_violations == 0


def test_lane_counts_have_no_shadows_and_runs_are_deterministic() -> None:
    def run(seed: int) -> tuple[str, int]:
        sim = Simulation.from_scenario(
            bundled("grid_3x3"), seed=seed, duration=600, debug_checks=True
        )
        total_changes = 0
        for _ in range(600):
            sim.step()
            counts = sim.lanes.vehicle_counts().sum()
            on_conn = int(
                np.count_nonzero(
                    sim._engine.vehicles.link[sim._engine.vehicles.active]
                    >= sim.network.compiled.n_lanes
                )
            )
            assert counts + on_conn == len(sim.vehicles)
            total_changes += int(
                np.count_nonzero(sim._engine.events.type == EventType.vehicle_changed_lane.code)
            )
        return sim.state_digest(), total_changes

    a, b = run(4), run(4)
    assert a == b and a[1] > 0


def test_lane_changing_can_be_switched_off() -> None:
    sim = Simulation.from_scenario(bundled("grid_3x3"), seed=1, duration=300, lane_changing=False)
    result = sim.run()
    assert result.event_counts.get("vehicle_changed_lane", 0) == 0
