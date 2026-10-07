from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np

from urbanflow import EventType, Simulation, generate
from urbanflow.engine import Engine
from urbanflow.engine.signalling import LaneStats, detector_occupancy
from urbanflow.scenario import Scenario, ScenarioBuilder


def _road(ambulance: bool) -> Scenario:
    # 600 m two-lane road; a slow tractor in lane 0, an ambulance behind it in lane 0
    b = ScenarioBuilder("yield", duration=240)
    b.boundary("W", (0.0, 0.0)).boundary("E", (600.0, 0.0))
    b.road("W_E", "W", "E", lanes=2)
    b.vehicle_type("tractor", max_speed=4.0)
    b.trip("tractor", 0.0, route=["W_E"], vehicle_type="tractor", depart_lane=0)
    if ambulance:
        b.trip("ems", 20.0, route=["W_E"], vehicle_type="emergency", depart_lane=0)
    return b.build()


def _lane_changes(scenario: Scenario) -> Counter[str]:
    sim = Simulation(scenario, seed=0, debug_checks=True)
    changes: Counter[str] = Counter()
    sim.events.subscribe(EventType.vehicle_changed_lane, lambda ev: changes.update([ev.vehicle]))
    result = sim.run()
    assert result.summary["vehicles.arrived"] == sum(1 for t in scenario.resolved.demand.trips)
    return changes


def test_slow_vehicle_moves_over_for_an_ambulance() -> None:
    assert _lane_changes(_road(ambulance=False))["tractor"] == 0
    changes = _lane_changes(_road(ambulance=True))
    assert changes["tractor"] >= 1 and changes["ems"] == 0  # it moved over; ems kept its lane


def test_emergency_movement_from_route_before_lane_change(
    make_engine: Callable[..., Engine],
) -> None:
    base = generate("single_intersection", demand_rate=100, duration=60)
    b = ScenarioBuilder.from_scenario(base)
    b.trip("ems", 0.0, route=["N_in", "W_out"], vehicle_type="emergency")
    e = make_engine(b.build())
    for _ in range(10):  # departs at t = 0, inserted within a few steps
        e.step()
    h = e.vehicles.id_to_handle["ems"]
    e.vehicles.next_conn[h] = -1  # as in a lane that cannot turn right yet
    run = e.vehicles.running()
    occupied = detector_occupancy(e.network, e.vehicles, run)
    net, idle = e.network, np.zeros(e.network.n_lanes)
    pair = (net.road_index["N_in"], net.road_index["W_out"])
    _, conn, _ = LaneStats(net, e.vehicles, e.types, run, occupied, idle, e.routes).emergency
    assert conn.tolist() == [net.road_pair_conns[pair][0]]
    _, conn, _ = LaneStats(net, e.vehicles, e.types, run, occupied, idle).emergency
    assert conn.tolist() == [-1]
