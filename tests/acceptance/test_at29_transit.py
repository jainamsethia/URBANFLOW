"""AT-29: on the bundled transit scenario buses dwell at every stop for ``dwell ± dt``, and
the other traffic queues behind a dwelling bus or passes it by changing lanes (plan V)."""

from __future__ import annotations

import numpy as np
import pytest

import urbanflow
from urbanflow import EventType, Simulation

pytestmark = pytest.mark.acceptance


def test_buses_serve_every_stop_and_traffic_gets_past() -> None:
    scenario = urbanflow.Scenario.load(urbanflow.bundled("transit"))
    dwell = {s.dwell for line in scenario.resolved.demand.transit for s in line.stops}
    assert dwell == {20.0}
    sim = Simulation(scenario, duration=1800, seed=0, debug_checks=True)
    eng = sim._engine
    veh = eng.vehicles
    passes = 0  # lane changes out of a lane where a bus dwells
    episodes: dict[int, int] = {}  # handle -> steps of its current dwell
    finished: list[int] = []
    while not sim.done:
        run = veh.running()
        stop_lanes = veh.link[run[veh.dwell_left[run] > 0]]
        sim.step()
        ev = eng.events
        changed = ev.type == EventType.vehicle_changed_lane.code
        passes += int(np.isin(ev.aux[changed], stop_lanes).sum())
        run = veh.running()
        now = {int(h) for h in run[veh.dwell_left[run] > 0]}
        for h in now:
            episodes[h] = episodes.get(h, 0) + 1
        finished.extend(episodes.pop(h) for h in set(episodes) - now)
    summary = sim.get_results().summary
    assert summary["transit.stops_missed"] == 0
    assert summary["transit.stops_served"] >= 40  # 2 lines x 6 buses x 5 stops, minus en route
    assert len(finished) == summary["transit.stops_served"]
    assert all(19 <= steps <= 21 for steps in finished)  # dwell 20 s +- dt (dt = 1 s)
    assert passes > 0
