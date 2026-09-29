"""Analytic physics references (plan R.2): free road, IDM equilibrium, stopping, dt parity."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from urbanflow.core import constants as C
from urbanflow.core.events import EventType
from urbanflow.engine import Engine
from urbanflow.scenario import Scenario, ScenarioBuilder

pytestmark = pytest.mark.integration

MakeEngine = Callable[..., Engine]
DET = {"speed_factor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}}
V0 = C.SPEED_LIMIT  # det cars drive exactly at the limit


def _straight(length: float, **trip: Any) -> Scenario:
    """One boundary-to-boundary road "AB" of ``length`` m and one deterministic trip "v"."""
    b = ScenarioBuilder("straight", duration=3600)
    b.vehicle_type("det", **DET)
    b.boundary("A", (0.0, 0.0))
    b.boundary("B", (length, 0.0))
    b.road("AB", "A", "B")
    b.trip("v", 0.0, route=["AB"], vehicle_type="det", **trip)
    return b.build()


def _arrival(e: Engine, steps: int) -> tuple[float, list[float]]:
    """Run until the single vehicle arrives: (arrival time, its speed after every step)."""
    speeds: list[float] = []
    for _ in range(steps):
        e.step()
        ev = e.events
        done = ev.time[ev.type == EventType.vehicle_arrived.code]
        if done.size:
            return float(done[0]), speeds
        speeds.append(float(e.vehicles.speed[e.vehicles.running()[0]]))
    raise AssertionError("the vehicle did not arrive")


def _ballistic_reference(distance: float) -> float:
    """Free-road IDM from rest over ``distance`` m (RK4, h = 1 ms): the travel time."""
    a, delta, h = C.IDM_ACCEL, C.IDM_DELTA, 1e-3

    def f(v: float) -> float:
        return a * (1 - (v / V0) ** delta)

    x = v = t = 0.0
    while True:
        k1 = f(v)
        k2 = f(v + h * k1 / 2)
        k3 = f(v + h * k2 / 2)
        k4 = f(v + h * k3)
        dv = h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        dx = h * (v + dv / 2)
        if x + dx >= distance:
            return t + h * (distance - x) / dx
        x, v, t = x + dx, v + dv, t + h


@pytest.mark.parametrize("dt", [0.25, 0.5, 1.0])
def test_free_road_reaches_v0_on_the_ballistic_travel_time(
    make_engine: MakeEngine, dt: float
) -> None:
    e = make_engine(_straight(1000.0, depart_speed=0.0), dt=dt)
    arrival, speeds = _arrival(e, int(200 / dt))
    reference = _ballistic_reference(1000.0 - C.VEHICLE_LENGTH)  # inserted with pos = length
    assert abs(arrival - reference) < 0.5
    assert max(speeds) <= V0 + 1e-9
    assert speeds[-1] >= 0.99 * V0


def test_dt_parity_of_free_flow_travel_time(make_engine: MakeEngine) -> None:
    times = []
    for dt in (0.25, 0.5, 1.0):
        e = make_engine(_straight(1000.0, depart_speed=0.0), dt=dt)
        times.append(_arrival(e, int(200 / dt))[0])
    assert max(times) / min(times) - 1 < 0.02


def test_interpolated_arrival_time(make_engine: MakeEngine) -> None:
    e = make_engine(_straight(1000.0), dt=1.0)  # depart "max": cruising at v0
    arrival, _ = _arrival(e, 200)
    assert arrival == pytest.approx((1000.0 - C.VEHICLE_LENGTH) / V0, abs=1e-9)
    assert arrival % 1.0 > 1e-3  # inside a step, not on the grid


def test_idm_equilibrium_gap(make_engine: MakeEngine) -> None:
    b = ScenarioBuilder("follow", duration=3600)
    b.vehicle_type("det", **DET)
    b.boundary("A", (0.0, 0.0))
    b.boundary("B", (5000.0, 0.0))
    b.road("AB", "A", "B")
    b.trip("lead", 0.0, route=["AB"], vehicle_type="det")
    b.trip("follow", 4.0, route=["AB"], vehicle_type="det")
    e = make_engine(b.build(), dt=0.5)
    e.step()
    v = 10.0
    e.commands.set_speed("lead", v)
    for _ in range(600):
        e.step()
    veh = e.vehicles
    lead, follow = veh.handle_of("lead"), veh.handle_of("follow")
    gap = veh.pos[lead] - veh.length[lead] - veh.pos[follow]
    s_e = (C.IDM_MIN_GAP + v * C.IDM_HEADWAY) / math.sqrt(1 - (v / V0) ** C.IDM_DELTA)
    assert veh.speed[follow] == pytest.approx(v, abs=1e-3)
    assert gap == pytest.approx(s_e, rel=0.01)


def test_stops_before_the_lane_end(make_engine: MakeEngine) -> None:
    """A vehicle on a lane without a connector to its next road waits at the lane end."""
    b = ScenarioBuilder("dead_end", duration=600)
    b.vehicle_type("det", **DET)
    b.boundary("W", (-200.0, 0.0))
    b.intersection("J", (0.0, 0.0), kind="uncontrolled")
    b.boundary("E", (100.0, 0.0))
    b.road("W_in", "W", "J", lanes=2)
    b.road("E_out", "J", "E")
    b.movement("J", "W_in", "E_out", connections=[(1, 0)])
    b.trip("v", 0.0, route=["W_in", "E_out"], vehicle_type="det", depart_lane=0)
    e = make_engine(b.build(), dt=0.5)
    veh = e.vehicles
    decel = []
    for _ in range(120):
        e.step()
        decel.append(-float(veh.accel[veh.handle_of("v")]))
    h = veh.handle_of("v")
    end = float(e.network.link_length[veh.link[h]])
    assert veh.next_conn[h] == -1 and veh.held[h]
    assert end - 0.6 <= veh.pos[h] <= end - 0.4
    assert veh.speed[h] == pytest.approx(0.0, abs=1e-3)
    assert max(decel) <= C.IDM_DECEL
    assert e.safety_cap_violations == 0


def test_multi_hop_in_one_step(
    make_engine: MakeEngine, junction_builder: Callable[..., ScenarioBuilder]
) -> None:
    """At dt = 2 s a fast vehicle crosses a whole straight connector within one step."""
    b = junction_builder(arm=210.0)  # a step starts 5.3 m before the stop line
    b.vehicle_type("det", **DET)
    b.trip("v", 0.0, route=["W_in", "E_out"], vehicle_type="det")
    e = make_engine(b.build(), dt=2.0)
    net = e.network
    conn = net.link_index["W_in_0->E_out_0"]
    assert net.link_length[conn] < V0 * 2.0
    hops = []
    for _ in range(40):
        e.step()
        ev = e.events
        entered = ev.link[ev.type == EventType.vehicle_entered_link.code].tolist()
        hops.append(entered)
    assert [conn, net.link_index["E_out_0"]] in hops
    assert np.isfinite(e.vehicles.pos).all()
