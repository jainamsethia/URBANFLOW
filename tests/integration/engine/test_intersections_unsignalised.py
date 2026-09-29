"""Unsignalised intersections end to end (plan F.3, R.2 intersection tests, AT-13 subset)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from urbanflow import generate
from urbanflow.core.events import EventType
from urbanflow.engine import Engine
from urbanflow.network.conflicts import ConflictKind
from urbanflow.scenario import ScenarioBuilder

pytestmark = pytest.mark.integration

MakeEngine = Callable[..., Engine]
Builder = Callable[..., ScenarioBuilder]
DET = {"speed_factor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}}
WE, NS = ["W_in", "E_out"], ["N_in", "S_out"]


def _two_trips(
    make_engine: MakeEngine, junction_builder: Builder, kind: str, t_we: float, t_ns: float
) -> tuple[dict[str, float], Engine, list[bool]]:
    """Run W->E and N->S (crossing) trips; travel times and, per step, zone co-occupancy."""
    b = junction_builder(kind=kind)
    b.vehicle_type("det", **DET)
    b.trip("we", t_we, route=WE, vehicle_type="det")
    b.trip("ns", t_ns, route=NS, vehicle_type="det")
    e = make_engine(b.build(), dt=0.5)
    net, veh = e.network, e.vehicles
    conns = {"we": net.link_index["W_in_0->E_out_0"], "ns": net.link_index["N_in_0->S_out_0"]}
    k = next(
        k
        for k in range(net.n_conflicts)
        if {int(net.conf_a[k]), int(net.conf_b[k])} == set(conns.values())
    )
    assert net.conf_kind[k] == ConflictKind.crossing
    zone = {
        name: (net.conf_zone_a if net.conf_a[k] == c else net.conf_zone_b)[k]
        for name, c in conns.items()
    }
    travel: dict[str, float] = {}
    both_inside: list[bool] = []
    for _ in range(240):
        e.step()
        inside = []
        for name, c in conns.items():
            h = veh.id_to_handle.get(name)
            if h is None or not veh.active[h]:
                continue
            lk, pos, length = int(veh.link[h]), float(veh.pos[h]), float(veh.length[h])
            if lk == net.conn_to_lane[c - net.n_lanes]:
                pos += float(net.link_length[c])  # rear may still be on the connector
            elif lk != c:
                continue
            z_in, z_out = zone[name]
            inside.append(pos - length < z_out and pos > z_in)
        both_inside.append(len(inside) == 2 and all(inside))
        ev = e.events
        for h, t in zip(
            ev.handle[ev.type == EventType.vehicle_arrived.code].tolist(),
            ev.time[ev.type == EventType.vehicle_arrived.code].tolist(),
            strict=True,
        ):
            travel[str(veh.ids[h])] = t - float(veh.insert_time[h])
    assert set(travel) == {"we", "ns"}
    return travel, e, both_inside


def _free_flow(make_engine: MakeEngine, junction_builder: Builder) -> float:
    travel, _, _ = _two_trips(make_engine, junction_builder, "uncontrolled", 0.0, 60.0)
    return travel["we"]


def test_first_come_first_served_at_an_uncontrolled_junction(
    make_engine: MakeEngine, junction_builder: Builder
) -> None:
    free = _free_flow(make_engine, junction_builder)
    for first, second, t_we, t_ns in (("we", "ns", 0.0, 1.0), ("ns", "we", 1.0, 0.0)):
        travel, e, both_inside = _two_trips(
            make_engine, junction_builder, "uncontrolled", t_we, t_ns
        )
        assert travel[first] == pytest.approx(free, abs=0.5)  # the earlier one is not delayed
        assert travel[second] > free + 2.0  # the later one yields
        assert not any(both_inside) and e.forced_commits == 0


def test_minor_yields_to_major_at_a_priority_junction(
    make_engine: MakeEngine, junction_builder: Builder
) -> None:
    free = _free_flow(make_engine, junction_builder)
    # E-W is the major axis; the minor N->S vehicle arrives first but must yield
    travel, e, both_inside = _two_trips(make_engine, junction_builder, "priority", 1.0, 0.0)
    assert travel["we"] == pytest.approx(free, abs=0.5)
    assert travel["ns"] > free + 2.0
    assert not any(both_inside) and e.forced_commits == 0
    # the same timing without priority: first come, first served
    travel, _, _ = _two_trips(make_engine, junction_builder, "uncontrolled", 1.0, 0.0)
    assert travel["ns"] == pytest.approx(free, abs=0.5) and travel["we"] > free + 2.0


def test_dont_block_the_box_under_exit_spillback(make_engine: MakeEngine) -> None:
    b = ScenarioBuilder("spillback", duration=600)
    b.vehicle_type("det", **DET)
    b.boundary("W", (-200.0, 0.0))
    b.intersection("J", (0.0, 0.0), kind="uncontrolled")
    b.boundary("E", (100.0, 0.0))  # the blocker needs ~48 m to stop at b
    b.road("W_in", "W", "J")
    b.road("E_out", "J", "E")
    b.trip("blocker", 0.0, route=["W_in", "E_out"], vehicle_type="det")
    b.flow("f", route=["W_in", "E_out"], period=2.0, begin=1.0, count=20, vehicle_type="det")
    e = make_engine(b.build(), dt=0.5)
    net, veh = e.network, e.vehicles
    out, conn = net.link_index["E_out_0"], net.link_index["W_in_0->E_out_0"]
    blocked = False
    for _ in range(400):
        e.step()
        h = veh.id_to_handle["blocker"]
        if not blocked and veh.link[h] == out:
            e.commands.set_speed("blocker", 0.0)
            blocked = True
        run = veh.running()
        stopped_inside = run[(veh.link[run] == conn) & veh.halting[run]]
        assert stopped_inside.size == 0  # never halting inside the intersection
    run = veh.running()
    assert blocked and not (veh.link[run] == conn).any()
    queue = run[veh.link[run] == out]
    assert queue.size >= 3
    rears = np.sort(veh.pos[queue] - veh.length[queue])
    assert rears[0] >= 0.0  # the whole queue fits on the exit lane
    approach = run[veh.link[run] == net.link_index["W_in_0"]]
    front = approach[np.argmax(veh.pos[approach])]
    end = float(net.link_length[net.link_index["W_in_0"]])
    assert veh.held[front] and end - 0.6 <= veh.pos[front] <= end - 0.4
    assert e.forced_commits == 0 and e.safety_cap_violations == 0


def test_saturated_uncontrolled_run_has_no_overlap(make_engine: MakeEngine) -> None:
    """1800 s over capacity with debug checks: I4 (and every other check) holds each step."""
    scenario = generate("single_intersection", kind="uncontrolled", lanes=1, demand_rate=900.0)
    e = make_engine(scenario, seed=2, duration=1800)
    for _ in range(1800):
        e.step()
    assert e.backlog > 0  # saturated: demand exceeds what the junction serves
    assert e.arrived > 500
    assert e.safety_cap_violations == 0 and e.forced_commits == 0


def test_priority_junction_keeps_the_major_flowing(make_engine: MakeEngine) -> None:
    """Opposing major left turns must not wait for each other's queued straights (deadlock)."""
    scenario = generate("single_intersection", kind="priority", demand_rate=900.0)
    e = make_engine(scenario, seed=4, duration=900)
    arrived: dict[str, int] = {}
    for _ in range(900):
        e.step()
        ev = e.events
        for h in ev.handle[ev.type == EventType.vehicle_arrived.code].tolist():
            origin = str(e.vehicles.ids[h]).split("_to_")[0]
            arrived[origin] = arrived.get(origin, 0) + 1
    assert arrived.get("E", 0) > 150 and arrived.get("W", 0) > 150
    assert e.safety_cap_violations == 0
