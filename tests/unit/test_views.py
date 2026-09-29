"""Views, vector state and vehicle control (plan AA 5.4, 5.5, AD.1 8.10)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from urbanflow import Scenario, ScenarioBuilder, Simulation, generate
from urbanflow.core import constants as C
from urbanflow.core.errors import CommandError, NotFoundError
from urbanflow.core.types import LinkKind, VehicleClass, VehicleStatus
from urbanflow.views import stopline_queue, vehicle_xy

DET = {"speed_factor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}}


@pytest.fixture(scope="module")
def junction() -> Scenario:
    return generate("single_intersection", kind="uncontrolled", duration=600)


@pytest.fixture
def busy(junction: Scenario) -> Simulation:
    sim = Simulation(junction, seed=1)
    sim.run(until=200)
    return sim


def _straight(**simulation: Any) -> ScenarioBuilder:
    b = ScenarioBuilder("straight", **simulation)
    b.vehicle_type("det", **DET)
    b.boundary("A", (100.0, 50.0))
    b.boundary("B", (600.0, 50.0))
    b.road("AB", "A", "B")
    return b


# ------------------------------------------------------------------------------- state arrays
def test_state_arrays_are_aligned_read_only_copies(busy: Simulation) -> None:
    state, veh = busy.state, busy.vehicles
    n = len(veh)
    assert n > 0
    for name in ("uids", "type_idx", "links", "positions", "speeds", "accels", "headings"):
        arr = getattr(state, name)
        assert arr.shape == (n,) and not arr.flags.writeable, name
    assert state.xy.shape == (n, 2) and not state.xy.flags.writeable
    assert len(state.ids) == n
    for i, vid in enumerate(state.ids):
        view = veh[vid]
        assert view.uid == state.uids[i]
        assert (view.link_index, view.position, view.speed) == (
            state.links[i],
            state.positions[i],
            state.speeds[i],
        )
        assert (view.x, view.y) == tuple(state.xy[i])
        assert view.waiting_time == state.waiting_times[i]
        assert view.distance == state.distances[i]


def test_state_cache_lives_for_one_step(busy: Simulation) -> None:
    speeds = busy.state.speeds
    assert busy.state.speeds is speeds  # cached within the step
    counts = busy.lanes.vehicle_counts()
    assert busy.lanes.vehicle_counts() is counts
    busy.step()
    assert busy.state.speeds is not speeds
    assert busy.lanes.vehicle_counts() is not counts
    before = busy.state.uids
    busy.reset()
    assert busy.state.uids is not before and busy.state.uids.size == 0


def test_raw_is_live_and_read_only(busy: Simulation) -> None:
    raw = busy.state.raw
    active = raw["active"]
    assert not active.flags.writeable and not raw["speed"].flags.writeable
    assert set(raw) >= {"uid", "link", "pos", "speed", "status", "active", "v0"}
    assert np.array_equal(np.sort(raw["uid"][np.flatnonzero(active)]), np.sort(busy.state.uids))
    with pytest.raises(ValueError, match="read-only"):
        raw["speed"][0] = 1.0


def test_vehicle_xy_follows_the_lane_and_the_median() -> None:
    b = _straight(duration=60)
    b.trip("v", 0.0, route=["AB"], vehicle_type="det")
    sim = Simulation(b.build())
    net = sim.network.compiled
    lane = net.lane_of("AB", 0)
    xy, heading = vehicle_xy(
        net, np.array([lane, lane]), np.array([10.0, 10.0]), np.array([0.0, 1.0])
    )
    start = net.link_points(lane)[0] + net.origin
    assert heading.tolist() == pytest.approx([0.0, 0.0])  # eastbound
    assert xy[0].tolist() == pytest.approx([start[0] + 10.0, start[1]])
    assert xy[1].tolist() == pytest.approx([start[0] + 10.0, start[1] + 1.0])  # left = median
    sim.step(2)
    view = sim.vehicles["v"]
    assert view.heading == pytest.approx(0.0)
    assert (view.x, view.y) == pytest.approx((start[0] + view.position, start[1]))


# ------------------------------------------------------------------------------- vehicles
def test_vehicle_view_fields() -> None:
    b = _straight(duration=60)
    b.trip("v", 0.0, route=["AB"], vehicle_type="det")
    b.trip("w", 30.0, route=["AB"], vehicle_type="bus")
    sim = Simulation(b.build(), dt=0.5)
    sim.step(4)
    v = sim.vehicles["v"]
    assert (v.id, v.uid, v.type, v.vclass, v.status) == (
        "v",
        0,
        "det",
        VehicleClass.car,
        VehicleStatus.running,
    )
    assert (v.length, v.max_accel, v.decel, v.headway, v.min_gap) == (
        C.VEHICLE_LENGTH,
        C.IDM_ACCEL,
        C.IDM_DECEL,
        C.IDM_HEADWAY,
        C.IDM_MIN_GAP,
    )
    assert v.reaction_time == 0.5 and v.model_params == {"delta": C.IDM_DELTA}
    assert (v.road, v.lane, v.lane_index, v.connector, v.intersection) == (
        "AB",
        "AB_0",
        0,
        None,
        None,
    )
    assert v.route == ("AB",) and v.route_index == 0 and v.next_road is None
    assert v.destination == "AB" and v.depart_time == 0.0 and v.insert_time == 0.0
    assert v.travel_time == 2.0 and v.desired_speed == pytest.approx(C.SPEED_LIMIT)
    assert v.speed_override is None and v.stops == 0
    assert v.distance == pytest.approx(v.position - C.VEHICLE_LENGTH)
    sim.vehicles.set_speed("v", 0.0)
    assert sim.vehicles["v"].speed_override == 0.0
    assert sim.vehicles["v"].desired_speed == 0.0


def test_waiting_vehicles_have_no_location(junction: Scenario) -> None:
    sim = Simulation(junction)
    vid = sim.vehicles.add(route=["E_in", "W_out"])
    view = sim.vehicles[vid]
    assert view.status is VehicleStatus.waiting_insert
    assert (view.road, view.position, view.x, view.link_index, view.travel_time) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert vid in sim.vehicles and len(sim.vehicles) == 0
    assert sim.vehicles.ids("waiting_insert") == [vid]
    assert sim.vehicles.count(VehicleStatus.waiting_insert) == 1


def test_lookups_and_hints(busy: Simulation) -> None:
    veh = busy.vehicles
    first = next(iter(veh))
    assert veh.get(first.id) == first and veh.get("nope") is None
    with pytest.raises(NotFoundError, match=r'unknown vehicle "E_to_W\.99999"'):
        veh["E_to_W.99999"]
    arrived = veh.ids(VehicleStatus.arrived)
    assert arrived and len(arrived) == veh.count("arrived")
    with pytest.raises(NotFoundError, match="no longer in the simulation"):
        veh[arrived[0]]
    assert arrived[0] not in veh
    assert veh.id_of(veh.uid(arrived[0])) == arrived[0]
    assert veh.uid(first.id) == first.uid and veh.id_of(first.uid) == first.id
    with pytest.raises(NotFoundError, match="uid"):
        veh.id_of(10**6)
    with pytest.raises(NotFoundError, match="did you mean"):
        veh.ids("runing")
    assert veh.ids("pending") == [] and veh.count("pending") == 0


def test_iteration_and_ids_are_in_uid_order(busy: Simulation) -> None:
    uids = [v.uid for v in busy.vehicles]
    assert uids == sorted(uids) and len(uids) == len(busy.vehicles)
    assert busy.vehicles.ids() == [v.id for v in busy.vehicles]


def test_to_columns(busy: Simulation) -> None:
    cols = busy.vehicles.to_columns()
    n = len(busy.vehicles)
    assert all(len(col) == n for col in cols.values())
    assert cols["id"] == busy.vehicles.ids()
    assert np.all(np.diff(np.asarray(cols["uid"], dtype=np.int64)) > 0)


def test_control_add_remove_and_statuses(junction: Scenario) -> None:
    sim = Simulation(junction, seed=3)
    sim.step(30)
    vid = sim.vehicles.add(route=["E_in", "W_out"], id="probe")
    assert vid == "probe"
    sim.step()
    assert sim.vehicles["probe"].status is VehicleStatus.running
    victim = sim.vehicles.ids()[0]
    running = len(sim.vehicles)
    sim.vehicles.remove(victim)
    assert victim not in sim.vehicles and len(sim.vehicles) == running - 1
    assert sim.vehicles.ids("removed") == [victim] and sim.vehicles.count("removed") == 1
    with pytest.raises(CommandError):
        sim.vehicles.set_speed("probe", -1.0)
    sim.step()
    assert sim.metrics.summary()["vehicles.removed"] == 1
    total = sum(sim.vehicles.count(s) for s in VehicleStatus)
    assert total == sim.metrics.summary()["vehicles.generated"]


def test_set_speed_takes_effect_next_step() -> None:
    b = _straight(duration=120)
    b.trip("v", 0.0, route=["AB"], vehicle_type="det")
    sim = Simulation(b.build(), dt=0.5)
    sim.step()
    sim.vehicles.set_speed("v", 5.0, duration=30.0)
    for _ in range(60):
        sim.step()
    assert sim.vehicles["v"].speed == pytest.approx(5.0, abs=0.05)
    sim.run(until=40)
    assert sim.vehicles["v"].speed_override is None  # expired after 30 s


# ------------------------------------------------------------------------------- lanes, roads
def test_lane_count_conservation(junction: Scenario) -> None:
    sim = Simulation(junction, seed=7)
    for _ in range(300):
        sim.step()
        on_connectors = int(
            np.count_nonzero(sim.network.link_kind[sim.state.links] == LinkKind.connector.code)
        )
        lanes = int(sim.lanes.vehicle_counts().sum())
        assert lanes + on_connectors == len(sim.vehicles)
        assert int(sim.roads.vehicle_counts().sum()) == lanes


def test_lane_views_and_vectors(busy: Simulation) -> None:
    lanes = busy.lanes
    assert len(lanes) == busy.network.n_lanes == len(lanes.ids)
    counts, halting = lanes.vehicle_counts(), lanes.halting_counts()
    speeds, occupancy, queues = lanes.mean_speeds(), lanes.occupancy(), lanes.queue_lengths()
    for arr in (counts, halting, speeds, occupancy, queues):
        assert arr.shape == (len(lanes),) and not arr.flags.writeable
    for i, lane in enumerate(lanes):
        assert lane.id == lanes.ids[i] and lanes.index(lane.id) == i
        assert lane.vehicle_count == counts[i] == len(lane.vehicle_ids)
        assert lane.halting_count == halting[i] and lane.queue_length == queues[i]
        assert queues[i] <= halting[i]
        if lane.vehicle_count == 0:
            assert lane.mean_speed is None and math.isnan(speeds[i])
            continue
        views = [busy.vehicles[v] for v in lane.vehicle_ids]
        positions = [v.position for v in views]
        assert positions == sorted(positions, reverse=True)  # front to back
        assert lane.mean_speed == pytest.approx(np.mean([v.speed for v in views]))
        expected = min(1.0, sum(v.length for v in views) / lane.length)
        assert lane.occupancy == pytest.approx(expected)
    assert lanes[0] == lanes[lanes.ids[0]]
    with pytest.raises(NotFoundError, match=r'unknown lane "E_im_0" \(did you mean "E_in_0"'):
        lanes["E_im_0"]
    with pytest.raises(NotFoundError, match="out of range"):
        lanes[len(lanes)]


def test_road_views_and_vectors(busy: Simulation) -> None:
    roads, lanes = busy.roads, busy.lanes
    for road in roads:
        members = [lanes[lane] for lane in road.lanes]
        assert road.n_lanes == len(members)
        assert road.vehicle_count == sum(m.vehicle_count for m in members)
        assert road.halting_count == sum(m.halting_count for m in members)
        i = roads.index(road.id)
        assert roads.queue_lengths()[i] == sum(m.queue_length for m in members)
        if road.vehicle_count == 0:
            assert road.mean_speed is None
            assert road.travel_time == pytest.approx(road.length / road.speed_limit)
        else:
            assert road.mean_speed is not None
            assert road.travel_time == pytest.approx(
                road.length / max(road.mean_speed, C.MIN_EFFECTIVE_SPEED)
            )
        jam = sum(max(1.0, m.length / C.VEH_SPACING_REF_M) for m in members)
        assert road.utilization == pytest.approx(min(1.0, road.vehicle_count / jam))
    e_in = roads["E_in"]
    assert (e_in.from_intersection, e_in.to_intersection) == ("E", "J")
    with pytest.raises(NotFoundError, match='did you mean "E_in"'):
        roads["E_inn"]


def test_queue_is_the_halting_run_from_the_stop_line() -> None:
    lane_length = np.array([100.0, 100.0, 100.0])
    link = np.array([0, 0, 0, 0, 1, 1, 2])
    pos = np.array([99.0, 92.0, 85.0, 60.0, 80.0, 70.0, 99.5])
    halting = np.array([True, True, False, True, True, True, False])
    # lane 0: two halting at the front, then a moving car; lane 1: front 20 m back (> 10 m)
    assert stopline_queue(link, pos, halting, lane_length).tolist() == [2, 0, 0]
    assert stopline_queue(link[:0], pos[:0], halting[:0], lane_length).tolist() == [0, 0, 0]


def _naive_queue(link: list[int], pos: list[float], halting: list[bool], n: int) -> list[int]:
    out = [0] * n
    for lane in range(n):
        cars = sorted(
            ((p, h) for k, p, h in zip(link, pos, halting, strict=True) if k == lane),
            reverse=True,
        )
        if not cars or 100.0 - cars[0][0] > C.QUEUE_FRONT_TOLERANCE_M:
            continue
        for _, h in cars:
            if not h:
                break
            out[lane] += 1
    return out


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(0, 3), st.floats(0, 100, allow_nan=False), st.booleans()),
        max_size=30,
        unique_by=lambda t: (t[0], t[1]),
    )
)
def test_queue_matches_a_naive_loop(cars: list[tuple[int, float, bool]]) -> None:
    link = [c[0] for c in cars]
    pos = [c[1] for c in cars]
    halting = [c[2] for c in cars]
    got = stopline_queue(
        np.array(link, dtype=np.intp),
        np.array(pos, dtype=np.float64),
        np.array(halting, dtype=bool),
        np.full(4, 100.0),
    )
    assert got.tolist() == _naive_queue(link, pos, halting, 4)


# ------------------------------------------------------------------------------- network
def test_network_info(busy: Simulation) -> None:
    info = busy.network
    net = info.compiled
    assert info.lane_ids == net.link_ids[: net.n_lanes]
    assert info.connection_ids == net.link_ids[net.n_lanes :]
    assert info.road_ids == net.road_ids and info.intersection_ids == net.int_ids
    assert info.movement_ids == net.mov_ids
    assert info.lane_length.shape == (net.n_lanes,) and not info.lane_length.flags.writeable
    assert info.index("lane", "E_in_1") == net.link_index["E_in_1"]
    conn = info.connection_ids[3]
    assert info.index("connection", conn) == 3
    assert info.index("link", conn) == net.n_lanes + 3
    assert info.index("road", "W_out") == net.road_index["W_out"]
    assert info.index("intersection", "J") == net.int_index["J"]
    with pytest.raises(NotFoundError, match="unknown lane"):
        info.index("lane", conn)  # a connector is not a lane
    with pytest.raises(NotFoundError, match="unknown connection"):
        info.index("connection", "E_in_1")
    with pytest.raises(NotFoundError, match='did you mean "road"'):
        info.index("raod", "x")
    geometry = info.geometry()
    assert {"links", "origin", "bbox"} <= set(geometry)
    graph = info.graph()
    graph.remove_node(0)
    assert info.graph().number_of_nodes() == net.n_roads  # a copy each time
