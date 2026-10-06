"""Insertion queues and safe insertion (plan G.8): FIFO, lanes, space, depart speed."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from urbanflow.core import constants as C
from urbanflow.core.types import VehicleStatus
from urbanflow.demand import InsertionQueues, SpawnRequest, enqueue, insert_step, lane_tails
from urbanflow.demand.insertion import lane_rears
from urbanflow.network import CompiledNetwork
from urbanflow.routing import RouteTable, plan_connector, valid_mask
from urbanflow.scenario.schema import VehicleTypeSpec
from urbanflow.vehicles import IDM, VehicleTable, VehicleTypes, safe_speed, stop_budget
from urbanflow.vehicles.table import DEPART_LANE_BEST, DEPART_LANE_FIRST

DT = 1.0
NAN = math.nan


@dataclass
class World:
    net: CompiledNetwork
    veh: VehicleTable
    types: VehicleTypes
    routes: RouteTable
    queues: InsertionQueues

    def route(self, *roads: str) -> int:
        return self.routes.intern([self.net.road_index[r] for r in roads])

    def wait(
        self,
        vid: str,
        roads: tuple[str, ...] = ("W_J1", "J1_J2", "J2_E"),
        lane: int = DEPART_LANE_BEST,
        speed: float = NAN,
        vtype: str = "car",
        factor: float = 1.0,
    ) -> int:
        req = SpawnRequest(
            vid, 0, self.types.index[vtype], self.route(*roads), factor, lane, speed, 0.0
        )
        return enqueue(self.veh, self.types, self.routes, self.queues, req)

    def place(self, vid: str, link: str, pos: float, speed: float) -> int:
        """A running vehicle (car) at ``pos`` on ``link``."""
        h = self.veh.alloc(vid)
        self.veh.status[h] = VehicleStatus.running.code
        self.veh.active[h] = True
        self.veh.type_idx[h] = self.types.index["car"]
        self.veh.length[h] = C.VEHICLE_LENGTH
        self.veh.link[h] = self.net.link_index[link]
        self.veh.pos[h] = pos
        self.veh.speed[h] = speed
        return h

    def insert(self, reserved: dict[str, float] | None = None, limit: float = math.inf):
        res = np.zeros(self.net.n_lanes)
        for lane, value in (reserved or {}).items():
            res[self.net.link_index[lane]] = value
        tails = lane_tails(self.net, self.veh, self.types)
        return insert_step(
            self.net,
            self.veh,
            self.types,
            self.routes,
            self.queues,
            tails,
            res,
            dt=DT,
            time=12.0,
            limit=limit,
        )

    def lanes(self, handles: np.ndarray) -> list[str]:
        return [self.net.link_ids[int(self.veh.link[h])] for h in handles]


@pytest.fixture
def world(corridor_net: CompiledNetwork) -> World:
    types = VehicleTypes.from_specs(corridor_net.vehicle_types, IDM.Params)
    return World(corridor_net, VehicleTable(8), types, RouteTable(), InsertionQueues(corridor_net))


def test_queues(world: World) -> None:
    q = world.queues
    q.push(3, 10)
    q.push(1, 11)
    q.push(3, 12)
    assert q.roads() == [1, 3]
    assert list(q[3]) == [10, 12]
    assert len(q) == 3
    assert list(q[4]) == []
    q[1].popleft()
    assert q.roads() == [3]
    q.clear()
    assert len(q) == 0
    # the static lane -> incoming connectors index
    net = world.net
    for lane in range(net.n_lanes):
        into = q.in_conn[q.in_ptr[lane] : q.in_ptr[lane + 1]].tolist()
        assert into == (net.n_lanes + np.flatnonzero(net.conn_to_lane == lane)).tolist()


def test_enqueue_writes_the_row(world: World) -> None:
    h = world.wait("f.0", lane=1, speed=3.0, vtype="bus", factor=0.95)
    veh = world.veh
    assert veh.status[h] == VehicleStatus.waiting_insert.code
    assert not veh.active[h]
    assert veh.type_idx[h] == world.types.index["bus"]
    assert veh.length[h] == 12.0
    assert veh.width[h] == pytest.approx(2.55)
    assert veh.speed_factor[h] == pytest.approx(0.95)
    assert (veh.depart_lane[h], veh.depart_speed[h]) == (1, 3.0)
    assert list(world.queues[world.net.road_index["W_J1"]]) == [h]


def test_inserted_vehicle_state(world: World) -> None:
    h = world.wait("f.0", factor=0.9)
    (got,) = world.insert()
    net, veh = world.net, world.veh
    lane = net.link_index["W_J1_0"]  # equal free space: lowest index
    assert got == h
    assert veh.status[h] == VehicleStatus.running.code
    assert veh.active[h]
    assert (veh.link[h], veh.pos[h]) == (lane, C.VEHICLE_LENGTH)  # rear at the lane start
    v0 = min(C.VEHICLE_MAX_SPEED, np.float32(0.9) * net.link_speed_limit[lane])
    assert veh.v0[h] == pytest.approx(v0)
    assert veh.speed[h] == pytest.approx(v0)  # empty lane: "max" gives v0
    route = world.routes.get(int(veh.route_id[h]))
    assert veh.route_cursor[h] == 0
    assert veh.valid_mask[h] == valid_mask(net, int(route[0]), int(route[1]))
    assert veh.next_conn[h] == plan_connector(net, lane, route, 0)
    assert veh.insert_time[h] == 12.0
    assert len(world.queues) == 0


def test_at_most_one_per_lane_per_step_and_best_lane(world: World) -> None:
    hs = [world.wait(f"f.{k}") for k in range(3)]
    first = world.insert()
    assert first.tolist() == hs[:2]
    assert world.lanes(first) == ["W_J1_0", "W_J1_1"]
    assert list(world.queues[world.net.road_index["W_J1"]]) == [hs[2]]
    # next step: lane 0 holds a vehicle with its rear at 0 -> "best" picks lane... both are
    # blocked by the vehicles just inserted (rear at 0 < length + s0)
    assert world.insert().tolist() == []
    world.veh.pos[hs[1]] = 40.0  # lane 1 cleared ahead
    assert world.lanes(world.insert()) == ["W_J1_1"]


def test_best_prefers_most_free_space(world: World) -> None:
    world.place("a", "W_J1_0", 200.0, 5.0)
    world.place("b", "W_J1_1", 100.0, 5.0)
    world.wait("f.0")
    assert world.lanes(world.insert()) == ["W_J1_0"]
    world.wait("f.1")
    assert world.lanes(world.insert(reserved={"W_J1_1": 0.0, "W_J1_0": 150.0})) == ["W_J1_1"]


def test_first_random_and_exact_lanes(world: World) -> None:
    for k in range(3):
        world.wait(f"first.{k}", lane=DEPART_LANE_FIRST)
    assert world.lanes(world.insert()) == ["W_J1_0"]  # one lane only: one per step
    world.queues.clear()
    world.veh.reset()
    world.wait("x.0", lane=1)
    world.wait("x.1", lane=1)
    assert world.lanes(world.insert()) == ["W_J1_1"]
    # an explicit lane that cannot reach the next road is no candidate (G.8): the head waits
    world.queues.clear()
    world.veh.reset()
    world.wait("n.0", roads=("J1_J2", "J2_N"), lane=2)  # lane 2 only reaches J2_S
    assert world.insert().tolist() == []
    world.wait("n.1", roads=("J1_J2", "J2_S"))  # FIFO: blocked behind it
    assert world.insert().tolist() == []


def test_lane_rears_include_connector_overhang(world: World) -> None:
    """A vehicle on an outgoing connector whose rear hangs back over the lane end is the
    lane's most upstream body: an otherwise empty lane is not free up to its end."""
    net = world.net
    lane = net.link_index["J1_J2_1"]
    length = float(net.link_length[lane])
    bus = world.place("bus", "J1_J2_1->J2_E_1", 2.0, 0.0)
    world.veh.length[bus] = 12.0  # rear at -10 m: 10 m back on J1_J2_1
    tails = lane_tails(net, world.veh, world.types)
    assert tails.rear[lane] == pytest.approx(length - 10.0)
    other = net.link_index["J1_J2_0"]  # a sibling lane is not affected
    assert tails.rear[other] == net.link_length[other]
    run = world.veh.running()
    body, rear = lane_rears(net, world.veh.link[run], world.veh.pos[run] - 12.0, world.veh.uid[run])
    assert run[body[lane]] == bus and rear[lane] == pytest.approx(length - 10.0)
    # a vehicle with its front on the lane further upstream is the body instead
    car = world.place("car", "J1_J2_1", 50.0, 3.0)
    tails = lane_tails(net, world.veh, world.types)
    assert tails.rear[lane] == pytest.approx(50.0 - C.VEHICLE_LENGTH)
    assert tails.speed[lane] == 3.0
    world.veh.pos[car] = length - 1.0  # now the overhang is further upstream
    assert lane_tails(net, world.veh, world.types).rear[lane] == pytest.approx(length - 10.0)


def test_fifo_head_blocks_its_road(world: World) -> None:
    world.place("stopped", "W_J1_1", C.VEHICLE_LENGTH + 3.0, 0.0)  # rear at 3 m
    world.wait("head", lane=1)
    tail = world.wait("tail")  # "best" could use lane 0, but it is behind the head
    assert world.insert().tolist() == []
    other = world.wait("other", roads=("J1_J2", "J2_E"))  # a different road is independent
    assert world.insert().tolist() == [other]
    world.veh.pos[0] = 60.0
    got = world.insert()
    assert world.veh.ids[got[0]] == "head"
    assert got[1] == tail


def test_reserved_space_blocks(world: World) -> None:
    world.wait("f.0", lane=0)
    lane_len = world.net.link_length[world.net.link_index["W_J1_0"]]
    blocked = lane_len - C.VEHICLE_LENGTH - C.IDM_MIN_GAP + 0.1
    assert world.insert(reserved={"W_J1_0": blocked}).tolist() == []
    assert len(world.insert(reserved={"W_J1_0": blocked - 0.2})) == 1


def test_limit_caps_insertions(world: World) -> None:
    world.wait("a")
    world.wait("b")
    assert len(world.insert(limit=1)) == 1
    assert len(world.queues) == 1


def test_depart_speed_behind_a_slow_leader(world: World) -> None:
    lead = world.place("lead", "W_J1_0", 25.0, 2.0)
    h = world.wait("f.0", lane=0)
    world.insert()
    v = world.veh.speed[h]
    b_hat = world.types.b_hat[world.types.index["car"]]
    budget = 25.0 - C.VEHICLE_LENGTH - C.VEHICLE_LENGTH + 4.0 / (2 * 6.0) - C.SAFETY_MARGIN
    assert 0 < v < world.veh.v0[h]
    assert v * DT + v * v / (2 * b_hat) == pytest.approx(budget)  # stationary solution
    # a numeric depart speed above that bound waits; one below it departs as given
    world.veh.pos[lead] = 25.0
    world.veh.pos[h] = 60.0
    world.veh.link[h] = world.net.link_index["J2_E_0"]
    slow = world.wait("f.1", lane=0, speed=float(v) + 0.5)
    assert world.insert().tolist() == []
    world.queues.clear()
    ok = world.wait("f.2", lane=0, speed=float(v) - 0.5)
    assert world.insert().tolist() == [ok]
    assert world.veh.speed[ok] == pytest.approx(float(v) - 0.5)
    assert world.veh.status[slow] == VehicleStatus.waiting_insert.code


def test_no_room_for_a_stationary_solution(corridor_net: CompiledNetwork) -> None:
    # a type with s0 = 0 passes the space check with a 0.3 m rear gap, but then
    # C = 0.3 - s_m < 0: no speed satisfies v dt + v^2/(2 b_hat) <= C, so it waits
    specs = (*corridor_net.vehicle_types, VehicleTypeSpec(id="tight", min_gap=0.0))
    types = VehicleTypes.from_specs(specs, IDM.Params)
    w = World(corridor_net, VehicleTable(4), types, RouteTable(), InsertionQueues(corridor_net))
    lead = w.place("lead", "W_J1_0", 2 * C.VEHICLE_LENGTH + 0.3, 0.0)
    h = w.wait("f.0", lane=0, vtype="tight")
    assert w.insert().tolist() == []
    w.veh.pos[lead] += 0.3  # C = 0.1 m: a crawl is safe
    assert w.insert().tolist() == [h]
    assert 0 < w.veh.speed[h] < 0.1


def test_connector_vehicle_must_be_able_to_follow(world: World) -> None:
    net = world.net
    conn = "W_J1_0->J1_J2_0"
    length = net.link_length[net.link_index[conn]]
    fast = world.place("fast", conn, length - 0.5, 13.0)  # about to enter J1_J2_0
    at_speed = world.wait("n.0", roads=("J1_J2", "J2_N"))  # only lane 0 is valid
    assert world.insert().tolist() == [at_speed]  # 13 m/s can follow a v0 departure
    world.veh.free_deferred(at_speed)
    # a standing start right in front of it would need more than b_emerg: wait
    h = world.wait("n.1", roads=("J1_J2", "J2_N"), speed=0.0)
    assert world.insert().tolist() == []
    world.veh.pos[fast], world.veh.speed[fast] = 1.0, 1.0  # far upstream and slow
    assert world.insert().tolist() == [h]
    # the follower's cap behind the new vehicle is reachable within its b_emerg
    j = fast
    gap = length - world.veh.pos[j]
    budget = stop_budget(np.array([gap]), np.array([world.veh.speed[h]]), np.array([6.0]))
    r = safe_speed(np.array([1.0]), budget, np.array([6.0]), DT)
    assert r[0] >= 1.0 - 6.0 * DT


@given(
    lead_pos=st.floats(2 * C.VEHICLE_LENGTH + C.IDM_MIN_GAP, 280.0),
    lead_speed=st.floats(0, 15),
)
@settings(max_examples=100, deadline=None)
def test_depart_speed_is_safe(
    corridor_net: CompiledNetwork, lead_pos: float, lead_speed: float
) -> None:
    """G.8: the inserted speed satisfies the G.2 invariant v^2/(2 b_hat) <= C behind the
    lane's last vehicle, so the next step's cap is feasible (depart speed <= safe speed)."""
    types = VehicleTypes.from_specs(corridor_net.vehicle_types, IDM.Params)
    w = World(corridor_net, VehicleTable(4), types, RouteTable(), InsertionQueues(corridor_net))
    w.place("lead", "W_J1_0", lead_pos, lead_speed)
    h = w.wait("f.0", lane=0)
    w.insert()
    assert w.veh.active[h]
    v = float(w.veh.speed[h])
    b_hat = float(types.b_hat[types.index["car"]])
    gap = lead_pos - C.VEHICLE_LENGTH - float(w.veh.pos[h])
    budget = gap + lead_speed**2 / (2 * 6.0) - C.SAFETY_MARGIN
    assert v <= w.veh.v0[h] + 1e-12
    assert v * DT + v * v / (2 * b_hat) <= budget + 1e-9
    assert v * v / (2 * b_hat) <= budget + 1e-9
    r = safe_speed(np.array([v]), np.array([budget]), np.array([b_hat]), DT)
    assert r[0] >= v - 1e-9  # it may keep its depart speed through the first step


def test_lane_tails(world: World) -> None:
    net, veh = world.net, world.veh
    empty = lane_tails(net, veh, world.types)
    assert np.array_equal(empty.rear, net.link_length[: net.n_lanes])
    assert not empty.speed.any()
    assert (empty.conn_last == -1).all()
    world.place("a", "W_J1_0", 80.0, 3.0)
    world.place("b", "W_J1_0", 30.0, 2.0)  # the tail of W_J1_0
    c1 = world.place("c1", "W_J1_0->J1_J2_0", 1.0, 4.0)
    c2 = world.place("c2", "W_J1_0->J1_J2_0", 6.0, 5.0)  # nearest the connector end
    tails = lane_tails(net, veh, world.types)
    lane = net.link_index["W_J1_0"]
    assert tails.rear[lane] == pytest.approx(30.0 - C.VEHICLE_LENGTH)
    assert tails.speed[lane] == 2.0
    assert tails.b_emerg[lane] == C.IDM_EMERGENCY_DECEL
    assert math.isinf(tails.b_emerg[net.link_index["W_J1_1"]])
    conn = net.link_index["W_J1_0->J1_J2_0"] - net.n_lanes
    assert tails.conn_last[conn] == c2
    assert c1 != c2


def test_queues_state_dict_round_trip(world: World) -> None:
    import json

    q = world.queues
    q.push(3, 10)
    q.push(1, 11)
    q.push(3, 12)
    q.push(2, 13)
    q[2].popleft()  # an empty queue is not part of the state
    state = json.loads(json.dumps(q.state_dict()))
    assert state == {"queues": [[1, [11]], [3, [10, 12]]]}
    other = InsertionQueues(world.net)
    other.push(5, 99)
    other.load_state_dict(state)
    assert other.roads() == [1, 3] and list(other[3]) == [10, 12] and len(other) == 3
