"""Link transitions (plan F.1 step 8): hops, arrivals, clamps, lane entry and events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import pytest

from urbanflow.core.errors import SimulationError
from urbanflow.core.events import EventType
from urbanflow.core.types import IntArray, VehicleStatus
from urbanflow.engine.advance import Advance, advance_links, crossing_time
from urbanflow.routing import RoutingContext, ShortestPathRouter, plan_connector
from urbanflow.vehicles import desired_speed

from .conftest import World

CORRIDOR_E = ("W_J1", "J1_J2", "J2_E")


def _advance(
    w: World,
    dx: list[float],
    v_start: list[float],
    *,
    dt: float = 1.0,
    time: float = 0.0,
    router: object | None = None,
) -> Advance:
    w.events.clear()
    return advance_links(
        w.net,
        w.veh,
        w.types,
        w.routes,
        router or ShortestPathRouter(),  # type: ignore[arg-type]
        w.run(),
        np.asarray(dx, dtype=float),
        np.asarray(v_start, dtype=float),
        w.events,
        step=7,
        time=time,
        dt=dt,
    )


def _events(w: World) -> list[tuple[str, str, float]]:
    ev = w.events
    return [
        (EventType.from_code(int(t)).value, w.net.link_ids[int(lk)], float(tm))
        for t, lk, tm in zip(ev.type, ev.link, ev.time, strict=True)
    ]


def test_crossing_time() -> None:
    v = np.array([10.0, 0.0, 4.0, 5.0])
    acc = np.array([0.0, 2.0, -4.0, 0.0])
    x = np.array([4.0, 0.5, 1.5, 0.0])
    assert crossing_time(v, acc, x, 1.0) == pytest.approx([0.4, 0.5**0.5, 0.5, 0.0])
    assert crossing_time(np.array([1.0]), np.array([0.0]), np.array([9.0]), 2.0)[0] == 2.0


def test_multi_hop_in_one_step(corridor_world: World) -> None:
    w = corridor_world
    net = w.net
    lane, to = w.link("W_J1_0"), w.link("J1_J2_1")
    conn = w.link("W_J1_0->J1_J2_1")
    length = net.link_length
    h = w.place("W_J1_0", float(length[lane]) - 1.0, 0.0, route=CORRIDOR_E, committed=True)
    assert w.veh.next_conn[h] == conn
    w.veh.commit_seq[h] = 4
    w.veh.link_waiting_time[h] = 3.0
    dx = 1.0 + float(length[conn]) + 3.0
    v = dx / 2.0
    w.veh.speed[h] = v
    adv = _advance(w, [dx], [v], dt=2.0, time=10.0)
    assert (int(w.veh.link[h]), float(w.veh.pos[h])) == (to, pytest.approx(3.0))
    assert w.veh.route_cursor[h] == 1
    assert not w.veh.committed[h] and w.veh.commit_seq[h] == -1
    assert w.veh.link_waiting_time[h] == 0.0
    assert w.veh.lock_conn[h] == conn  # kept across lane entry: the rear is still on it
    assert w.veh.next_conn[h] == plan_connector(net, to, w.routes.get(0), 1)
    assert w.veh.v0[h] == pytest.approx(desired_speed(net, w.veh, w.types, np.array([h]))[0])
    assert adv.crossed.tolist() == [h] and adv.crossed_committed.tolist() == [True]
    assert adv.arrived.size == 0 and adv.violations == 0 and adv.dx[0] == pytest.approx(dx)
    t1 = 10.0 + 1.0 / v
    t2 = 10.0 + (1.0 + float(length[conn])) / v
    assert _events(w) == [
        ("vehicle_exited_link", "W_J1_0", pytest.approx(t1)),
        ("vehicle_entered_link", "W_J1_0->J1_J2_1", pytest.approx(t1)),
        ("vehicle_exited_link", "W_J1_0->J1_J2_1", pytest.approx(t2)),
        ("vehicle_entered_link", "J1_J2_1", pytest.approx(t2)),
    ]
    j1 = net.int_index["J1"]
    assert w.events.intersection.tolist() == [-1, j1, j1, -1]
    assert set(w.events.step.tolist()) == {7}
    assert set(w.events.uid.tolist()) == {int(w.veh.uid[h])}


def test_connector_entry_refreshes_v0(corridor_world: World) -> None:
    w = corridor_world
    lane = float(w.net.link_length[w.link("J1_J2_2")])
    h = w.place("J1_J2_2", lane - 1.0, 5.0, route=("J1_J2", "J2_S"), committed=True)
    _advance(w, [2.0], [5.0])
    assert w.net.link_ids[int(w.veh.link[h])] == "J1_J2_2->J2_S_0"
    assert w.veh.v0[h] == pytest.approx(w.net.link_speed_limit[w.veh.link[h]])
    assert w.veh.route_cursor[h] == 0 and w.veh.committed[h]
    assert w.veh.lock_conn[h] == w.veh.link[h]  # connector entry locks it (F.3 foe scan)


@pytest.mark.parametrize(
    ("start", "v_start", "v_new", "dx", "tau"),
    [
        (4.0, 10.0, 10.0, 10.0, 0.4),  # cruising
        (0.5, 0.0, 2.0, 1.0, 0.5**0.5),  # accelerating from rest at 2 m/s^2
        (1.5, 4.0, 0.0, 2.0, 0.5),  # stopping inside the step at 4 m/s^2
    ],
)
def test_interpolated_arrival(
    junction: World, start: float, v_start: float, v_new: float, dx: float, tau: float
) -> None:
    w = junction
    lane = float(w.net.link_length[w.link("E_out_0")])
    h = w.place("E_out_0", lane - start, v_start, route=("E_out",))
    w.veh.speed[h] = v_new
    adv = _advance(w, [dx], [v_start], time=20.0)
    assert adv.arrived.tolist() == [h]
    assert w.veh.status[h] == VehicleStatus.arrived.code and not w.veh.active[h]
    assert w.veh.pos[h] == lane and adv.dx[0] == pytest.approx(start)
    assert _events(w) == [
        ("vehicle_exited_link", "E_out_0", pytest.approx(20.0 + tau)),
        ("vehicle_arrived", "E_out_0", pytest.approx(20.0 + tau)),
    ]


def test_arrival_after_hops(junction: World) -> None:
    w = junction
    conn = w.link("W_in_0->E_out_0")
    length = float(w.net.link_length[conn])
    h = w.place("W_in_0->E_out_0", length - 1.0, 20.0, route=("W_in", "E_out"))
    w.veh.speed[h] = 20.0
    lane = float(w.net.link_length[w.link("E_out_0")])
    adv = _advance(w, [1.0 + lane + 2.0], [20.0], dt=15.0)
    assert adv.arrived.tolist() == [h]
    kinds = [e[0] for e in _events(w)]
    assert kinds == [
        "vehicle_exited_link",
        "vehicle_entered_link",
        "vehicle_exited_link",
        "vehicle_arrived",
    ]
    assert adv.dx[0] == pytest.approx(1.0 + lane)


def test_uncommitted_vehicle_is_clamped_at_the_stop_line(junction: World) -> None:
    w = junction
    lane = float(w.net.link_length[w.link("W_in_0")])
    h = w.place("W_in_0", lane - 1.0, 3.0, route=("W_in", "E_out"))
    w.veh.speed[h] = 3.0
    adv = _advance(w, [3.0], [3.0])
    assert (float(w.veh.pos[h]), float(w.veh.speed[h])) == (lane, 0.0)
    assert int(w.veh.link[h]) == w.link("W_in_0")
    assert adv.violations == 1 and adv.dx[0] == pytest.approx(1.0) and len(w.events) == 0
    assert w.veh.accel[h] == pytest.approx(-3.0)


def test_no_hop_within_the_link(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", 10.0, 5.0, route=("W_in", "E_out"))
    adv = _advance(w, [5.0], [5.0])
    assert w.veh.pos[h] == 15.0 and len(w.events) == 0 and adv.crossed.size == 0


class _Detour:
    """Reroutes every vehicle entering J1_J2 to J2_N (or to an invalid route)."""

    name: ClassVar[str] = "detour"

    def __init__(self, rest: tuple[int, ...]) -> None:
        self.rest = rest
        self.seen: list[int] = []

    def on_road_entry(
        self, handles: IntArray, ctx: RoutingContext
    ) -> Mapping[int, tuple[int, ...]]:
        self.seen += handles.tolist()
        assert ctx.step == 7
        return {int(h): self.rest for h in handles}


def test_reroute_hook_on_lane_entry(corridor_world: World) -> None:
    w = corridor_world
    net = w.net
    conn = w.link("W_J1_0->J1_J2_1")
    h = w.place("W_J1_0->J1_J2_1", float(net.link_length[conn]) - 1.0, 5.0, route=CORRIDOR_E)
    road = net.road_index
    router = _Detour((road["J1_J2"], road["J2_N"]))
    _advance(w, [2.0], [5.0], router=router)
    assert router.seen == [h]
    route = w.routes.get(int(w.veh.route_id[h])).tolist()
    assert route == [road["W_J1"], road["J1_J2"], road["J2_N"]]
    assert w.veh.next_conn[h] == -1  # J1_J2 lane 1 cannot reach J2_N: lane change needed
    assert w.veh.valid_mask[h] == 1  # only lane 0 of J1_J2 leads to J2_N
    w2 = type(w)(net)
    w2.place("W_J1_0->J1_J2_1", float(net.link_length[conn]) - 1.0, 5.0, route=CORRIDOR_E)
    with pytest.raises(SimulationError, match="invalid route"):
        _advance(w2, [2.0], [5.0], router=_Detour((road["J2_E"],)))
