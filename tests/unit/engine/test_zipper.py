"""Lane-end swap deadlocks: two stopped vehicles each needing the other's lane trade places."""

from __future__ import annotations

from collections.abc import Callable

from urbanflow import generate
from urbanflow.core.events import EventBuffer
from urbanflow.engine import Engine
from urbanflow.engine.lane_changes import zipper_swaps
from urbanflow.scenario import ScenarioBuilder


def test_side_by_side_swap(make_engine: Callable[..., Engine]) -> None:
    # 2-lane approach W_in: lane 0 serves left + straight, lane 1 straight + right
    base = generate("single_intersection", demand_rate=50, duration=300)
    b = ScenarioBuilder.from_scenario(base)
    b.trip("right", 0.0, route=["W_in", "S_out"])  # needs lane 1
    b.trip("left", 0.0, route=["W_in", "N_out"])  # needs lane 0
    e = make_engine(b.build())
    while not {"right", "left"} <= {e.vehicles.ids[h] for h in e.vehicles.running().tolist()}:
        e.step()
    veh, net = e.vehicles, e.network
    road = net.road_index["W_in"]
    lane0 = int(net.road_lane_start[road])
    end = float(net.link_length[lane0]) - 3.0
    rt, lt = e.vehicles.id_to_handle["right"], e.vehicles.id_to_handle["left"]
    for h, lane in ((rt, lane0), (lt, lane0 + 1)):  # each in the wrong lane, side by side
        veh.link[h], veh.pos[h], veh.speed[h], veh.next_conn[h] = lane, end, 0.0, -1
    swaps = zipper_swaps(
        net, veh, e.types, e.routes, veh.running(), EventBuffer(), step=1, time=1.0
    )
    assert sorted(swaps.handles.tolist()) == sorted([rt, lt])
    assert (int(veh.link[rt]), int(veh.link[lt])) == (lane0 + 1, lane0)
    assert veh.next_conn[rt] >= 0 and veh.next_conn[lt] >= 0
    for _ in range(60):  # debug invariants (no overlaps) hold afterwards
        e.step()
