"""Vehicle commands (plan F.4): validation, effect in the next step, events and the log."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from urbanflow.core.errors import CommandError, NotFoundError
from urbanflow.core.events import NO_VEHICLE, EventType
from urbanflow.core.types import VehicleStatus
from urbanflow.engine import Engine
from urbanflow.scenario import ScenarioBuilder

WE = ["W_in", "E_out"]
MakeEngine = Callable[..., Engine]


@pytest.fixture
def engine(make_engine: MakeEngine, junction_builder: Callable[..., ScenarioBuilder]) -> Engine:
    b = junction_builder()
    b.flow("f", route=["N_in", "S_out"], period=1000.0, begin=500.0)
    b.trip("t0", 900.0, route=["S_in", "N_out"])
    return make_engine(b.build())


def _events(e: Engine) -> list[tuple[str, int]]:
    ev = e.events
    return [
        (EventType.from_code(int(t)).value, int(u)) for t, u in zip(ev.type, ev.uid, strict=True)
    ]


def test_add_vehicle_is_inserted_by_the_next_step(engine: Engine) -> None:
    e = engine
    vid = e.commands.add_vehicle(route=WE)
    assert vid == "api.0"
    veh = e.vehicles
    h = veh.handle_of(vid)
    assert veh.status[h] == VehicleStatus.waiting_insert.code
    assert (e.generated, e.backlog) == (1, 1)
    assert veh.source_idx[h] == -1 and veh.depart_time[h] == 0.0
    e.step()
    uid = int(veh.uid[h])
    assert _events(e) == [
        ("vehicle_departed", uid),
        ("vehicle_inserted", uid),
        ("vehicle_entered_link", uid),
    ]
    assert veh.status[h] == VehicleStatus.running.code and e.backlog == 0
    assert veh.link[h] == e.network.lane_of("W_in", 0)
    assert e.commands.add_vehicle(route=WE) == "api.1"
    step, name, args = e.commands.command_log[0]
    assert (step, name, args["id"], args["route"]) == (0, "add_vehicle", "api.0", WE)
    assert e.commands.command_log[1][0] == 1


def test_add_vehicle_by_origin_and_destination(engine: Engine) -> None:
    e = engine
    vid = e.commands.add_vehicle(origin="W_in", destination="N_out", vehicle_type="bus", id="b1")
    h = e.vehicles.handle_of(vid)
    route = e.routes.get(int(e.vehicles.route_id[h]))
    assert [e.network.road_ids[r] for r in route] == ["W_in", "N_out"]
    assert e.types.ids[int(e.vehicles.type_idx[h])] == "bus"


def test_speed_factor_and_random_lane_come_from_the_vehicle_params_stream(
    make_engine: MakeEngine, junction_builder: Callable[..., ScenarioBuilder]
) -> None:
    factors = []
    for _ in range(2):
        e = make_engine(junction_builder(lanes=2).build())
        e.commands.add_vehicle(route=WE, depart_lane="random")
        e.commands.add_vehicle(route=WE, depart_lane="random")
        veh = e.vehicles
        factors.append([(float(veh.speed_factor[h]), int(veh.depart_lane[h])) for h in (0, 1)])
    assert factors[0] == factors[1]
    assert factors[0][0][0] != factors[0][1][0]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"route": ["W_inn", "E_out"]}, NotFoundError, 'did you mean "W_in"'),
        ({"route": ["W_in", "W_out"]}, CommandError, "not connected"),
        ({"route": WE, "origin": "W_in", "destination": "E_out"}, CommandError, "add_vehicle"),
        ({}, CommandError, "add_vehicle"),
        ({"route": WE, "vehicle_type": "cra"}, NotFoundError, 'did you mean "car"'),
        ({"route": WE, "depart_lane": 3}, CommandError, "has 1 lane"),
        ({"route": WE, "depart_lane": "middle"}, CommandError, "depart_lane"),
        ({"route": WE, "depart_speed": -1.0}, CommandError, "depart_speed"),
        ({"route": WE, "id": "has space"}, CommandError, "add_vehicle"),
        ({"route": WE, "id": "f.3"}, CommandError, "reserved"),
        ({"route": WE, "id": "t0"}, CommandError, "reserved"),
        ({"origin": "W_out", "destination": "E_out"}, CommandError, "not reachable"),
    ],
)
def test_add_vehicle_validation(
    engine: Engine, kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        engine.commands.add_vehicle(**kwargs)
    assert engine.generated == 0 and engine.backlog == 0 and not engine.commands.command_log


def test_add_vehicle_rejects_a_depart_lane_without_a_connection(make_engine: MakeEngine) -> None:
    """Review repro: such a vehicle could never leave its first lane (no lane changes)."""
    b = ScenarioBuilder("dead_end", duration=600)
    b.boundary("W", (-200.0, 0.0))
    b.intersection("J", (0.0, 0.0), kind="uncontrolled")
    b.boundary("E", (100.0, 0.0))
    b.road("W_in", "W", "J", lanes=2)
    b.road("E_out", "J", "E")
    b.movement("J", "W_in", "E_out", connections=[(1, 0)])
    e = make_engine(b.build())
    match = 'depart_lane 0 of road "W_in" has no connection to the next road "E_out"'
    with pytest.raises(CommandError, match=match):
        e.commands.add_vehicle(route=WE, depart_lane=0)
    with pytest.raises(CommandError, match=match):  # origin/destination: the router's leg
        e.commands.add_vehicle(origin="W_in", destination="E_out", depart_lane=0)
    assert e.generated == 0 and not e.commands.command_log
    e.commands.add_vehicle(route=WE, depart_lane=1, id="ok")
    e.commands.add_vehicle(route=["W_in"], depart_lane=0, id="last")  # last road: any lane
    assert e.generated == 2


def test_add_vehicle_ids_are_unique_in_a_run(engine: Engine) -> None:
    e = engine
    e.commands.add_vehicle(route=WE, id="x")
    with pytest.raises(CommandError, match="already used"):
        e.commands.add_vehicle(route=WE, id="x")
    assert e.commands.add_vehicle(route=WE, id="f.x") == "f.x"  # not a flow serial
    e.commands.add_vehicle(route=WE, id="api.0")
    assert e.commands.add_vehicle(route=WE) == "api.1"  # the default skips taken ids


def test_remove_a_running_vehicle(engine: Engine) -> None:
    e = engine
    vid = e.commands.add_vehicle(route=WE)
    e.step()
    veh = e.vehicles
    h = veh.handle_of(vid)
    uid, link = int(veh.uid[h]), int(veh.link[h])
    e.commands.remove_vehicle(vid)
    assert veh.status[h] == VehicleStatus.removed.code and not veh.active[h]
    assert e.removed == 1 and vid not in veh.id_to_handle
    with pytest.raises(NotFoundError, match="no longer in the simulation"):
        e.commands.remove_vehicle(vid)
    e.step()  # debug checks: conservation holds with the removal
    assert _events(e) == [("vehicle_removed", uid)]
    assert int(e.events.link[0]) == link and e.events.step[0] == 2
    assert e.commands.command_log[-1][1:] == ("remove_vehicle", {"vehicle_id": vid})


def test_removed_handle_is_not_reused_in_the_step_reporting_it(
    make_engine: MakeEngine, junction_builder: Callable[..., ScenarioBuilder]
) -> None:
    """Review repro: a spawn in the step that reports a removal must not reuse its handle,
    so no handle names two uids in one step's events."""
    b = junction_builder()
    b.flow("f", route=WE, rate=3600.0)
    e = make_engine(b.build())
    for _ in range(5):
        e.step()
    veh = e.vehicles
    running = veh.running()
    e.commands.remove_vehicle(veh.ids[int(running[0])] or "")
    for _ in range(3):
        e.step()
        seen: dict[int, set[int]] = {}
        for h, u in zip(e.events.handle.tolist(), e.events.uid.tolist(), strict=True):
            seen.setdefault(h, set()).add(u)
        assert all(len(uids) == 1 for h, uids in seen.items() if h != NO_VEHICLE), seen


def test_remove_a_waiting_vehicle(engine: Engine) -> None:
    e = engine
    vid = e.commands.add_vehicle(route=WE)
    uid = int(e.vehicles.uid[e.vehicles.handle_of(vid)])
    e.commands.remove_vehicle(vid)
    assert e.backlog == 0 and (e.generated, e.removed) == (1, 1)
    e.step()
    assert _events(e) == [("vehicle_departed", uid), ("vehicle_removed", uid)]
    assert int(e.events.link[1]) == -1 and len(e.vehicles.running()) == 0


def test_unknown_vehicle(engine: Engine) -> None:
    engine.commands.add_vehicle(route=WE, id="car1")
    with pytest.raises(NotFoundError, match='did you mean "car1"'):
        engine.commands.remove_vehicle("car2")
    with pytest.raises(NotFoundError, match="car2"):
        engine.commands.set_speed("car2", 3.0)


def test_set_speed_overrides_the_desired_speed(engine: Engine) -> None:
    e = engine
    vid = e.commands.add_vehicle(route=WE)
    e.step()
    h = e.vehicles.handle_of(vid)
    e.commands.set_speed(vid, 3.0, duration=20.0)
    assert e.vehicles.override_until[h] == 21.0  # time 1 + 20 s
    for _ in range(19):
        e.step()
    assert e.vehicles.speed[h] == pytest.approx(3.0, abs=0.05)
    e.step()
    e.step()  # the override expired at t = 21
    assert math.isnan(e.vehicles.speed_override[h]) and e.vehicles.speed[h] > 3.0
    e.commands.set_speed(vid, 2.0)
    assert math.isinf(e.vehicles.override_until[h])
    e.commands.set_speed(vid, None)
    assert math.isnan(e.vehicles.speed_override[h])
    names = [name for _, name, _ in e.commands.command_log]
    assert names == ["add_vehicle", "set_speed", "set_speed", "set_speed"]


@pytest.mark.parametrize(
    ("speed", "duration"),
    [(-1.0, None), (math.nan, None), (math.inf, None), ("fast", None), (True, None), (3.0, 0.0)],
)
def test_set_speed_validation(engine: Engine, speed: Any, duration: Any) -> None:
    vid = engine.commands.add_vehicle(route=WE)
    with pytest.raises(CommandError):
        engine.commands.set_speed(vid, speed, duration)


def test_reset_clears_the_log(engine: Engine) -> None:
    engine.commands.add_vehicle(route=WE)
    engine.reset()
    assert not engine.commands.command_log and engine.generated == 0
    assert engine.commands.add_vehicle(route=WE) == "api.0"


# ------------------------------------------------------------------------------- set_route
def _roads(e: Engine, h: int) -> list[str]:
    return [e.network.road_ids[r] for r in e.routes.get(int(e.vehicles.route_id[h]))]


def _step_until(e: Engine, done: Callable[[], bool], limit: int = 200) -> None:
    for _ in range(limit):
        if done():
            return
        e.step()
    raise AssertionError("condition not reached")


def test_set_route_replans_a_running_vehicle(engine: Engine) -> None:
    e = engine
    net, veh = e.network, e.vehicles
    vid = e.commands.add_vehicle(route=WE)
    e.step()
    h = veh.handle_of(vid)
    assert net.link_ids[int(veh.next_conn[h])] == "W_in_0->E_out_0"
    e.commands.set_route(vid, ["W_in", "N_out"])
    assert _roads(e, h) == ["W_in", "N_out"] and int(veh.route_cursor[h]) == 0
    assert net.link_ids[int(veh.next_conn[h])] == "W_in_0->N_out_0"
    assert veh.valid_mask[h] == 1
    assert e.commands.command_log[-1] == (
        1,
        "set_route",
        {"vehicle_id": vid, "roads": ["W_in", "N_out"]},
    )
    arrived = []
    while not arrived:
        e.step()
        ev = e.events
        sel = ev.type == EventType.vehicle_arrived.code
        arrived = [net.link_ids[int(k)] for k in ev.link[sel]]
    assert arrived == ["N_out_0"]


def test_set_route_can_end_on_the_current_road_and_extend_it(engine: Engine) -> None:
    e = engine
    veh = e.vehicles
    vid = e.commands.add_vehicle(route=WE)
    e.step()
    h = veh.handle_of(vid)
    e.commands.set_route(vid, ["W_in"])  # arrive at the end of this lane
    assert int(veh.next_conn[h]) == -1
    e.commands.set_route(vid, ("W_in", "S_out"))
    assert _roads(e, h) == ["W_in", "S_out"] and int(veh.next_conn[h]) >= 0


@pytest.mark.parametrize(
    ("roads", "error", "match"),
    [
        (["N_in", "S_out"], CommandError, 'must start at its current road "W_in", not "N_in"'),
        (["W_in", "E_out", "N_out"], CommandError, 'no movement from road "E_out" to road "N_out"'),
        (["W_in", "W_out"], CommandError, "route is not connected"),
        (["W_in", "N_outt"], NotFoundError, r'unknown road "N_outt" \(did you mean "N_out"'),
        ([], CommandError, "non-empty list of road ids"),
        ("W_in", CommandError, "non-empty list of road ids"),
    ],
)
def test_set_route_validation(
    engine: Engine, roads: Any, error: type[Exception], match: str
) -> None:
    e = engine
    vid = e.commands.add_vehicle(route=WE)
    e.step()
    h = e.vehicles.handle_of(vid)
    with pytest.raises(error, match=match):
        e.commands.set_route(vid, roads)
    assert _roads(e, h) == WE and len(e.commands.command_log) == 1


def test_set_route_of_unknown_or_departed_vehicles(engine: Engine) -> None:
    e = engine
    with pytest.raises(NotFoundError, match='unknown vehicle "ghost"'):
        e.commands.set_route("ghost", WE)
    vid = e.commands.add_vehicle(route=WE)
    e.commands.remove_vehicle(vid)
    with pytest.raises(NotFoundError, match="no longer in the simulation"):
        e.commands.set_route(vid, WE)


def test_a_committed_vehicle_keeps_its_connector(engine: Engine) -> None:
    e = engine
    net, veh = e.network, e.vehicles
    vid = e.commands.add_vehicle(route=WE)
    e.step()
    h = veh.handle_of(vid)
    _step_until(e, lambda: bool(veh.committed[h]))
    assert int(veh.link[h]) < net.n_lanes  # still on its approach lane
    conn = int(veh.next_conn[h])
    with pytest.raises(CommandError) as info:
        e.commands.set_route(vid, ["W_in", "N_out"])
    assert str(info.value) == (
        f"vehicle {vid} is committed to connector W_in_0->E_out_0; "
        "the new route must continue through it"
    )
    with pytest.raises(CommandError, match="committed to connector"):
        e.commands.set_route(vid, ["W_in"])  # stopping short is not possible either
    e.commands.set_route(vid, WE)  # through the connector's road: fine
    assert int(veh.next_conn[h]) == conn and bool(veh.committed[h])


def test_set_route_on_a_connector_starts_at_the_next_road(engine: Engine) -> None:
    e = engine
    net, veh = e.network, e.vehicles
    vid = e.commands.add_vehicle(route=WE)
    e.step()
    h = veh.handle_of(vid)
    _step_until(e, lambda: int(veh.link[h]) >= net.n_lanes)
    with pytest.raises(CommandError, match='must start at the road its connector leads to "E_out"'):
        e.commands.set_route(vid, WE)
    e.commands.set_route(vid, ["E_out"])
    assert _roads(e, h) == WE and int(veh.route_cursor[h]) == 0
    _step_until(e, lambda: vid not in veh.id_to_handle)  # arrives normally


def test_set_route_of_a_waiting_vehicle(
    make_engine: MakeEngine, junction_builder: Callable[..., ScenarioBuilder]
) -> None:
    e = make_engine(junction_builder(lanes=2).build())
    veh = e.vehicles
    # W_in: lane 0 (median) turns left to N_out, lane 1 (curb) turns right to S_out
    vid = e.commands.add_vehicle(route=["W_in", "S_out"], depart_lane=1)
    h = veh.handle_of(vid)
    with pytest.raises(CommandError, match='must start at its first road "W_in"'):
        e.commands.set_route(vid, ["N_in", "S_out"])
    with pytest.raises(CommandError, match=r"depart lane 1 .* no connection to the next road"):
        e.commands.set_route(vid, ["W_in", "N_out"])
    e.commands.set_route(vid, WE)  # both lanes go straight
    assert _roads(e, h) == WE
    e.step()
    assert veh.status[h] == VehicleStatus.running.code
    assert e.network.link_ids[int(veh.link[h])] == "W_in_1"


def test_a_lane_that_cannot_reach_the_new_road_needs_a_lane_change(
    make_engine: MakeEngine, junction_builder: Callable[..., ScenarioBuilder]
) -> None:
    e = make_engine(junction_builder(lanes=2).build())
    veh = e.vehicles
    vid = e.commands.add_vehicle(route=["W_in", "S_out"], depart_lane=1)
    e.step()
    h = veh.handle_of(vid)
    e.commands.set_route(vid, ["W_in", "N_out"])
    assert veh.valid_mask[h] == 0b01 and int(veh.next_conn[h]) == -1  # mandatory change


def test_rejected_commands_change_nothing(engine: Engine) -> None:
    before = engine.digest()
    with pytest.raises(CommandError):
        engine.commands.add_vehicle(route=["W_in", "W_out"])
    with pytest.raises(CommandError):
        engine.commands.add_vehicle(origin="W_out", destination="E_out")
    assert engine.digest() == before  # not even the default id counter or an RNG stream
    assert engine.commands.add_vehicle(route=WE) == "api.0"
