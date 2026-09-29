"""Vehicle commands (plan F.4): validation, effect in the next step, events and the log."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from urbanflow.core.errors import CommandError, NotFoundError
from urbanflow.core.events import EventType
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
