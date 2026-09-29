"""Per-vehicle statistics after the move (plan F.1 step 9, AG.1 R2)."""

from __future__ import annotations

import numpy as np
import pytest

from urbanflow.core.events import EventType
from urbanflow.core.types import VehicleStatus
from urbanflow.engine.bookkeeping import bookkeeping_step

from .conftest import World

WE = ("W_in", "E_out")
HALT = 0.1


def _book(
    w: World,
    dx: float | list[float] = 0.0,
    v0: float | list[float] = 10.0,
    *,
    arrived: list[int] | None = None,
    dt: float = 1.0,
    halting_speed: float = HALT,
) -> list[tuple[str, int]]:
    run = np.flatnonzero(w.veh.active[: w.veh.top] | np.isin(np.arange(w.veh.top), arrived or []))
    w.events.clear()
    bookkeeping_step(
        w.net,
        w.veh,
        run,
        np.broadcast_to(np.asarray(dx, dtype=float), run.shape).copy(),
        np.broadcast_to(np.asarray(v0, dtype=float), run.shape).copy(),
        np.asarray(arrived or [], dtype=np.intp),
        w.events,
        dt=dt,
        halting_speed=halting_speed,
        step=3,
        time=3.0,
    )
    return [
        (EventType.from_code(int(t)).value, int(u))
        for t, u in zip(w.events.type, w.events.uid, strict=True)
    ]


def test_stop_and_resume_transitions(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", 50.0, 0.05, route=WE)
    uid = int(w.veh.uid[h])
    assert _book(w, dt=0.5) == [("vehicle_stopped", uid)]
    assert w.veh.halting[h] and w.veh.stops[h] == 1
    assert w.veh.waiting_time[h] == 0.5 and w.veh.link_waiting_time[h] == 0.5
    assert _book(w, dt=0.5) == []  # still halting: no new stop
    assert w.veh.stops[h] == 1 and w.veh.waiting_time[h] == 1.0
    w.veh.speed[h] = 5.0
    assert _book(w, dt=0.5) == [("vehicle_resumed", uid)]
    assert not w.veh.halting[h] and w.veh.waiting_time[h] == 1.0
    w.veh.speed[h] = 0.0
    _book(w)
    assert w.veh.stops[h] == 2


def test_departing_at_rest_is_not_a_stop(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", 50.0, 0.0, route=WE)
    w.veh.halting[h] = True  # prev-halting = depart_speed < v_halt (set at insertion)
    assert _book(w) == []
    assert w.veh.stops[h] == 0 and w.veh.waiting_time[h] == 1.0


def test_halting_threshold_comes_from_the_config(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", 50.0, 0.5, route=WE)
    _book(w, halting_speed=1.0)
    assert w.veh.halting[h]


def test_distance_and_free_flow_time(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", 50.0, 8.0, route=WE)
    _book(w, dx=10.0, v0=5.0)
    _book(w, dx=6.0, v0=12.0)
    assert w.veh.distance[h] == 16.0
    assert w.veh.ff_time[h] == pytest.approx(10.0 / 5.0 + 6.0 / 12.0)


def test_stuck_time(junction: World) -> None:
    w = junction
    held = w.place("W_in_0", 240.0, 0.0, route=WE)
    queued = w.place("W_in_0", 200.0, 0.0, route=WE)
    inside = w.place("N_in_0->S_out_0", 3.0, 0.0, route=("N_in", "S_out"))
    w.veh.held[held] = True
    _book(w, dt=0.5)
    _book(w, dt=0.5)
    assert w.veh.stuck_time[held] == 1.0 and w.veh.stuck_time[inside] == 1.0
    assert w.veh.stuck_time[queued] == 0.0  # halting behind a leader is not "stuck"
    w.veh.speed[held] = 1.0
    _book(w)
    assert w.veh.stuck_time[held] == 0.0  # continuous halting only


def test_arrived_vehicles_are_freed_deferred(junction: World) -> None:
    w = junction
    h = w.place("E_out_0", 200.0, 10.0, route=("E_out",))
    vid = w.veh.ids[h]
    w.veh.status[h] = VehicleStatus.arrived.code
    w.veh.active[h] = False
    w.veh.halting[h] = True
    _book(w, dx=4.0, arrived=[h])
    assert vid not in w.veh.id_to_handle and not w.veh.halting[h]
    assert w.veh.distance[h] == 4.0  # the final partial step counts
    assert w.veh.alloc("next") != h  # not reusable before the next recycle
    w.veh.recycle()
    assert w.veh.alloc("again") == h
