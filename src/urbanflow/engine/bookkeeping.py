"""Per-vehicle statistics after the move (plan F.1 step 9, AG.1 R2, J.1).

* ``distance += dx``; ``ff_time += dx / v0`` with ``v0`` of the link held at step start;
* ``halting = speed < config.halting_speed`` (running vehicles, end-of-step state);
  moving -> halting is a ``vehicle_stopped`` event and ``stops += 1``, halting -> moving a
  ``vehicle_resumed`` event;
* ``waiting_time`` (trip-cumulative) and ``link_waiting_time`` (reset on link entry) grow
  by ``dt`` while halting;
* ``stuck_time`` counts continuous halting while ``held`` or on a connector (watchdog input);
* arrived vehicles are freed (deferred: their handles are reusable from the next step).
"""

from __future__ import annotations

import numpy as np

from urbanflow.core.events import EventBuffer, EventType
from urbanflow.core.types import FloatArray, IntArray
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.vehicles.table import VehicleTable

__all__ = ["bookkeeping_step"]


def bookkeeping_step(
    net: CompiledNetwork,
    veh: VehicleTable,
    run: IntArray,
    dx: FloatArray,
    v0_start: FloatArray,
    arrived: IntArray,
    events: EventBuffer,
    *,
    dt: float,
    halting_speed: float,
    step: int,
    time: float,
) -> None:
    """F.1 step 9 for ``run`` (the vehicles that ran this step, arrivals included).

    ``dx`` is the distance moved on the network and ``v0_start`` each vehicle's desired
    speed at the start of the step; events are stamped ``step`` and ``time`` (step end).
    """
    veh.distance[run] += dx
    veh.ff_time[run] += dx / v0_start
    live = run[veh.active[run]]
    halting = veh.speed[live] < halting_speed
    was = veh.halting[live]
    veh.halting[live] = halting
    veh.halting[arrived] = False
    veh.stops[live[halting & ~was]] += 1
    wait = dt * halting
    veh.waiting_time[live] += wait
    veh.link_waiting_time[live] += wait
    stuck = halting & (veh.held[live] | (veh.link[live] >= net.n_lanes))
    veh.stuck_time[live] = np.where(stuck, veh.stuck_time[live] + dt, 0.0)
    for kind, h in (
        (EventType.vehicle_stopped, live[halting & ~was]),
        (EventType.vehicle_resumed, live[~halting & was]),
    ):
        events.append_many(kind, step, time, handles=h, uids=veh.uid[h], links=veh.link[h])
    veh.free_deferred(arrived)
