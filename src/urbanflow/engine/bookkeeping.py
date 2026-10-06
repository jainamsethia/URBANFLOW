"""Per-vehicle statistics and the watchdog after the move (plan F.1 step 9, F.3, AG.1 R2, J.1).

* ``distance += dx``; ``ff_time += dx / v0`` with ``v0`` of the link held at step start;
* ``halting = speed < config.halting_speed`` (running vehicles, end-of-step state);
  moving -> halting is a ``vehicle_stopped`` event and ``stops += 1``, halting -> moving a
  ``vehicle_resumed`` event;
* ``waiting_time`` (trip-cumulative) and ``link_waiting_time`` (reset on link entry) grow
  by ``dt`` while halting;
* ``stuck_time`` counts continuous halting while ``held`` or on a connector;
* **watchdog** (``deadlock_timeout > 0``): every vehicle with ``stuck_time >=
  deadlock_timeout`` is teleported, connector vehicles first, then by uid. On its last
  road it arrives; otherwise it goes to the back of the insertion queue of its next road
  (``waiting_insert``, route = the remaining roads, depart lane ``best`` at the maximum
  safe speed), so insertion re-checks space and reservations (G.8). Each teleport counts
  in ``teleports`` and emits ``vehicle_teleported``;
* arrived vehicles are freed (deferred: their handles are reusable from the next step).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from urbanflow.core.constants import LOG_MAX_IDS
from urbanflow.core.events import EventBuffer, EventType
from urbanflow.core.types import FloatArray, IntArray, VehicleStatus
from urbanflow.demand.insertion import InsertionQueues
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.routing.base import RouteTable
from urbanflow.vehicles.table import DEPART_LANE_BEST, VehicleTable

__all__ = ["Bookkeeping", "bookkeeping_step"]

_log = logging.getLogger("urbanflow.engine")
_ARRIVED = VehicleStatus.arrived.code
_WAITING = VehicleStatus.waiting_insert.code


@dataclass(frozen=True, slots=True)
class Bookkeeping:
    """Output of :func:`bookkeeping_step`."""

    teleported: IntArray
    """Handles teleported by the watchdog this step."""
    arrived: IntArray
    """Those of them that arrived (they were on their last road)."""


def bookkeeping_step(
    net: CompiledNetwork,
    veh: VehicleTable,
    routes: RouteTable,
    queues: InsertionQueues,
    run: IntArray,
    dx: FloatArray,
    v0_start: FloatArray,
    arrived: IntArray,
    events: EventBuffer,
    *,
    dt: float,
    halting_speed: float,
    deadlock_timeout: float,
    step: int,
    time: float,
) -> Bookkeeping:
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
    on_lane = veh.link[live] < net.n_lanes
    stuck = halting & (veh.held[live] | ~on_lane)
    veh.stuck_time[live] = np.where(stuck, veh.stuck_time[live] + dt, 0.0)
    for kind, h in (
        (EventType.vehicle_stopped, live[halting & ~was]),
        (EventType.vehicle_resumed, live[~halting & was]),
    ):
        events.append_many(kind, step, time, handles=h, uids=veh.uid[h], links=veh.link[h])
    none = np.zeros(0, dtype=np.intp)
    out = Bookkeeping(none, none)
    if deadlock_timeout > 0:
        out = _watchdog(net, veh, routes, queues, live, events, deadlock_timeout, step, time)
    veh.free_deferred(np.concatenate([arrived, out.arrived]))
    return out


def _watchdog(
    net: CompiledNetwork,
    veh: VehicleTable,
    routes: RouteTable,
    queues: InsertionQueues,
    live: IntArray,
    events: EventBuffer,
    timeout: float,
    step: int,
    time: float,
) -> Bookkeeping:
    """Teleport the vehicles stuck for ``timeout`` s or more (see the module docstring)."""
    h = live[veh.stuck_time[live] >= timeout]
    if not h.size:
        return Bookkeeping(h, h)
    h = h[np.lexsort((veh.uid[h], veh.link[h] < net.n_lanes))]  # connector vehicles first
    link = veh.link[h].copy()
    veh.teleports[h] += 1
    veh.active[h] = False
    veh.committed[h] = veh.granted[h] = veh.forced[h] = veh.held[h] = veh.halting[h] = False
    veh.commit_seq[h] = -1
    veh.lock_conn[h] = -1
    veh.next_conn[h] = -1
    veh.stuck_time[h] = veh.link_waiting_time[h] = 0.0
    events.append_many(
        EventType.vehicle_teleported, step, time, handles=h, uids=veh.uid[h], links=link
    )
    done: list[int] = []
    for k in h.tolist():
        route = routes.get(int(veh.route_id[k]))
        cursor = int(veh.route_cursor[k])
        if cursor + 1 >= len(route):  # on its last road: arrives
            veh.status[k] = _ARRIVED
            done.append(k)
            continue
        rest = route[cursor + 1 :]
        veh.status[k] = _WAITING
        veh.route_id[k] = routes.intern(rest.tolist())
        veh.route_cursor[k] = 0
        veh.link[k] = -1
        veh.pos[k] = veh.speed[k] = veh.accel[k] = 0.0
        veh.depart_lane[k] = DEPART_LANE_BEST
        veh.depart_speed[k] = np.nan
        queues.push(int(rest[0]), k)
    arrived = np.asarray(done, dtype=np.intp)
    events.append_many(
        EventType.vehicle_arrived,
        step,
        time,
        handles=arrived,
        uids=veh.uid[arrived],
        links=veh.link[arrived],
    )
    first = ", ".join(veh.ids[k] or "" for k in h[:LOG_MAX_IDS].tolist())
    _log.warning(
        "watchdog: %d vehicle(s) stuck for %g s teleported (%s%s)",
        h.size,
        timeout,
        first,
        ", ..." if h.size > LOG_MAX_IDS else "",
    )
    return Bookkeeping(h, arrived)
