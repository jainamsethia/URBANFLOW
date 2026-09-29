"""Link transitions after the longitudinal update (plan F.1 step 8).

``pos += dx``; while a vehicle's front is past the end of its link it hops, repeatedly
within one step: lane -> planned connector -> the connector's ``to_lane`` -> ...

* the front passing the end of a lane on the **last road** of the route is an arrival,
  whatever the downstream intersection kind; its time is interpolated within the step;
* an uncommitted vehicle is never advanced past a stop line: ``pos = L``, ``v = 0``, counted
  in ``safety_cap_violations`` (reachable only after a counted violation);
* lane entry: ``route_cursor += 1``, ``v0``, ``valid_mask``, ``next_conn`` (I.2),
  ``committed``/``granted``/``forced``/``commit_seq`` cleared, ``link_waiting_time = 0``,
  then the router's ``on_road_entry`` reroute hook. ``v0`` is also refreshed on connector
  entry (it is the desired speed on the current link, E.4);
* connector entry sets ``lock_conn`` to the connector: it is kept across the lane entry
  until bookkeeping sees the rear leave the connector, so admission still treats the
  vehicle as a foe of that connector's zones (F.3). (Zone grants replace this with
  the E.4 set-on-grant rule.)

Crossing times assume constant acceleration within the step (``(v' - v)/dt``, or the
constant deceleration of an in-step stop): the time to cover ``x`` from speed ``v`` is
``2x / (v + sqrt(v^2 + 2 a x))``. Events: ``vehicle_exited_link``,
``vehicle_entered_link`` and ``vehicle_arrived``, each at its crossing time.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from urbanflow.core.constants import POSITION_EPS
from urbanflow.core.errors import SimulationError
from urbanflow.core.events import EventBuffer, EventType
from urbanflow.core.types import BoolArray, FloatArray, IntArray, VehicleStatus
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.routing.base import Router, RouteTable, RoutingContext
from urbanflow.routing.lanes import plan_connector, valid_mask
from urbanflow.vehicles.car_following import desired_speed
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = ["Advance", "advance_links", "crossing_time"]

_ARRIVED = VehicleStatus.arrived.code


@dataclass(frozen=True, slots=True)
class Advance:
    """Output of :func:`advance_links`."""

    dx: FloatArray
    """Distance actually travelled on the network (arrivals stop at the lane end), m."""
    arrived: IntArray
    """Handles that arrived this step."""
    crossed: IntArray
    """Handles that crossed a stop line (lane -> connector) this step."""
    crossed_committed: BoolArray
    """Their ``committed`` flag at the moment of crossing (invariant I8)."""
    violations: int
    """Uncommitted vehicles clamped at a stop line."""


def crossing_time(v: FloatArray, acc: FloatArray, x: FloatArray, dt: float) -> FloatArray:
    """Time to travel ``x`` from speed ``v`` at constant ``acc``, clipped to ``[0, dt]``."""
    root = np.sqrt(np.maximum(v * v + 2 * acc * x, 0.0))
    denom = v + root
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(denom > 0, 2 * x / denom, 0.0)
    return np.clip(tau, 0.0, dt)


def _enter_lanes(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    routes: RouteTable,
    handles: IntArray,
) -> None:
    """Lane-entry hooks for vehicles whose ``link`` is now a lane of ``route[cursor + 1]``."""
    veh.route_cursor[handles] += 1
    veh.v0[handles] = desired_speed(net, veh, types, handles)
    veh.committed[handles] = False
    veh.granted[handles] = False
    veh.forced[handles] = False
    veh.commit_seq[handles] = -1
    veh.link_waiting_time[handles] = 0
    _replan(net, veh, routes, handles)


def _replan(net: CompiledNetwork, veh: VehicleTable, routes: RouteTable, handles: IntArray) -> None:
    """``valid_mask`` and ``next_conn`` of vehicles on a lane of their current road (I.2)."""
    for h in handles.tolist():
        route = routes.get(int(veh.route_id[h]))
        cursor = int(veh.route_cursor[h])
        nxt = int(route[cursor + 1]) if cursor + 1 < len(route) else -1
        veh.valid_mask[h] = valid_mask(net, int(route[cursor]), nxt)
        veh.next_conn[h] = plan_connector(net, int(veh.link[h]), route, cursor)


def _reroute(
    net: CompiledNetwork,
    veh: VehicleTable,
    routes: RouteTable,
    router: Router,
    handles: IntArray,
    ctx: RoutingContext,
) -> None:
    """Apply ``router.on_road_entry`` reroutes: the rest of the route from the current road."""
    changes = router.on_road_entry(handles, ctx)
    for h in sorted(changes):
        rest = tuple(int(r) for r in changes[h])
        cursor = int(veh.route_cursor[h])
        old = routes.get(int(veh.route_id[h]))
        road = int(old[cursor])
        connected = all(pair in net.road_pair_conns for pair in pairwise(rest))
        if not rest or rest[0] != road or not connected:
            raise SimulationError(
                f'router "{router.name}" returned an invalid route for vehicle '
                f'"{veh.ids[h]}": it must start at road "{net.road_ids[road]}" and be connected'
            )
        veh.route_id[h] = routes.intern((*old[:cursor].tolist(), *rest))
        _replan(net, veh, routes, np.array([h]))


def advance_links(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    routes: RouteTable,
    router: Router,
    run: IntArray,
    dx: FloatArray,
    v_start: FloatArray,
    events: EventBuffer,
    *,
    step: int,
    time: float,
    dt: float,
) -> Advance:
    """F.1 step 8 for ``run`` (``dx`` and ``v_start`` from the longitudinal step).

    ``time`` is the step's start time; events carry ``step`` and their crossing time.
    """
    n_lanes = net.n_lanes
    lengths = net.link_length
    v_new = veh.speed[run]
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(v_new > 0, (v_new - v_start) / dt, -(v_start**2) / (2 * dx))
    veh.pos[run] += dx
    moved = dx.copy()
    arrived: list[IntArray] = []
    crossed: list[IntArray] = []
    crossed_committed: list[BoolArray] = []
    violations = 0
    route_len = routes.lengths

    active = np.flatnonzero(_past_end(net, veh, run))
    while active.size:
        h = run[active]
        link = veh.link[h].astype(np.intp)
        over = veh.pos[h] - lengths[link]
        when = time + crossing_time(v_start[active], acc[active], dx[active] - over, dt)
        on_lane = link < n_lanes
        last = on_lane & (route_len[veh.route_id[h]] - 1 == veh.route_cursor[h])
        committed = veh.committed[h]
        hop_lane = on_lane & ~last & committed
        clamp = on_lane & ~last & ~committed
        hop_conn = ~on_lane
        events.append_many(
            EventType.vehicle_exited_link,
            step,
            when[~clamp],
            handles=h[~clamp],
            uids=veh.uid[h[~clamp]],
            links=link[~clamp],
            intersections=_intersection(net, link[~clamp]),
        )
        # arrivals: the front passed the end of a lane on the last road
        done = h[last]
        if done.size:
            veh.status[done] = _ARRIVED
            veh.active[done] = False
            veh.pos[done] = lengths[link[last]]
            moved[active[last]] -= over[last]
            arrived.append(done)
            events.append_many(
                EventType.vehicle_arrived,
                step,
                when[last],
                handles=done,
                uids=veh.uid[done],
                links=link[last],
            )
        # clamps: never past a stop line without a commit
        stuck = h[clamp]
        if stuck.size:
            veh.pos[stuck] = lengths[link[clamp]]
            veh.speed[stuck] = 0.0
            veh.accel[stuck] = -v_start[active[clamp]] / dt
            moved[active[clamp]] -= over[clamp]
            violations += int(stuck.size)
        # lane -> planned connector
        into_conn = h[hop_lane]
        conn = veh.next_conn[into_conn].astype(np.intp)
        crossed.append(into_conn)
        crossed_committed.append(committed[hop_lane])
        veh.pos[into_conn] -= lengths[link[hop_lane]]
        veh.link[into_conn] = conn
        veh.lock_conn[into_conn] = conn
        veh.v0[into_conn] = desired_speed(net, veh, types, into_conn)
        # connector -> its to_lane (a new road)
        into_lane = h[hop_conn]
        lane = net.conn_to_lane[link[hop_conn] - n_lanes]
        veh.pos[into_lane] -= lengths[link[hop_conn]]
        veh.link[into_lane] = lane
        _enter_lanes(net, veh, types, routes, into_lane)
        entered = h[hop_lane | hop_conn]
        events.append_many(
            EventType.vehicle_entered_link,
            step,
            when[hop_lane | hop_conn],
            handles=entered,
            uids=veh.uid[entered],
            links=veh.link[entered],
            intersections=_intersection(net, veh.link[entered].astype(np.intp)),
        )
        if into_lane.size:
            ctx = RoutingContext(step, time, routes, veh.route_id, veh.route_cursor, events)
            _reroute(net, veh, routes, router, into_lane, ctx)
            route_len = routes.lengths  # reroutes may have interned new routes
        again = hop_lane | hop_conn
        active = active[again][_past_end(net, veh, h[again])]

    def cat(parts: list[IntArray]) -> IntArray:
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.intp)

    return Advance(
        dx=moved,
        arrived=cat(arrived),
        crossed=cat(crossed),
        crossed_committed=(
            np.concatenate(crossed_committed) if crossed_committed else np.zeros(0, dtype=bool)
        ),
        violations=violations,
    )


def _past_end(net: CompiledNetwork, veh: VehicleTable, handles: IntArray) -> BoolArray:
    """Fronts past their link's end; overshoots within ``POSITION_EPS`` (float noise of a
    cap that stops a vehicle exactly at the line) are snapped back to the end instead."""
    end = net.link_length[veh.link[handles]]
    pos = veh.pos[handles]
    snap = (pos > end) & (pos <= end + POSITION_EPS)
    veh.pos[handles[snap]] = end[snap]
    return pos > end + POSITION_EPS


def _intersection(net: CompiledNetwork, links: IntArray) -> IntArray:
    """The intersection of connector links, -1 for lanes (the ``intersection`` event column)."""
    return np.where(links >= net.n_lanes, net.link_intersection[links], -1)
