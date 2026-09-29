"""Insertion queues and safe insertion of waiting vehicles (plan G.8, E.6).

Spawned vehicles wait in one FIFO queue per first road (:func:`enqueue`). Each step,
:func:`insert_step` walks the roads with waiting vehicles in road-index order and tries the
queue heads: at most one insertion per lane per step, and a head that cannot be inserted
blocks its road (FIFO). A new vehicle is placed with its rear at the lane start
(``pos = length``) at a depart speed that satisfies the G.2 cap behind the lane's last
vehicle, respecting space reserved by committed inbound vehicles (F.3) and the vehicles
still on connectors into the lane.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from urbanflow.core.constants import SAFETY_MARGIN
from urbanflow.core.types import FloatArray, IntArray, VehicleStatus
from urbanflow.demand.spawners import SpawnRequest
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.routing.base import RouteTable
from urbanflow.routing.lanes import plan_connector, valid_mask
from urbanflow.vehicles.car_following import desired_speed, safe_speed, stop_budget
from urbanflow.vehicles.table import DEPART_LANE_BEST, DEPART_LANE_FIRST, VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = ["InsertionQueues", "LaneTails", "enqueue", "insert_step", "lane_tails"]

_WAITING = VehicleStatus.waiting_insert.code
_RUNNING = VehicleStatus.running.code


class InsertionQueues:
    """One FIFO deque of waiting handles per first road (E.6).

    Also holds the network's static lane -> incoming-connector index used by the
    connector-vehicle check of :func:`insert_step`.
    """

    def __init__(self, net: CompiledNetwork) -> None:
        self._queues: dict[int, deque[int]] = {}
        order = np.argsort(net.conn_to_lane, kind="stable")
        self.in_conn: IntArray = net.n_lanes + order
        """Connector link ids grouped by target lane (CSR values)."""
        self.in_ptr: IntArray = np.searchsorted(net.conn_to_lane[order], np.arange(net.n_lanes + 1))
        """CSR pointers: connectors into lane ``l`` are ``in_conn[in_ptr[l]:in_ptr[l+1]]``."""

    def __len__(self) -> int:
        """Backlog: vehicles waiting to be inserted."""
        return sum(len(q) for q in self._queues.values())

    def __getitem__(self, road: int) -> deque[int]:
        return self._queues.get(road, deque())

    def push(self, road: int, handle: int) -> None:
        """Append ``handle`` to the queue of ``road``."""
        self._queues.setdefault(road, deque()).append(handle)

    def roads(self) -> list[int]:
        """Roads with waiting vehicles, ascending."""
        return sorted(r for r, q in self._queues.items() if q)

    def clear(self) -> None:
        self._queues.clear()


def enqueue(
    veh: VehicleTable,
    types: VehicleTypes,
    routes: RouteTable,
    queues: InsertionQueues,
    req: SpawnRequest,
) -> int:
    """Create the table row of ``req`` (status ``waiting_insert``) and queue it; its handle."""
    h = veh.alloc(req.vehicle_id)
    t = req.type_idx
    veh.status[h] = _WAITING
    veh.type_idx[h] = t
    veh.source_idx[h] = req.source_idx
    veh.length[h] = types.length[t]
    veh.width[h] = types.width[t]
    veh.speed_factor[h] = req.speed_factor
    veh.route_id[h] = req.route_id
    veh.depart_time[h] = req.depart_time
    veh.depart_lane[h] = req.depart_lane
    veh.depart_speed[h] = req.depart_speed
    queues.push(int(routes.get(req.route_id)[0]), h)
    return h


@dataclass(frozen=True, slots=True)
class LaneTails:
    """Upstream ends of every link, as insertion sees them (from the step's start state)."""

    rear: FloatArray
    """Per lane: rear position of the last (most upstream) vehicle; the lane length if empty."""
    speed: FloatArray
    """Per lane: speed of that vehicle (0 if empty)."""
    b_emerg: FloatArray
    """Per lane: its emergency deceleration (``inf`` if empty)."""
    conn_last: IntArray
    """Per connector: handle of the vehicle nearest its end, -1 if empty."""


def lane_tails(net: CompiledNetwork, veh: VehicleTable, types: VehicleTypes) -> LaneTails:
    """:class:`LaneTails` of the running vehicles (ties broken by uid)."""
    run = veh.running()
    order = run[np.lexsort((veh.uid[run], veh.pos[run], veh.link[run]))]
    change = np.flatnonzero(np.diff(veh.link[order])) + 1
    first = order[np.r_[0, change]] if order.size else order  # smallest pos per link
    last = order[np.r_[change - 1, -1]] if order.size else order  # largest pos per link
    n_lanes = net.n_lanes
    rear = net.link_length[:n_lanes].copy()
    speed = np.zeros(n_lanes)
    b_emerg = np.full(n_lanes, np.inf)
    tail = first[veh.link[first] < n_lanes]
    lanes = veh.link[tail]
    rear[lanes] = veh.pos[tail] - veh.length[tail]
    speed[lanes] = veh.speed[tail]
    b_emerg[lanes] = types.emergency_decel[veh.type_idx[tail]]
    conn_last = np.full(net.n_conn, -1, dtype=np.intp)
    head = last[veh.link[last] >= n_lanes]
    conn_last[veh.link[head] - n_lanes] = head
    return LaneTails(rear, speed, b_emerg, conn_last)


def insert_step(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    routes: RouteTable,
    queues: InsertionQueues,
    tails: LaneTails,
    reserved: FloatArray,
    *,
    dt: float,
    time: float,
    limit: float = math.inf,
) -> IntArray:
    """G.8: insert queue heads; returns the inserted handles (in insertion order).

    ``reserved`` is F.3's per-lane ``reserved(l)``; ``limit`` caps the number of insertions
    (``max_vehicles``). Inserted vehicles become running on their lane with ``pos =
    length``, the depart speed, ``v0``, ``valid_mask``, ``next_conn`` and ``insert_time``.
    """
    inserted: list[int] = []
    used: set[int] = set()  # lanes that received a vehicle this step (membership only)
    for road in queues.roads():
        queue = queues[road]
        while queue and len(inserted) < limit:
            h = queue[0]
            if not _try_insert(net, veh, types, routes, queues, tails, reserved, used, h, dt):
                break  # FIFO: the head blocks its road
            queue.popleft()
            veh.insert_time[h] = time
            inserted.append(h)
    return np.asarray(inserted, dtype=np.intp)


def _try_insert(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    routes: RouteTable,
    queues: InsertionQueues,
    tails: LaneTails,
    reserved: FloatArray,
    used: set[int],
    h: int,
    dt: float,
) -> bool:
    route = routes.get(int(veh.route_id[h]))
    first = int(route[0])
    mask = valid_mask(net, first, int(route[1]) if len(route) > 1 else -1)
    start = int(net.road_lane_start[first])
    valid = [start + k for k in range(int(net.road_n_lanes[first])) if mask >> k & 1]
    code = int(veh.depart_lane[h])
    if code == DEPART_LANE_BEST:  # most free space, then lowest index
        candidates = sorted(valid, key=lambda lane: (reserved[lane] - tails.rear[lane], lane))
    elif code == DEPART_LANE_FIRST:
        candidates = valid[:1]
    else:  # the lane drawn at spawn ("random") or the requested index
        candidates = [start + code]
    lane = next((c for c in candidates if c not in used), -1)
    if lane < 0:
        return False
    t = int(veh.type_idx[h])
    length, s0 = float(veh.length[h]), float(types.min_gap[t])
    x_r = float(tails.rear[lane])
    if x_r - length - s0 < reserved[lane]:
        return False
    # stationary solution of v dt + v^2 / (2 b_hat) <= C behind the lane's last vehicle
    budget = x_r - length + tails.speed[lane] ** 2 / (2 * tails.b_emerg[lane]) - SAFETY_MARGIN
    if budget < 0:
        return False
    b_hat = float(types.b_hat[t])
    bound = b_hat * (-dt + math.sqrt(dt * dt + 2 * budget / b_hat))
    hs = np.array([h])
    v0 = float(desired_speed(net, veh, types, hs, np.array([lane]))[0])
    requested = float(veh.depart_speed[h])
    v_d = min(v0, bound) if math.isnan(requested) else requested
    if v_d > bound:
        return False
    # vehicles nearest the end of each connector into the lane must be able to follow
    conns = queues.in_conn[queues.in_ptr[lane] : queues.in_ptr[lane + 1]]
    js = tails.conn_last[conns - net.n_lanes]
    js = js[js >= 0]
    if js.size:
        jt = veh.type_idx[js]
        gap = net.link_length[veh.link[js]] - veh.pos[js]
        b_new = np.full(js.size, types.emergency_decel[t])
        budget_j = stop_budget(gap, np.full(js.size, v_d), b_new)
        v_j = veh.speed[js]
        r = safe_speed(v_j, budget_j, types.b_hat[jt], dt)
        if np.any(r < v_j - types.emergency_decel[jt] * dt):
            return False
    veh.status[h] = _RUNNING
    veh.active[h] = True
    veh.link[h] = lane
    veh.pos[h] = length
    veh.speed[h] = v_d
    veh.accel[h] = 0.0
    veh.v0[h] = v0
    veh.route_cursor[h] = 0
    veh.valid_mask[h] = mask
    veh.next_conn[h] = plan_connector(net, lane, route, 0)
    used.add(lane)
    return True
