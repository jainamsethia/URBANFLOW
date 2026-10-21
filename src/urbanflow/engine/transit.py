"""Transit buses serving their stops (plan G.1 bus stops, I.3).

Every stop of every line (lines in file order, stops in line order) is one row of
:class:`TransitStops`: its road, stop lane (a link id), the position where the bus front
stops, the dwell time and the next stop of the same line. A bus carries the row of its next
stop in ``stop_idx`` (-1 once every stop is served or missed). Stops are matched by road,
not by route index, so a bus moved on by the watchdog (its route becomes the rest of the
route) keeps its remaining stops.

When a bus's next stop is on the *next* road of its route, lane changes (sub-step 5) treat
the lanes with a connector to the stop lane as the only valid ones, so the bus arrives on
the stop lane (a far-side stop just past a junction leaves little room to change lanes).
While the stop is on the road it drives on and its front is at most ``STOP_REACH`` past it:

* lane changes treat the stop lane as the only valid lane, with the urgency measured to
  the stop, so the bus moves over to it and stays there;
* the stop is an obstacle (sub-steps 6-7), like a stop line: IDM gap ``d + s0 - 0.5`` and
  safe-speed cap ``d``, with ``d`` the path distance from the front to the stop, as long as
  the bus can still stop there (``v^2 / (2d) <= b_emerg``). It is seen on the stop lane,
  and already through the junction: on the connector into the stop lane and on the lane
  whose planned connector leads there (a far-side stop needs braking before the junction);
* a bus halting on the stop lane with its front within ``STOP_REACH`` of the stop starts
  dwelling (``dwell_left = dwell``): it neither moves nor changes lanes until the dwell is
  over, then the stop is *served* and the bus heads for its next stop.

A bus whose front passes the stop by more than ``STOP_REACH`` (on any lane), or that leaves
the stop's road without stopping, has *missed* the stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.types import FloatArray, IntArray
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.routing.base import RouteTable
from urbanflow.routing.lanes import feeding_lanes
from urbanflow.scenario.schema import DemandSpec
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = [
    "StopTargets",
    "TransitStops",
    "hold_dwelling",
    "stop_obstacles",
    "stop_targets",
    "transit_step",
]


@dataclass(frozen=True, slots=True)
class TransitStops:
    """Every stop of every transit line (static per scenario)."""

    road: IntArray
    """Road of the stop."""
    lane: IntArray
    """Link id of the stop lane."""
    pos: FloatArray
    """Where the bus front stops, m along the lane."""
    dwell: FloatArray
    """Dwell time, s."""
    next: IntArray
    """Next stop of the same line, -1 after the last."""
    ids: tuple[str, ...] = ()
    """Stop ids (``"{line}.{k}"`` unless given)."""
    _feed: dict[tuple[int, int], int] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def build(cls, net: CompiledNetwork, demand: DemandSpec) -> TransitStops:
        """The stops of the *resolved* ``demand`` (stop lanes default to the curb lane)."""
        road: list[int] = []
        lane: list[int] = []
        pos: list[float] = []
        dwell: list[float] = []
        nxt: list[int] = []
        ids: list[str] = []
        for line in demand.transit:
            base = len(road)
            for k, stop in enumerate(line.stops):
                r = net.road_index[stop.road]
                index = stop.lane if stop.lane is not None else int(net.road_n_lanes[r]) - 1
                road.append(r)
                lane.append(int(net.road_lane_start[r]) + index)
                pos.append(stop.position)
                dwell.append(stop.dwell)
                nxt.append(base + k + 1 if k + 1 < len(line.stops) else -1)
                ids.append(stop.id or f"{line.id}.{k}")
        return cls(
            np.asarray(road, dtype=np.intp),
            np.asarray(lane, dtype=np.intp),
            np.asarray(pos, dtype=np.float64),
            np.asarray(dwell, dtype=np.float64),
            np.asarray(nxt, dtype=np.intp),
            tuple(ids),
        )

    def __len__(self) -> int:
        return int(self.road.size)

    def feeders(self, net: CompiledNetwork, road: int, g: int) -> int:
        """Lane bits of ``road`` with a connector to the lane of stop ``g`` (cached)."""
        key = (road, g)
        if key not in self._feed:
            self._feed[key] = feeding_lanes(net, road, int(self.road[g]), int(self.lane[g]))
        return self._feed[key]


@dataclass(frozen=True, slots=True)
class StopTargets:
    """Per running vehicle (aligned with ``run``): where its next stop asks it to drive."""

    mask: IntArray
    """Lane bits of the current road the bus must use (0: no constraint)."""
    dist: FloatArray
    """Distance that makes the lane change urgent, m: to the stop on its road, NaN (the
    lane end) on the road before it."""
    lane: IntArray
    """Link id of the stop lane when the stop is on this road and ahead, -1 otherwise."""
    pos: FloatArray
    """Stop position, m (NaN when ``lane`` is -1)."""


def stop_targets(
    net: CompiledNetwork,
    veh: VehicleTable,
    routes: RouteTable,
    stops: TransitStops,
    run: IntArray,
) -> StopTargets:
    """The lanes and stop of every bus of ``run`` on a lane (see the module docstring)."""
    n = run.size
    mask = np.zeros(n, dtype=np.int64)
    dist = np.full(n, np.nan)
    lane = np.full(n, -1, dtype=np.intp)
    pos = np.full(n, np.nan)
    g_run = veh.stop_idx[run]
    for i in np.flatnonzero((g_run >= 0) & (veh.link[run] < net.n_lanes)).tolist():
        h, g = int(run[i]), int(g_run[i])
        road, target = int(net.link_road[veh.link[h]]), int(stops.road[g])
        if road == target:
            if veh.pos[h] <= stops.pos[g] + C.STOP_REACH:
                lane[i], pos[i] = stops.lane[g], stops.pos[g]
                mask[i] = 1 << int(net.link_lane_index[stops.lane[g]])
                dist[i] = max(float(stops.pos[g] - veh.pos[h]), 0.0)
            continue
        route, cursor = routes.get(int(veh.route_id[h])), int(veh.route_cursor[h])
        if cursor + 1 < len(route) and int(route[cursor + 1]) == target:
            mask[i] = stops.feeders(net, road, g)
    return StopTargets(mask, dist, lane, pos)


def stop_obstacles(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    stops: TransitStops,
    run: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """``(obstacle_gap, cap_gap)`` of the next stop of every bus of ``run`` that sees it
    (on the stop lane, on the connector into it, or on the lane whose planned connector
    leads there), while it can still stop there; ``inf`` elsewhere."""
    n_lanes = net.n_lanes
    d = np.full(run.size, np.inf)
    g_run = veh.stop_idx[run]
    for i in np.flatnonzero(g_run >= 0).tolist():
        h, g = int(run[i]), int(g_run[i])
        link, stop_lane = int(veh.link[h]), int(stops.lane[g])
        to_go = float(net.link_length[link] - veh.pos[h])  # to the end of the current link
        if link == stop_lane:
            d[i] = float(stops.pos[g] - veh.pos[h])
        elif link >= n_lanes and int(net.conn_to_lane[link - n_lanes]) == stop_lane:
            d[i] = to_go + float(stops.pos[g])
        elif link < n_lanes and veh.next_conn[h] >= 0:
            conn = int(veh.next_conn[h])
            if int(net.conn_to_lane[conn - n_lanes]) == stop_lane:
                d[i] = to_go + float(net.link_length[conn]) + float(stops.pos[g])
    ti = veh.type_idx[run]
    v = veh.speed[run]
    seen = np.isfinite(d) & (d >= -C.STOP_REACH)
    d = np.where(seen, np.maximum(d, 0.0), np.inf)
    active = seen & (v * v <= 2 * d * types.emergency_decel[ti])
    gap = np.where(active, d + types.min_gap[ti] - C.STOP_LINE_CLEARANCE, np.inf)
    return gap, np.where(active, d, np.inf)


def hold_dwelling(veh: VehicleTable, run: IntArray, dx: FloatArray, dt: float) -> None:
    """Dwelling buses stay where they are: ``v' = 0``, ``dx = 0`` (no model call, G.1)."""
    k = np.flatnonzero(veh.dwell_left[run] > 0)
    if k.size:
        h = run[k]
        veh.accel[h] = -veh.speed[h] / dt
        veh.speed[h] = 0.0
        dx[k] = 0.0


def transit_step(
    net: CompiledNetwork,
    veh: VehicleTable,
    routes: RouteTable,
    stops: TransitStops,
    run: IntArray,
    *,
    dt: float,
    halting_speed: float,
) -> tuple[int, int]:
    """After the step's movement: dwell timers, dwell starts and missed stops of the buses
    of ``run`` that are still running. Returns ``(served, missed)`` stop counts."""
    served = missed = 0
    for h in run[(veh.stop_idx[run] >= 0) & veh.active[run]].tolist():
        g = int(veh.stop_idx[h])
        if veh.dwell_left[h] > 0:  # dwelling: count down, then the stop is served
            left = float(veh.dwell_left[h]) - dt
            if left > C.TIME_EPS:
                veh.dwell_left[h] = left
                continue
            veh.dwell_left[h] = 0.0
            veh.stop_idx[h] = stops.next[g]
            served += 1
            continue
        link = int(veh.link[h])
        if link < net.n_lanes and int(net.link_road[link]) == int(stops.road[g]):
            beyond = float(veh.pos[h] - stops.pos[g])
            if (
                link == int(stops.lane[g])
                and abs(beyond) <= C.STOP_REACH
                and (veh.speed[h] < halting_speed)
            ):
                if stops.dwell[g] > 0:
                    veh.dwell_left[h] = stops.dwell[g]
                else:
                    veh.stop_idx[h] = stops.next[g]
                    served += 1
                continue
            if beyond <= C.STOP_REACH:
                continue  # still ahead
        else:
            route = routes.get(int(veh.route_id[h]))
            rest = route[int(veh.route_cursor[h]) + 1 :]
            if int(stops.road[g]) in rest.tolist():
                continue  # the stop's road is still to come
        veh.stop_idx[h] = stops.next[g]  # passed it, or left its road without stopping
        missed += 1
    return served, missed
