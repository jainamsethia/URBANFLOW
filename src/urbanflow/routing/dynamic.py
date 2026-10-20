"""``dynamic``: shortest paths by current travel times, with en-route rerouting (plan I.1).

Each road's travel time is learnt from the vehicles that drove it: the time from entering
the road to entering the next one (so it includes the usual wait at its stop line), as an
exponential average per road. Every ``reroute_period`` seconds the weights are refreshed
with these estimates, never below free flow and raised to the age of the longest-waiting
vehicle still on the road (so a road that stops moving shows up before anyone leaves it).
New vehicles get the current fastest path. A vehicle whose route came from this router
(an origin/destination flow without waypoints, not a fixed route) is offered a new path
from the road it just entered; it takes it only if it is more than ``REROUTE_HYSTERESIS``
(10 %) faster than the rest of its current route, so traffic does not flip back and forth.
Estimates only change at period boundaries, so runs stay deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any, ClassVar

import networkx as nx
import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.config import RoutingWeight, SimulationConfig
from urbanflow.core.errors import NotFoundError, SimulationError
from urbanflow.core.events import EventType
from urbanflow.core.types import FloatArray, IntArray
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.routing.base import RoutingContext, register_router
from urbanflow.routing.shortest import ShortestPathRouter

__all__ = ["DynamicRouter"]

MIN_SPEED = 1.0  # m/s; floor of speed limits in free-flow times
SMOOTHING = 0.2  # weight of each new traversal time in a road's estimate


@register_router("dynamic")
class DynamicRouter(ShortestPathRouter):
    """Fastest paths by live travel times; vehicles reroute around jams."""

    name: ClassVar[str] = "dynamic"

    def __init__(self, weight: RoutingWeight | None = None) -> None:
        super().__init__(weight)
        self.period = C.REROUTE_PERIOD
        self.travel_time: FloatArray = np.zeros(0)
        """Current per-road travel-time estimates, s."""
        self._freeflow: FloatArray = np.zeros(0)
        self._next_refresh = 0.0
        self._learnt: FloatArray = np.zeros(0)
        self._entered: dict[int, tuple[int, float]] = {}
        """uid -> (road, time it entered the road)."""
        self._seen = (-1, 0)
        """(step, events already read in that step)."""
        self._flexible: set[tuple[int, ...]] = set()
        self._timed: nx.DiGraph | None = None
        self._paths: dict[tuple[int, int, tuple[int, ...]], tuple[int, ...]] = {}

    def reset(self, net: CompiledNetwork, graph: nx.DiGraph, config: SimulationConfig) -> None:
        """Free-flow estimates; the first refresh after one period."""
        super().reset(net, graph, config)
        self.period = config.reroute_period
        self._freeflow = net.road_length / np.maximum(net.road_speed_limit, MIN_SPEED)
        self.travel_time = self._freeflow.copy()
        self._learnt = self._freeflow.copy()
        self._entered.clear()
        self._seen = (-1, 0)
        self._next_refresh = self.period
        self._flexible.clear()
        self._timed = graph.copy()
        self._apply_weights()

    # ------------------------------------------------------------------ paths
    def _apply_weights(self) -> None:
        if self._timed is None:
            return
        tt = self.travel_time.tolist()
        nx.set_edge_attributes(self._timed, {(a, b): tt[b] for a, b in self._timed.edges}, "tt")
        self._paths.clear()

    def _fastest(self, origin: int, destination: int, via: tuple[int, ...]) -> tuple[int, ...]:
        key = (origin, destination, via)
        path = self._paths.get(key)
        if path is not None:
            return path
        if self._timed is None or self._net is None:
            raise SimulationError("DynamicRouter.reset() must be called before route()")
        legs = [origin]
        for a, b in pairwise((origin, *via, destination)):
            try:
                legs += nx.dijkstra_path(self._timed, a, b, weight="tt")[1:]
            except nx.NetworkXNoPath:
                ids = self._net.road_ids
                raise NotFoundError(
                    f'"{ids[destination]}" is not reachable from "{ids[origin]}"'
                ) from None
        path = self._paths[key] = tuple(int(r) for r in legs)
        return path

    def route(self, origin: int, destination: int, via: tuple[int, ...] = ()) -> tuple[int, ...]:
        """The currently fastest path (routes without waypoints may be rerouted later)."""
        path = self._fastest(origin, destination, via)
        if not via:
            self._flexible.add(path)
        return path

    def _cost(self, roads: tuple[int, ...]) -> float:
        tt = self.travel_time
        return float(sum(tt[r] for r in roads[1:]))

    # ------------------------------------------------------------------ rerouting
    def _learn(self, ctx: RoutingContext) -> None:
        """Traversal times from this step's road entries (events not read before)."""
        net = self._net
        if net is None:
            return
        ev = ctx.events
        start = self._seen[1] if self._seen[0] == ctx.step else 0
        self._seen = (ctx.step, len(ev))
        if start >= len(ev):
            return
        kinds = ev.type[start:]
        pick = np.flatnonzero(kinds == EventType.vehicle_entered_link.code)
        links = ev.link[start:][pick]
        on_lane = links < net.n_lanes
        uids = ev.uid[start:][pick][on_lane].tolist()
        times = ev.time[start:][pick][on_lane].tolist()
        roads = net.link_road[links[on_lane]].tolist()
        for u, road, t in zip(uids, roads, times, strict=True):
            prev = self._entered.get(u)
            if prev is not None and prev[0] == road:
                continue
            if prev is not None:
                r0 = prev[0]
                self._learnt[r0] += SMOOTHING * ((t - prev[1]) - self._learnt[r0])
            self._entered[u] = (road, t)

    def _refresh(self, ctx: RoutingContext) -> None:
        net = self._net
        if net is None:
            return
        tt = np.maximum(self._learnt, self._freeflow)
        if ctx.running is not None and ctx.uid is not None:
            alive = set(ctx.uid[ctx.running].tolist())
            self._entered = {u: e for u, e in self._entered.items() if u in alive}
            for road, since in self._entered.values():
                tt[road] = max(tt[road], ctx.time - since)
        self.travel_time = tt
        self._apply_weights()

    def on_road_entry(
        self, handles: IntArray, ctx: RoutingContext
    ) -> Mapping[int, tuple[int, ...]]:
        """Faster paths for router-planned vehicles that just entered a road."""
        self._learn(ctx)
        while ctx.time >= self._next_refresh - 1e-9:
            self._refresh(ctx)
            self._next_refresh += self.period
        changes: dict[int, tuple[int, ...]] = {}
        for h in handles.tolist():
            full = tuple(ctx.routes.get(int(ctx.route_id[h])).tolist())
            cursor = int(ctx.route_cursor[h])
            if full not in self._flexible or cursor + 2 >= len(full):
                continue  # fixed route, or nothing left to choose
            rest = full[cursor:]
            better = self._fastest(rest[0], rest[-1], ())
            if better != rest and self._cost(better) < (1 - C.REROUTE_HYSTERESIS) * self._cost(
                rest
            ):
                changes[h] = better
                self._flexible.add(full[:cursor] + better)
        return changes

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict[str, Any]:
        """Travel-time estimates, the next refresh time and the reroutable routes."""
        return {
            "travel_time": self.travel_time.tolist(),
            "learnt": self._learnt.tolist(),
            "entered": sorted([u, r, t] for u, (r, t) in self._entered.items()),
            "seen": list(self._seen),
            "next_refresh": self._next_refresh,
            "flexible": sorted(list(r) for r in self._flexible),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state_dict` output."""
        self.travel_time = np.asarray(state["travel_time"], dtype=np.float64)
        self._learnt = np.asarray(state["learnt"], dtype=np.float64)
        self._entered = {int(u): (int(r), float(t)) for u, r, t in state["entered"]}
        self._seen = (int(state["seen"][0]), int(state["seen"][1]))
        self._next_refresh = float(state["next_refresh"])
        self._flexible = {tuple(int(x) for x in r) for r in state["flexible"]}
        if self._timed is not None:
            self._apply_weights()
