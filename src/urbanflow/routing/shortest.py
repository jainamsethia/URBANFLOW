"""Static shortest-path router with a per-OD cache (plan I.1)."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any, ClassVar

import networkx as nx

from urbanflow.core.config import RoutingWeight, SimulationConfig
from urbanflow.core.errors import NotFoundError, SimulationError
from urbanflow.core.types import IntArray
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.routing.base import RoutingContext, register_router

__all__ = ["ShortestPathRouter"]


@register_router("shortest")
class ShortestPathRouter:
    """Dijkstra over the road graph (E.5 #8), concatenated through ``via`` waypoints.

    ``weight`` is ``"freeflow_time"`` or ``"length"``; None takes ``config.routing_weight``
    at :meth:`reset`. Paths are cached per ``(origin, destination, via)``: computed once
    per OD, not per vehicle. It never reroutes.
    """

    name: ClassVar[str] = "shortest"

    def __init__(self, weight: RoutingWeight | None = None) -> None:
        self._requested = weight
        self.weight: RoutingWeight = weight or "freeflow_time"
        self._net: CompiledNetwork | None = None
        self._graph: nx.DiGraph | None = None
        self.cache: dict[tuple[int, int, tuple[int, ...]], tuple[int, ...]] = {}
        """Resolved paths by ``(origin, destination, via)``."""

    def reset(self, net: CompiledNetwork, graph: nx.DiGraph, config: SimulationConfig) -> None:
        """Bind to ``net``/``graph`` and clear the cache."""
        self._net, self._graph = net, graph
        self.weight = self._requested or config.routing_weight
        self.cache.clear()

    def route(self, origin: int, destination: int, via: tuple[int, ...] = ()) -> tuple[int, ...]:
        """Shortest road sequence ``origin -> via... -> destination`` (both ends included)."""
        key = (origin, destination, via)
        path = self.cache.get(key)
        if path is None:
            path = self.cache[key] = self._solve(origin, destination, via)
        return path

    def on_road_entry(
        self,
        handles: IntArray,  # noqa: ARG002
        ctx: RoutingContext,  # noqa: ARG002
    ) -> Mapping[int, tuple[int, ...]]:
        """Static routes: never reroutes."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        """No runtime state: the path cache is a memo of the static road graph."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Nothing to restore (see :meth:`state_dict`)."""

    def _solve(self, origin: int, destination: int, via: tuple[int, ...]) -> tuple[int, ...]:
        net, graph = self._net, self._graph
        if net is None or graph is None:
            raise SimulationError("ShortestPathRouter.reset() must be called before route()")
        stops = (origin, *via, destination)
        for road in stops:
            if not 0 <= road < net.n_roads:
                raise NotFoundError(f"road index {road} out of range ({net.n_roads} roads)")
        path = [origin]
        for a, b in pairwise(stops):
            try:
                leg = nx.dijkstra_path(graph, a, b, weight=self.weight)
            except nx.NetworkXNoPath:
                ids = net.road_ids
                via_text = ", ".join(f'"{ids[r]}"' for r in via)
                raise NotFoundError(
                    f'"{ids[destination]}" is not reachable from "{ids[origin]}"'
                    + (f" via {via_text}" if via else "")
                ) from None
            path += leg[1:]
        return tuple(int(r) for r in path)
