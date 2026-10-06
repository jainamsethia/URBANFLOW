"""Router protocol, routing context, interned route storage and the router registry (I.1, E.6).

A route is a tuple of road indices including both ends. :class:`RouteTable` interns each
distinct route once, so vehicles store a small ``route_id`` and OD routes are shared.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import networkx as nx
import numpy as np

from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import NotFoundError
from urbanflow.core.events import EventBuffer
from urbanflow.core.registry import Registry
from urbanflow.core.types import IntArray
from urbanflow.network.compiled import CompiledNetwork

__all__ = ["RouteTable", "Router", "RoutingContext", "register_router", "router_registry"]


class RouteTable:
    """Interned road-index routes (E.6): ``intern(roads) -> id``, ``get(id) -> roads``."""

    def __init__(self) -> None:
        self.routes: list[IntArray] = []
        """Read-only ``int32`` road arrays by route id."""
        self._ids: dict[tuple[int, ...], int] = {}
        self._lengths = np.zeros(64, dtype=np.intp)  # grows by doubling

    @property
    def lengths(self) -> IntArray:
        """Number of roads of each route, by route id (read-only; re-read after
        :meth:`intern`, which may reallocate it)."""
        view = self._lengths[: len(self.routes)]
        view.flags.writeable = False
        return view

    def __len__(self) -> int:
        return len(self.routes)

    def intern(self, roads: Sequence[int]) -> int:
        """The id of ``roads`` (a new id the first time this sequence is seen)."""
        key = tuple(int(r) for r in roads)
        if not key:
            raise ValueError("a route needs at least one road")
        route_id = self._ids.get(key)
        if route_id is None:
            arr = np.asarray(key, dtype=np.int32)
            arr.setflags(write=False)
            route_id = self._ids[key] = len(self.routes)
            self.routes.append(arr)
            if route_id == self._lengths.size:
                self._lengths = np.resize(self._lengths, 2 * route_id)
            self._lengths[route_id] = len(key)
        return route_id

    def get(self, route_id: int) -> IntArray:
        """Road indices of route ``route_id`` (read-only)."""
        if not 0 <= route_id < len(self.routes):
            raise NotFoundError(f"unknown route id {route_id} ({len(self.routes)} routes)")
        return self.routes[route_id]

    def state_dict(self) -> dict[str, Any]:
        """JSON-safe state: ``{"routes": [[road, ...], ...]}`` in id order."""
        return {"routes": [r.tolist() for r in self.routes]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Replace the table by :meth:`state_dict` output (same ids)."""
        self.routes.clear()
        self._ids.clear()
        for roads in state["routes"]:
            self.intern(roads)
        if len(self.routes) != len(state["routes"]):
            raise ValueError("route table state has duplicate routes")


@dataclass(frozen=True, slots=True)
class RoutingContext:
    """What a router sees in :meth:`Router.on_road_entry` (per-handle vehicle columns)."""

    step: int
    time: float
    routes: RouteTable
    route_id: IntArray
    """``VehicleTable.route_id`` (indexed by handle)."""
    route_cursor: IntArray
    """``VehicleTable.route_cursor`` (indexed by handle)."""
    events: EventBuffer
    """Events of the current step so far (e.g. ``vehicle_exited_link`` travel times)."""


class Router(Protocol):
    """A route source (I.1); register with ``@register_router("name")``.

    A router with runtime state (e.g. travel-time estimates) also implements
    ``state_dict() -> dict`` (JSON-safe) and ``load_state_dict(state)``, which snapshots,
    restores and ``state_digest`` use; a router without them is treated as stateless.
    """

    name: ClassVar[str]

    def reset(self, net: CompiledNetwork, graph: nx.DiGraph, config: SimulationConfig) -> None:
        """Bind to a network and its road graph (called on every simulation reset)."""
        ...

    def route(self, origin: int, destination: int, via: tuple[int, ...]) -> tuple[int, ...]:
        """Road indices from ``origin`` to ``destination`` through ``via``, both ends included.

        Raises ``NotFoundError`` if the destination is unreachable.
        """
        ...

    def on_road_entry(
        self, handles: IntArray, ctx: RoutingContext
    ) -> Mapping[int, tuple[int, ...]]:
        """Optional reroutes ``{handle: new remaining route}`` of vehicles that entered a road."""
        ...


router_registry: Registry[type[Router]] = Registry("routers")


def register_router[R: type[Router]](name: str) -> Callable[[R], R]:
    """Class decorator registering a :class:`Router` as ``name`` (sets ``cls.name`` if unset)."""

    def decorator(cls: R) -> R:
        if "name" not in vars(cls):
            cls.name = name
        router_registry.register(name, cls)
        return cls

    return decorator
