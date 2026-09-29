"""Interned routes, the routing context and the router registry (plan E.6, I.1)."""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
import pytest

from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import ConfigError, NotFoundError
from urbanflow.core.events import EventBuffer
from urbanflow.core.types import IntArray
from urbanflow.network import CompiledNetwork
from urbanflow.routing import (
    RouteTable,
    RoutingContext,
    ShortestPathRouter,
    register_router,
    router_registry,
)


def test_intern_shares_identical_routes() -> None:
    table = RouteTable()
    a = table.intern([0, 2, 5])
    b = table.intern((1, 3))
    assert (a, b) == (0, 1)
    assert table.intern(np.array([0, 2, 5], dtype=np.int32)) == a  # same roads, same id
    assert table.intern([0, 2]) == 2  # a prefix is a different route
    assert len(table) == 3
    route = table.get(a)
    assert route.dtype == np.int32
    assert route.tolist() == [0, 2, 5]
    assert not route.flags.writeable
    assert [r.tolist() for r in table.routes] == [[0, 2, 5], [1, 3], [0, 2]]


def test_bad_routes() -> None:
    table = RouteTable()
    with pytest.raises(ValueError, match="at least one road"):
        table.intern([])
    with pytest.raises(NotFoundError, match="unknown route id 0"):
        table.get(0)
    table.intern([4])
    with pytest.raises(NotFoundError):
        table.get(-1)


def test_router_registry() -> None:
    assert router_registry.get("shortest") is ShortestPathRouter
    assert ShortestPathRouter.name == "shortest"

    @register_router("test_fixed")
    class Fixed:
        def reset(self, net: CompiledNetwork, graph: nx.DiGraph, config: SimulationConfig) -> None:
            self.net, self.graph, self.config = net, graph, config

        def route(self, origin: int, destination: int, via: tuple[int, ...]) -> tuple[int, ...]:
            return (origin, *via, destination)

        def on_road_entry(
            self, handles: IntArray, ctx: RoutingContext
        ) -> Mapping[int, tuple[int, ...]]:
            return {int(h): tuple(ctx.routes.get(0).tolist()) for h in handles}

    assert Fixed.name == "test_fixed"  # type: ignore[attr-defined]
    assert router_registry.get("test_fixed") is Fixed
    with pytest.raises(ConfigError, match="already registered"):
        router_registry.register("test_fixed", ShortestPathRouter)
    with pytest.raises(NotFoundError, match='did you mean "shortest"'):
        router_registry.get("shortes")

    routes = RouteTable()
    routes.intern([3, 4])
    ctx = RoutingContext(
        step=5,
        time=5.0,
        routes=routes,
        route_id=np.zeros(2, dtype=np.int32),
        route_cursor=np.zeros(2, dtype=np.int16),
        events=EventBuffer(),
    )
    assert Fixed().on_road_entry(np.array([1]), ctx) == {1: (3, 4)}
