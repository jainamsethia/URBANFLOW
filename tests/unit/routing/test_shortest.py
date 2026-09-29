"""Static shortest-path router (plan I.1): weights, via waypoints, OD cache, unreachable."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pytest

from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import NotFoundError, SimulationError
from urbanflow.core.events import EventBuffer
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.network.graph import build_road_graph
from urbanflow.routing import RouteTable, RoutingContext, ShortestPathRouter
from urbanflow.scenario import Scenario

# O -> X, then either X -> P -> Y (short, 5 m/s) or X -> Q -> Y (long, 30 m/s), then Y -> D.
DIAMOND: dict[str, Any] = {
    "format": "urbanflow.scenario",
    "version": "1.0",
    "meta": {"name": "diamond"},
    "network": {
        "intersections": [
            {"id": "O", "kind": "boundary", "point": [-200, 0]},
            {"id": "X", "kind": "uncontrolled", "point": [0, 0]},
            {"id": "P", "point": [100, 0]},
            {"id": "Q", "point": [100, 200]},
            {"id": "Y", "kind": "uncontrolled", "point": [200, 0]},
            {"id": "D", "kind": "boundary", "point": [400, 0]},
        ],
        "roads": [
            {"id": "O_X", "from": "O", "to": "X", "lanes": [{}]},
            {"id": "X_P", "from": "X", "to": "P", "lanes": [{}], "speed_limit": 5},
            {"id": "P_Y", "from": "P", "to": "Y", "lanes": [{}], "speed_limit": 5},
            {"id": "X_Q", "from": "X", "to": "Q", "lanes": [{}], "speed_limit": 30},
            {"id": "Q_Y", "from": "Q", "to": "Y", "lanes": [{}], "speed_limit": 30},
            {"id": "Y_D", "from": "Y", "to": "D", "lanes": [{}]},
        ],
    },
}


@pytest.fixture(scope="module")
def diamond() -> CompiledNetwork:
    return compile_network(Scenario.from_dict(DIAMOND))


def _router(net: CompiledNetwork, **config: Any) -> ShortestPathRouter:
    router = ShortestPathRouter()
    router.reset(net, build_road_graph(net), SimulationConfig(**config))
    return router


def _ids(net: CompiledNetwork, path: tuple[int, ...]) -> list[str]:
    return [net.road_ids[r] for r in path]


def _r(net: CompiledNetwork, road: str) -> int:
    return net.road_index[road]


def test_weight_time_vs_length(diamond: CompiledNetwork) -> None:
    o, d = _r(diamond, "O_X"), _r(diamond, "Y_D")
    fast = _router(diamond).route(o, d, ())  # default routing_weight = freeflow_time
    assert _ids(diamond, fast) == ["O_X", "X_Q", "Q_Y", "Y_D"]
    short = _router(diamond, routing_weight="length").route(o, d, ())
    assert _ids(diamond, short) == ["O_X", "X_P", "P_Y", "Y_D"]
    explicit = ShortestPathRouter(weight="length")
    explicit.reset(diamond, build_road_graph(diamond), SimulationConfig())  # arg wins
    assert explicit.route(o, d) == short


def test_via_waypoints_are_concatenated(diamond: CompiledNetwork) -> None:
    router = _router(diamond, routing_weight="length")
    o, d = _r(diamond, "O_X"), _r(diamond, "Y_D")
    path = router.route(o, d, (_r(diamond, "Q_Y"),))
    assert _ids(diamond, path) == ["O_X", "X_Q", "Q_Y", "Y_D"]
    assert router.route(o, o, ()) == (o,)


def test_paths_are_cached_per_od(diamond: CompiledNetwork, monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router(diamond)
    calls: list[tuple[int, int]] = []
    real = nx.dijkstra_path

    def counting(graph: nx.DiGraph, a: int, b: int, weight: str) -> list[int]:
        calls.append((a, b))
        return list(real(graph, a, b, weight=weight))

    monkeypatch.setattr(nx, "dijkstra_path", counting)
    o, d = _r(diamond, "O_X"), _r(diamond, "Y_D")
    first = router.route(o, d, ())
    assert router.route(o, d, ()) is first
    assert calls == [(o, d)]
    router.route(o, d, (_r(diamond, "X_P"),))  # a different key: two legs
    assert len(calls) == 3
    assert (o, d, ()) in router.cache
    router.reset(diamond, build_road_graph(diamond), SimulationConfig())
    assert router.cache == {}


def test_unreachable_names_the_roads(diamond: CompiledNetwork) -> None:
    router = _router(diamond)
    with pytest.raises(NotFoundError, match='"O_X" is not reachable from "Y_D"'):
        router.route(_r(diamond, "Y_D"), _r(diamond, "O_X"), ())
    with pytest.raises(NotFoundError, match='via "Y_D"'):
        router.route(_r(diamond, "O_X"), _r(diamond, "Q_Y"), (_r(diamond, "Y_D"),))
    with pytest.raises(NotFoundError, match="out of range"):
        router.route(0, 99, ())


def test_needs_reset_and_never_reroutes(diamond: CompiledNetwork) -> None:
    with pytest.raises(SimulationError, match="reset"):
        ShortestPathRouter().route(0, 1, ())
    ctx = RoutingContext(
        step=0,
        time=0.0,
        routes=RouteTable(),
        route_id=np.zeros(0, dtype=np.int32),
        route_cursor=np.zeros(0, dtype=np.int16),
        events=EventBuffer(),
    )
    assert _router(diamond).on_road_entry(np.array([0, 1]), ctx) == {}
