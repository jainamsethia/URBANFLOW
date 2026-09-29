"""Road graph (plan E.5 rule 8): nodes = roads, edges = movements, deterministic order."""

from __future__ import annotations

import networkx as nx
import pytest

from urbanflow.network import CompiledNetwork
from urbanflow.network.graph import build_road_graph


def test_edges_are_movements(demo_net: CompiledNetwork) -> None:
    graph = build_road_graph(demo_net)
    assert list(graph.nodes) == list(range(demo_net.n_roads))
    expected = sorted(
        zip(demo_net.mov_from_road.tolist(), demo_net.mov_to_road.tolist(), strict=True)
    )
    assert list(graph.edges) == expected  # added in sorted order
    for m in range(demo_net.n_movements):
        r, r_next = int(demo_net.mov_from_road[m]), int(demo_net.mov_to_road[m])
        assert graph.edges[r, r_next]["movement"] == m


def test_edge_weights(demo_net: CompiledNetwork) -> None:
    graph = build_road_graph(demo_net)
    w_in, e_out = demo_net.road_index["W_in"], demo_net.road_index["E_out"]
    edge = graph.edges[w_in, e_out]
    assert edge["length"] == pytest.approx(191.6)
    assert edge["freeflow_time"] == pytest.approx(191.6 / 13.89)


def test_shortest_path(demo_net: CompiledNetwork) -> None:
    graph = build_road_graph(demo_net)
    n_in, s_out = demo_net.road_index["N_in"], demo_net.road_index["S_out"]
    assert nx.shortest_path(graph, n_in, s_out, weight="freeflow_time") == [n_in, s_out]
    with pytest.raises(nx.NetworkXNoPath):
        nx.shortest_path(graph, s_out, n_in)  # exits lead nowhere
