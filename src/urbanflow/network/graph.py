"""Road graph for routing (plan E.5 rule 8): nodes are road indices, edges movements."""

from __future__ import annotations

import networkx as nx

from urbanflow.network.compiled import CompiledNetwork

__all__ = ["build_road_graph"]


def build_road_graph(net: CompiledNetwork) -> nx.DiGraph:
    """A ``networkx.DiGraph`` with a node per road and an edge ``r -> r'`` per movement.

    Edge attributes: ``length = road_length[r']``, ``freeflow_time = road_length[r'] /
    road_speed_limit[r']`` and ``movement`` (its index). Nodes and edges are added in
    sorted index order, so path tie-breaking is deterministic.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(net.n_roads))
    edges = sorted(
        zip(
            net.mov_from_road.tolist(),
            net.mov_to_road.tolist(),
            range(net.n_movements),
            strict=True,
        )
    )
    length, speed = net.road_length.tolist(), net.road_speed_limit.tolist()
    for r, r_next, m in edges:
        graph.add_edge(
            r,
            r_next,
            length=length[r_next],
            freeflow_time=length[r_next] / speed[r_next],
            movement=m,
        )
    return graph
