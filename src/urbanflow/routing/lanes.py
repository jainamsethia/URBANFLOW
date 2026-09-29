"""Lane and connector choice along a route (plan I.2)."""

from __future__ import annotations

from collections.abc import Sequence

from urbanflow.core.types import IntArray
from urbanflow.network.compiled import CompiledNetwork

__all__ = ["plan_connector", "valid_mask"]


def valid_mask(net: CompiledNetwork, road: int, next_road: int) -> int:
    """Bit ``k`` set iff lane ``k`` of ``road`` has a connector to ``next_road``.

    ``next_road = -1`` means ``road`` is the last road of the route: every lane is valid.
    """
    if next_road < 0:
        return (1 << int(net.road_n_lanes[road])) - 1
    return net.road_pair_mask.get((road, next_road), 0)


def plan_connector(
    net: CompiledNetwork, lane: int, route: Sequence[int] | IntArray, cursor: int
) -> int:
    """The connector a vehicle on ``lane`` (on road ``route[cursor]``) takes next, or -1.

    -1 on the last road of the route, or when ``lane`` has no connector to the next road
    (a mandatory lane change follows, G.4). Among this lane's connectors to the next road:
    prefer one whose ``to_lane`` is valid for the road after it (two-road lookahead), then
    the smallest ``|to_lane index - from_lane index|``, then the lowest connector index.
    """
    if cursor + 1 >= len(route):
        return -1
    road, nxt = int(route[cursor]), int(route[cursor + 1])
    after = int(route[cursor + 2]) if cursor + 2 < len(route) else -1
    ahead_mask = valid_mask(net, nxt, after)
    from_k = int(net.link_lane_index[lane])
    best: tuple[bool, int, int] | None = None
    for c in net.road_pair_conns.get((road, nxt), ()):
        if int(net.conn_from_lane[c - net.n_lanes]) != lane:
            continue
        to_k = int(net.link_lane_index[net.conn_to_lane[c - net.n_lanes]])
        key = (not (ahead_mask >> to_k) & 1, abs(to_k - from_k), c)
        if best is None or key < best:
            best = key
    return -1 if best is None else best[2]
