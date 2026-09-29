"""Lane and connector choice (plan I.2): valid masks, two-road lookahead, tie rules."""

from __future__ import annotations

import pytest

from urbanflow.network import CompiledNetwork
from urbanflow.routing import plan_connector, valid_mask


def _roads(net: CompiledNetwork, *ids: str) -> list[int]:
    return [net.road_index[r] for r in ids]


def _conn(net: CompiledNetwork, name: str) -> int:
    return net.link_index[name]


def test_valid_mask(corridor_net: CompiledNetwork) -> None:
    w, j12, e, n, s = _roads(corridor_net, "W_J1", "J1_J2", "J2_E", "J2_N", "J2_S")
    assert valid_mask(corridor_net, w, j12) == 0b11
    assert valid_mask(corridor_net, j12, n) == 0b001  # L{0}
    assert valid_mask(corridor_net, j12, e) == 0b010  # S{1}
    assert valid_mask(corridor_net, j12, s) == 0b100  # R{2}
    assert valid_mask(corridor_net, j12, -1) == 0b111  # last road: every lane
    assert valid_mask(corridor_net, e, -1) == 0b11
    assert valid_mask(corridor_net, n, w) == 0  # no movement


@pytest.mark.parametrize(
    ("lane", "last", "expected"),
    [
        # lookahead first: the to_lane must be valid for the road after next
        ("W_J1_0", "J2_N", "W_J1_0->J1_J2_0"),
        ("W_J1_0", "J2_E", "W_J1_0->J1_J2_1"),
        ("W_J1_1", "J2_S", "W_J1_1->J1_J2_2"),
        ("W_J1_1", "J2_N", "W_J1_1->J1_J2_0"),
    ],
)
def test_two_road_lookahead(
    corridor_net: CompiledNetwork, lane: str, last: str, expected: str
) -> None:
    route = _roads(corridor_net, "W_J1", "J1_J2", last)
    got = plan_connector(corridor_net, corridor_net.link_index[lane], route, 0)
    assert got == _conn(corridor_net, expected)


def test_tie_rules(corridor_net: CompiledNetwork) -> None:
    net = corridor_net
    lane0, lane1 = net.link_index["W_J1_0"], net.link_index["W_J1_1"]
    # no valid target for J2_E from lane 1 (-> 0 and -> 2): both |dk| = 1, lowest index wins
    route = _roads(net, "W_J1", "J1_J2", "J2_E")
    ties = [_conn(net, "W_J1_1->J1_J2_0"), _conn(net, "W_J1_1->J1_J2_2")]
    assert plan_connector(net, lane1, route, 0) == min(ties)
    # J1_J2 is the last road: every target is valid, so the smallest lane shift wins
    short = _roads(net, "W_J1", "J1_J2")
    assert plan_connector(net, lane0, short, 0) == _conn(net, "W_J1_0->J1_J2_0")
    assert plan_connector(net, lane1, short, 0) == min(ties)
    # accepts a numpy route as stored in the RouteTable
    import numpy as np

    assert plan_connector(net, lane0, np.asarray(route, dtype=np.int32), 0) == _conn(
        net, "W_J1_0->J1_J2_1"
    )


def test_invalid_lane_and_last_road(corridor_net: CompiledNetwork) -> None:
    net = corridor_net
    route = _roads(net, "W_J1", "J1_J2", "J2_S")
    j12_0, j12_2 = net.link_index["J1_J2_0"], net.link_index["J1_J2_2"]
    assert plan_connector(net, j12_0, route, 1) == -1  # needs a mandatory lane change
    assert plan_connector(net, j12_2, route, 1) == _conn(net, "J1_J2_2->J2_S_0")
    exit_lane = net.link_index["J2_S_0"]
    assert plan_connector(net, exit_lane, route, 2) == -1  # last road
