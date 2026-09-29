"""Fixtures shared by the runtime-model tests (routing, vehicles, demand)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.scenario import Scenario

# W -> J1 -> J2 -> {E, N, S}. J1 fans lanes out explicitly (lane 0 -> 0, 1; lane 1 -> 0, 2),
# so connector choice has real alternatives; at J2 the derived mapping of the 3-lane road
# gives L{0} -> J2_N, S{1} -> J2_E, R{2} -> J2_S (E.7 §1.5 step 6).
CORRIDOR: dict[str, Any] = {
    "format": "urbanflow.scenario",
    "version": "1.0",
    "meta": {"name": "corridor"},
    "simulation": {"dt": 1.0, "duration": 600, "seed": 7},
    "network": {
        "intersections": [
            {"id": "W", "kind": "boundary", "point": [-300, 0]},
            {
                "id": "J1",
                "kind": "uncontrolled",
                "point": [0, 0],
                "movements": [
                    {
                        "from_road": "W_J1",
                        "to_road": "J1_J2",
                        "connections": [
                            {"from_lane": 0, "to_lane": 0},
                            {"from_lane": 0, "to_lane": 1},
                            {"from_lane": 1, "to_lane": 0},
                            {"from_lane": 1, "to_lane": 2},
                        ],
                    }
                ],
            },
            {"id": "J2", "kind": "uncontrolled", "point": [300, 0]},
            {"id": "E", "kind": "boundary", "point": [600, 0]},
            {"id": "N", "kind": "boundary", "point": [300, 300]},
            {"id": "S", "kind": "boundary", "point": [300, -300]},
        ],
        "roads": [
            {"id": "W_J1", "from": "W", "to": "J1", "lanes": [{}, {}]},
            {"id": "J1_J2", "from": "J1", "to": "J2", "lanes": [{}, {}, {}]},
            {"id": "J2_E", "from": "J2", "to": "E", "lanes": [{}, {}]},
            {"id": "J2_N", "from": "J2", "to": "N", "lanes": [{}]},
            {"id": "J2_S", "from": "J2", "to": "S", "lanes": [{}]},
        ],
    },
}


def corridor_data(**demand: Any) -> dict[str, Any]:
    """A fresh corridor document with ``demand`` (e.g. ``flows=[...]``)."""
    data = copy.deepcopy(CORRIDOR)
    if demand:
        data["demand"] = demand
    return data


@pytest.fixture
def corridor_doc() -> Callable[..., dict[str, Any]]:
    """``corridor_doc(flows=[...], trips=[...])``: a fresh corridor document."""
    return corridor_data


@pytest.fixture(scope="session")
def corridor() -> Scenario:
    return Scenario.from_dict(CORRIDOR)


@pytest.fixture(scope="session")
def corridor_net(corridor: Scenario) -> CompiledNetwork:
    return compile_network(corridor)
