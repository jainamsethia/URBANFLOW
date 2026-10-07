"""AT-04: every bundled scenario compiles with golden counts, valid render geometry and
deterministic arrays (plan section V)."""

from __future__ import annotations

import hashlib

import jsonschema
import numpy as np
import pytest

import urbanflow
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.scenario.generators import BUNDLED_SCENARIOS
from urbanflow.visualization import RenderGeometry, render_geometry

pytestmark = pytest.mark.acceptance

# name -> (lanes, connectors, [crossing, merging, diverging] conflicts)
GOLDEN: dict[str, tuple[int, int, list[int]]] = {
    "corridor": (44, 70, [120, 50, 50]),  # 5 junctions: 2+1 lanes per direction
    "emergency": (44, 70, [120, 50, 50]),  # the corridor network
    "rush_hour": (44, 70, [120, 50, 50]),  # the corridor network
    "grid_3x3": (96, 144, [324, 72, 72]),  # 9 junctions x the single-junction counts
    "grid_4x4": (160, 256, [576, 128, 128]),
    "single_intersection": (16, 16, [36, 8, 8]),
}


def _array_hash(net: CompiledNetwork) -> str:
    digest = hashlib.sha256()
    for name, arr in net.arrays().items():
        digest.update(name.encode())
        digest.update(str(arr.dtype).encode() + str(arr.shape).encode())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def test_every_bundled_scenario_has_a_golden() -> None:
    assert sorted(GOLDEN) == sorted(BUNDLED_SCENARIOS)


@pytest.mark.parametrize("name", sorted(BUNDLED_SCENARIOS))
def test_bundled_scenario_compiles(name: str) -> None:
    scenario = urbanflow.Scenario.load(urbanflow.bundled(name))
    net = compile_network(scenario)
    lanes, connectors, conflicts = GOLDEN[name]
    assert (net.n_lanes, net.n_conn) == (lanes, connectors)
    assert np.bincount(net.conf_kind, minlength=3).tolist() == conflicts
    assert net.report.warnings == ()
    geometry = render_geometry(net).model_dump(mode="json")
    jsonschema.Draft202012Validator(RenderGeometry.model_json_schema()).validate(geometry)
    again = compile_network(urbanflow.Scenario.load(urbanflow.bundled(name)))
    assert _array_hash(again) == _array_hash(net)
    assert render_geometry(again).model_dump(mode="json") == geometry
