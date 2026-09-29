"""Render geometry export (plan E.5 rule 7, K.10): contents, order and schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from pydantic import ValidationError

from urbanflow.core.types import IntersectionKind, TurnKind
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.scenario import Scenario
from urbanflow.visualization import RenderGeometry, render_geometry
from urbanflow.visualization.geometry import GEOMETRY_VERSION, RenderLinks, convex_hull

DEMO = Path(__file__).resolve().parents[2] / "fixtures" / "scenarios" / "demo.json"


@pytest.fixture(scope="module")
def net() -> CompiledNetwork:
    return compile_network(Scenario.load(DEMO))


@pytest.fixture(scope="module")
def geo(net: CompiledNetwork) -> RenderGeometry:
    return render_geometry(net)


def _area(poly: list[tuple[float, float]]) -> float:
    x, y = np.array(poly).T
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def test_matches_its_json_schema(geo: RenderGeometry) -> None:
    data = geo.model_dump(mode="json")
    jsonschema.Draft202012Validator(RenderGeometry.model_json_schema()).validate(data)
    assert RenderGeometry.model_validate_json(json.dumps(data)) == geo


def test_header(geo: RenderGeometry, net: CompiledNetwork) -> None:
    assert geo.version == GEOMETRY_VERSION == 1
    assert geo.geometry_crc == net.geometry_crc
    assert geo.origin == (-200.0, -200.0) and geo.bbox == (0.0, 0.0, 400.0, 400.0)


def test_links_in_link_index_order(geo: RenderGeometry, net: CompiledNetwork) -> None:
    links = geo.links
    assert links.id == list(net.link_ids)
    assert links.kind == ["lane"] * net.n_lanes + ["connector"] * net.n_conn
    assert links.owner[0] == "N_in" and links.owner[net.n_lanes] == net.mov_ids[0]
    assert links.lane_width == pytest.approx(net.link_width.tolist())
    for k in (0, net.n_lanes, net.n_links - 1):
        np.testing.assert_allclose(links.path[k], net.link_points(k), atol=5e-4)


def test_movements(geo: RenderGeometry, net: CompiledNetwork) -> None:
    movs = geo.movements
    assert movs.id == list(net.mov_ids) and set(movs.intersection) == {"J"}
    assert (movs.from_road[0], movs.to_road[0], movs.turn[0]) == ("E_in", "N_out", TurnKind.right)
    flat = [c for links in movs.links for c in links]
    assert flat == net.mov_conn.tolist()


def test_signal_heads_sit_at_the_stop_line(geo: RenderGeometry, net: CompiledNetwork) -> None:
    heads = geo.signal_heads
    assert heads.movement == list(range(net.n_movements))
    positions = np.array(heads.position)
    assert len({tuple(p) for p in heads.position}) == net.n_movements  # shared lanes spread
    ends = net.pts[net.link_pt_start[1 : net.n_lanes + 1] - 1]
    gaps = np.linalg.norm(positions[:, None, :] - ends[None, :, :], axis=2).min(axis=1)
    assert (gaps <= net.link_width.max() / 2 + 1e-3).all()


def test_road_surfaces_cover_the_lanes(geo: RenderGeometry, net: CompiledNetwork) -> None:
    assert geo.roads.id == list(net.road_ids)
    for surface, length in zip(geo.roads.surfaces, net.road_length, strict=True):
        assert _area(surface) == pytest.approx(length * 6.4, rel=1e-4)  # 2 lanes x 3.2 m


def test_lane_markings(geo: RenderGeometry, net: CompiledNetwork) -> None:
    marks = geo.lane_markings
    assert marks.road == [r for r in range(net.n_roads) for _ in range(3)]
    assert marks.dashed == [False, True, False] * net.n_roads  # median, separator, curb
    median, separator, curb = (np.array(p) for p in marks.path[:3])  # N_in, southbound
    np.testing.assert_allclose([median[0, 0], separator[0, 0], curb[0, 0]], [200, 196.8, 193.6])


def test_stop_lines(geo: RenderGeometry, net: CompiledNetwork) -> None:
    incoming = [net.lane_of(r, k) for r in ("N_in", "S_in", "E_in", "W_in") for k in (0, 1)]
    assert geo.stop_lines.link == sorted(incoming)
    for link, path in zip(geo.stop_lines.link, geo.stop_lines.path, strict=True):
        a, b = np.array(path)
        assert np.linalg.norm(b - a) == pytest.approx(3.2)
        np.testing.assert_allclose((a + b) / 2, net.link_points(link)[-1], atol=1e-3)


def test_intersection_polygons(geo: RenderGeometry, net: CompiledNetwork) -> None:
    ints = geo.intersections
    assert ints.id == list(net.int_ids) and ints.radius == pytest.approx([8.4, 0, 0, 0, 0])
    assert ints.kind == [IntersectionKind.signalized] + [IntersectionKind.boundary] * 4
    assert ints.polygons[1:] == [[], [], [], []]
    poly = np.array(ints.polygons[0])
    stops = np.array([p for path in geo.stop_lines.path for p in path])
    hull = convex_hull(np.concatenate((poly, stops)))
    assert len(hull) == len(poly)  # every stop-line end lies inside or on the outline
    assert _area(ints.polygons[0]) > np.pi * 8.4**2 * 0.9


def test_vehicle_types(geo: RenderGeometry, net: CompiledNetwork) -> None:
    assert [t.id for t in geo.vehicle_types] == [t.id for t in net.vehicle_types]
    bus = next(t for t in geo.vehicle_types if t.id == "city_bus")
    assert (bus.vclass, bus.length, bus.width, bus.color) == ("bus", 12.0, 2.55, None)


def test_columns_must_have_equal_lengths() -> None:
    with pytest.raises(ValidationError, match="equal lengths"):
        RenderLinks(id=["a"], kind=[], owner=[], lane_width=[], path=[])


def test_convex_hull() -> None:
    square = np.array([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 0)], dtype=float)
    np.testing.assert_allclose(convex_hull(square), [(0, 0), (2, 0), (2, 2), (0, 2)])
    assert convex_hull(np.array([(1.0, 1.0), (1.0, 1.0)])).shape == (1, 2)
