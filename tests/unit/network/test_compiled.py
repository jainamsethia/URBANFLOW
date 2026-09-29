"""CompiledNetwork: vectorised point_at, read-only arrays, CSR consistency (plan E.3)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from urbanflow.core.constants import LINK_BASE_GAP
from urbanflow.core.errors import NotFoundError
from urbanflow.core.types import LinkKind
from urbanflow.geometry import Polyline
from urbanflow.network import CompiledNetwork


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_point_at_matches_polyline(demo_net: CompiledNetwork, data: st.DataObject) -> None:
    net = demo_net
    links = data.draw(st.lists(st.integers(0, net.n_links - 1), min_size=1, max_size=16))
    pos = data.draw(
        st.lists(st.floats(-5.0, 250.0, allow_nan=False), min_size=len(links), max_size=len(links))
    )
    xy, heading = net.point_at(links, pos)
    assert xy.shape == (len(links), 2) and heading.shape == (len(links),)
    for link, p, got, h in zip(links, pos, xy, heading, strict=True):
        line = Polyline(net.link_points(link))
        np.testing.assert_allclose(got, line.point_at(p), atol=1e-9)
        s = min(max(p, 0.0), line.length)
        if np.abs(line.cumulative[1:-1] - s).min(initial=math.inf) > 1e-6:  # not at a vertex
            assert h == pytest.approx(float(line.heading_at(p)))


def test_point_at_link_ends(demo_net: CompiledNetwork) -> None:
    net = demo_net
    links = np.arange(net.n_links)
    start, _ = net.point_at(links, np.zeros(net.n_links))
    end, _ = net.point_at(links, net.link_length + 1.0)
    np.testing.assert_allclose(start, net.pts[net.link_pt_start[:-1]])
    np.testing.assert_allclose(end, net.pts[net.link_pt_start[1:] - 1], atol=1e-9)


def test_arrays_are_read_only(demo_net: CompiledNetwork) -> None:
    arrays = demo_net.arrays()
    assert {"pts", "s_global", "conf_zone_a", "road_capacity_vph", "bbox"} <= set(arrays)
    assert demo_net.read_only()
    for name, arr in arrays.items():
        assert not arr.flags.writeable, name
        if arr.size:
            with pytest.raises(ValueError, match="read-only"):
                arr.flat[0] = arr.flat[0]
    with pytest.raises(TypeError):
        demo_net.road_index["x"] = 0  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        demo_net.pts = np.zeros((1, 2))  # type: ignore[misc]


def test_sizes_and_link_layout(demo_net: CompiledNetwork) -> None:
    net = demo_net
    assert repr(net) == (
        "CompiledNetwork(roads=8, lanes=16, connectors=16, movements=12, intersections=5, "
        f"conflicts=52, hash={net.scenario_hash[:12]})"
    )
    assert (net.n_roads, net.n_lanes, net.n_conn, net.n_links) == (8, 16, 16, 32)
    assert (net.n_movements, net.n_intersections) == (12, 5)
    assert net.link_kind.tolist() == [LinkKind.lane.code] * 16 + [LinkKind.connector.code] * 16
    assert (net.link_road[16:] == -1).all() and (net.link_lane_index[16:] == -1).all()
    assert (net.link_movement[:16] == -1).all() and (net.link_movement[16:] >= 0).all()
    np.testing.assert_array_equal(net.road_lane_start, np.arange(0, 16, 2))


def test_global_arc_key(demo_net: CompiledNetwork) -> None:
    net = demo_net
    counts = np.diff(net.link_pt_start)
    assert (counts >= 2).all() and net.link_pt_start[-1] == len(net.pts) == len(net.s_global)
    assert (np.diff(net.s_global) > 0).all()  # globally monotone
    np.testing.assert_allclose(np.diff(net.link_base), net.link_length[:-1] + LINK_BASE_GAP)
    np.testing.assert_allclose(net.s_global[net.link_pt_start[:-1]], net.link_base)
    np.testing.assert_allclose(
        net.s_global[net.link_pt_start[1:] - 1], net.link_base + net.link_length
    )
    for link in range(net.n_links):
        line = Polyline(net.link_points(link))
        heads = net.seg_heading[net.link_pt_start[link] : net.link_pt_start[link + 1]]
        np.testing.assert_allclose(heads, [*line.headings, line.headings[-1]])
        assert net.link_length[link] == pytest.approx(line.length)


def _rows(ptr: np.ndarray, values: np.ndarray, i: int) -> list[int]:
    return values[ptr[i] : ptr[i + 1]].tolist()


def test_csr_consistency(demo_net: CompiledNetwork) -> None:
    net = demo_net
    out = [_rows(net.lane_out_ptr, net.lane_out_conn, lane) for lane in range(net.n_lanes)]
    assert sorted(c for row in out for c in row) == list(range(net.n_lanes, net.n_links))
    for lane, row in enumerate(out):
        assert row == sorted(row)
        assert all(net.conn_from_lane[c - net.n_lanes] == lane for c in row)
    movs = [_rows(net.mov_conn_ptr, net.mov_conn, m) for m in range(net.n_movements)]
    assert sorted(c for row in movs for c in row) == list(range(net.n_lanes, net.n_links))
    assert all(net.link_movement[c] == m for m, row in enumerate(movs) for c in row)
    for j in range(net.n_intersections):
        ins = _rows(net.int_in_ptr, net.int_in_lanes, j)
        outs = _rows(net.int_out_ptr, net.int_out_lanes, j)
        lanes = np.arange(net.n_lanes)
        assert sorted(ins) == lanes[net.road_to[net.link_road[lanes]] == j].tolist()
        assert sorted(outs) == lanes[net.road_from[net.link_road[lanes]] == j].tolist()
    assert net.lane_left[:2].tolist() == [-1, 0] and net.lane_right[:2].tolist() == [1, -1]


def test_lane_of(demo_net: CompiledNetwork) -> None:
    assert demo_net.lane_of("N_in", 1) == 1
    assert demo_net.lane_of(2, 0) == 4
    assert demo_net.link_ids[demo_net.lane_of("W_out", 1)] == "W_out_1"
    with pytest.raises(NotFoundError, match='unknown road "N_ni" \\(did you mean "N_in"\\?\\)'):
        demo_net.lane_of("N_ni", 0)
    with pytest.raises(NotFoundError, match="has 2 lanes"):
        demo_net.lane_of("N_in", 2)
    with pytest.raises(NotFoundError, match="out of range"):
        demo_net.lane_of(8, 0)
