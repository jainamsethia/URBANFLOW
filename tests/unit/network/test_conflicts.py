"""Conflict zones between connectors (plan E.5 rule 5)."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

import urbanflow
from urbanflow.core.constants import CONFLICT_WIDTH, LOW_ANGLE_CROSSING
from urbanflow.geometry import Polyline, first_crossing
from urbanflow.network import CompiledNetwork, ConflictKind, compile_network
from urbanflow.network.conflicts import compute_conflicts, crossing_half_length
from urbanflow.scenario import Scenario

HALF = CONFLICT_WIDTH / 2


def _table(lines: list[list[tuple[float, float]]], from_lane: list[int], to_lane: list[int]) -> Any:
    n = len(lines)
    return compute_conflicts(
        [Polyline(p) for p in lines],
        np.zeros(n, dtype=np.int32),
        np.array(from_lane, dtype=np.int32),
        np.array(to_lane, dtype=np.int32),
        100,
    )


def _conflict(net: CompiledNetwork, a: str, b: str) -> int:
    pair = {net.link_index[a], net.link_index[b]}
    return next(c for c in range(net.n_conflicts) if {net.conf_a[c], net.conf_b[c]} == pair)


def test_four_way_junction_counts() -> None:
    # 2 lanes per arm, L{0} S{0,1} R{1}: 4 connectors per approach. Diverging: L/S0 and S1/R
    # per approach (8). Merging: two per exit (8). Crossing: 16 through x through
    # (perpendicular approaches, 2 x 2 lanes), 16 left x opposing/perpendicular through,
    # 4 adjacent left x left; right turns cross nothing (36).
    net = compile_network(urbanflow.Scenario.load(urbanflow.bundled("single_intersection")))
    counts = np.bincount(net.conf_kind, minlength=3).tolist()
    assert counts == [36, 8, 8]
    assert (net.conf_a < net.conf_b).all()
    assert (net.link_intersection[net.conf_a] == net.link_intersection[net.conf_b]).all()


def test_opposite_straights_do_not_conflict(demo_net: CompiledNetwork) -> None:
    ns = set(demo_net.mov_conn[demo_net.mov_conn_ptr[4] : demo_net.mov_conn_ptr[5]].tolist())
    sn = set(demo_net.mov_conn[demo_net.mov_conn_ptr[7] : demo_net.mov_conn_ptr[8]].tolist())
    assert demo_net.mov_ids[4] == "N_in->S_out" and demo_net.mov_ids[7] == "S_in->N_out"
    pairs = set(zip(demo_net.conf_a.tolist(), demo_net.conf_b.tolist(), strict=True))
    assert not any((a, b) in pairs or (b, a) in pairs for a in ns for b in sn)


def test_crossing_half_length_uses_the_acute_angle() -> None:
    assert crossing_half_length(math.pi / 2) == pytest.approx(HALF)
    for deg in (20, 47, 70):
        theta = math.radians(deg)
        assert crossing_half_length(theta) == pytest.approx(crossing_half_length(math.pi - theta))
    # permissive left against the opposing straight: θ ≈ 133° gives a 6.9 m zone, not 1.3 m
    assert 2 * crossing_half_length(math.radians(133)) == pytest.approx(6.9, abs=0.05)
    sizes = [crossing_half_length(math.radians(d)) for d in range(15, 91, 5)]
    assert sizes == sorted(sizes, reverse=True)


def test_perpendicular_crossing_zone() -> None:
    table = _table([[(0, 0), (20, 0)], [(8, -10), (8, 10)]], [0, 1], [2, 3])
    assert table.conf_kind.tolist() == [ConflictKind.crossing]
    np.testing.assert_allclose(table.conf_zone_a, [[8 - HALF, 8 + HALF]])
    np.testing.assert_allclose(table.conf_zone_b, [[10 - HALF, 10 + HALF]])
    assert (table.conf_a.tolist(), table.conf_b.tolist()) == ([100], [101])


def test_zones_are_clipped_to_the_connector() -> None:
    table = _table([[(0, 0), (20, 0)], [(0.5, -10), (0.5, 10)]], [0, 1], [2, 3])
    assert table.conf_zone_a[0].tolist() == pytest.approx([0.0, 0.5 + HALF])


def test_obtuse_permissive_left_zone(demo_net: CompiledNetwork) -> None:
    left, straight = "N_in_0->E_out_0", "S_in_0->N_out_0"
    c = _conflict(demo_net, left, straight)
    crossing = first_crossing(
        demo_net.link_points(demo_net.link_index[left]),
        demo_net.link_points(demo_net.link_index[straight]),
    )
    assert crossing is not None
    theta = crossing[2]
    assert math.degrees(theta) == pytest.approx(130, abs=4)  # the Bezier approximates 133°
    for zone in (demo_net.conf_zone_a[c], demo_net.conf_zone_b[c]):
        assert zone[1] - zone[0] == pytest.approx(2 * crossing_half_length(theta))
        assert zone[1] - zone[0] > 6.0


def test_low_angle_crossing_uses_the_separation_sampler() -> None:
    theta = math.radians(10)
    assert theta < LOW_ANGLE_CROSSING
    d = (60 * math.cos(theta), 60 * math.sin(theta))
    table = _table([[(-60, 0), (60, 0)], [(-d[0], -d[1]), d]], [0, 1], [2, 3])
    reach = CONFLICT_WIDTH / math.sin(theta)  # where the other line is W_max away
    np.testing.assert_allclose(table.conf_zone_a[0], [60 - reach, 60 + reach], atol=0.3)
    np.testing.assert_allclose(table.conf_zone_b[0], [60 - reach, 60 + reach], atol=0.3)


def test_diverging_and_merging_zones() -> None:
    fork = [[(0, 0), (30, 0)], [(0, 0), (30, 30)]]
    div = _table(fork, [5, 5], [6, 7])
    assert div.conf_kind.tolist() == [ConflictKind.diverging]
    z = div.conf_zone_a[0]
    assert z[0] == 0 and z[1] == pytest.approx(CONFLICT_WIDTH / math.sin(math.pi / 4), abs=0.3)
    join = [[(0, 0), (30, 0)], [(0, 30), (30, 0)]]
    merge = _table(join, [5, 6], [7, 7])
    assert merge.conf_kind.tolist() == [ConflictKind.merging]
    assert merge.conf_zone_a[0][1] == 30.0 and merge.conf_zone_b[0][1] == pytest.approx(
        30 * math.sqrt(2)
    )


def _zones_within_connectors(lines: list[list[tuple[float, float]]], table: Any) -> None:
    lengths = [Polyline(p).length for p in lines]
    for a, b, za, zb in zip(
        table.conf_a, table.conf_b, table.conf_zone_a, table.conf_zone_b, strict=True
    ):
        for link, (z_in, z_out) in ((a, za), (b, zb)):
            assert 0.0 <= z_in <= z_out <= lengths[link - 100], (link, z_in, z_out)


POINT = st.tuples(st.floats(-20, 20), st.floats(-20, 20))


@settings(max_examples=150, deadline=None)
@given(
    head=st.lists(POINT, min_size=1, max_size=4),
    other=st.lists(POINT, min_size=1, max_size=4),
    end=POINT,
    kind=st.sampled_from(["merge", "diverge", "free"]),
)
@example(  # a 3e-139 m segment: its arc length underflows, point_at divided by zero
    head=[(1.0, 0.0)],
    other=[(4.0, 1.0), (0.0, 3.1762528409741592e-139)],
    end=(0.0, 0.0),
    kind="free",
)
@example(  # merge where d_sep spans the whole connector: L - d_sep was -1.8e-15
    head=[(1.4, 3.0), (1.1, -1.9), (1.8, 1.2)],
    other=[(1.7, 1.1)],
    end=(2.1, -3.2),
    kind="merge",
)
def test_every_zone_lies_on_its_connector(
    head: list[tuple[float, float]],
    other: list[tuple[float, float]],
    end: tuple[float, float],
    kind: str,
) -> None:
    """0 <= z_in <= z_out <= L for every zone, whatever the conflict kind (clipped)."""
    if kind == "merge":  # common end point
        lines = [[*head, end], [*other, end]]
    elif kind == "diverge":  # common start point
        lines = [[end, *head], [end, *other]]
    else:
        lines = [[*head, end], [*other, (-end[0], -end[1])]]
    if any(len(set(p)) < len(p) or Polyline(p).length <= 1e-6 for p in lines):
        return  # degenerate connectors are rejected earlier (E905)
    table = _table(lines, [0, 0 if kind == "diverge" else 1], [2, 2 if kind == "merge" else 3])
    _zones_within_connectors(lines, table)


def test_double_crossing_merges_the_zones() -> None:
    table = _table([[(0, 0), (20, 0)], [(2, -5), (10, 5), (18, -5)]], [0, 1], [2, 3])
    assert table.crossed_twice == ((100, 101),)
    lo, hi = table.conf_zone_a[0]
    assert lo < 6 < 14 < hi  # covers both crossings (x = 6 and x = 14)


def test_conn_conf_csr_is_sorted_by_z_in(demo_net: CompiledNetwork) -> None:
    net = demo_net
    seen = np.zeros(net.n_conflicts, dtype=int)
    for c in range(net.n_conn):
        link = net.n_lanes + c
        rows = net.conn_conf[net.conn_conf_ptr[c] : net.conn_conf_ptr[c + 1]]
        sides = net.conn_conf_side[net.conn_conf_ptr[c] : net.conn_conf_ptr[c + 1]]
        z_in = [
            (net.conf_zone_a if s == 0 else net.conf_zone_b)[k][0]
            for k, s in zip(rows, sides, strict=True)
        ]
        assert z_in == sorted(z_in)
        for k, s in zip(rows.tolist(), sides.tolist(), strict=True):
            assert (net.conf_a if s == 0 else net.conf_b)[k] == link
            seen[k] += 1
    assert (seen == 2).all()


def test_no_connectors_gives_an_empty_table() -> None:
    table = compute_conflicts(
        [], np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32), 7
    )
    assert len(table) == 0 and table.conf_zone_a.shape == (0, 2)
    assert table.conn_conf_ptr.tolist() == [0]


def test_w304_from_the_compiler(demo_data: dict[str, Any], demo: Scenario) -> None:
    movements = [
        m.model_dump(mode="json", exclude_none=True)
        for m in demo.resolved.network.intersections[0].movements or ()
    ]
    for m in movements:
        if m["id"] == "N_in->S_out":  # bulge east across the opposing straight, and back
            m["connections"][0]["shape"] = [[-1.6, 8.4], [4.0, 0.0], [-1.6, -8.4]]
    data = copy.deepcopy(demo_data)
    data["network"]["intersections"][0]["movements"] = movements
    net = compile_network(Scenario.from_dict(data))
    w304 = [w for w in net.report.warnings if w.code == "W304"]
    assert [(w.path, w.message) for w in w304] == [
        (
            "network.intersections[0]",
            'connectors "N_in_0->S_out_0" and "S_in_0->N_out_0" cross twice; '
            "their conflict zones were merged",
        )
    ]
