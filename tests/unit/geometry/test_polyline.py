"""Polylines: arc length, interpolation, cuts, miter offsets, curvature (plan E.5)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from urbanflow.core.types import DriveSide
from urbanflow.geometry.intersect import point_polyline_distance
from urbanflow.geometry.polyline import Polyline, curvature_radii, offset_polyline

ZIGZAG = [[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]]  # segment lengths 5 and 6


def test_lengths_and_cumulative() -> None:
    p = Polyline(ZIGZAG)
    assert p.length == 11.0
    assert p.cumulative.tolist() == [0.0, 5.0, 11.0]
    assert p.headings.tolist() == pytest.approx([math.atan2(4, 3), math.pi / 2])


def test_point_at_interpolates_and_clamps() -> None:
    p = Polyline(ZIGZAG)
    got = p.point_at([-1.0, 0.0, 2.5, 5.0, 8.0, 11.0, 99.0])
    np.testing.assert_allclose(got, [[0, 0], [0, 0], [1.5, 2], [3, 4], [3, 7], [3, 10], [3, 10]])
    assert p.point_at(2.5).shape == (2,)
    assert p.point_at(np.zeros((2, 3))).shape == (2, 3, 2)


def test_an_underflowing_segment_stays_finite() -> None:
    """A segment too short to change the float arc length maps to its start point."""
    line = Polyline([(4.0, 1.0), (0.0, 3e-139), (0.0, 0.0)])
    assert line.cumulative[1] == line.cumulative[2]
    pts = line.point_at([line.length, line.length / 2])
    assert np.isfinite(pts).all()
    assert np.isfinite(point_polyline_distance([(1.0, 1.0)], line.points)).all()


def test_heading_at_vertex_belongs_to_next_segment() -> None:
    p = Polyline(ZIGZAG)
    h0, h1 = math.atan2(4, 3), math.pi / 2
    assert p.heading_at([-5.0, 4.99, 5.0, 11.0, 50.0]).tolist() == pytest.approx(
        [h0, h0, h1, h1, h1]
    )


def test_cut_along_arc_length() -> None:
    p = Polyline(ZIGZAG)
    cut = p.cut(2.5, 8.0)
    np.testing.assert_allclose(cut.points, [[1.5, 2.0], [3.0, 4.0], [3.0, 7.0]])
    assert cut.length == pytest.approx(5.5)
    # A cut exactly at a vertex does not duplicate it; bounds are clamped.
    np.testing.assert_allclose(p.cut(5.0, 50.0).points, [[3.0, 4.0], [3.0, 10.0]])
    with pytest.raises(ValueError, match="empty"):
        p.cut(4.0, 4.0)


def test_arrays_are_read_only() -> None:
    p = Polyline(ZIGZAG)
    with pytest.raises(ValueError, match="read-only"):
        p.points[0, 0] = 1.0


@pytest.mark.parametrize(
    ("points", "match"),
    [
        ([[0.0, 0.0]], "n >= 2"),
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "n >= 2"),
        ([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], "point 1 repeats"),
        ([[0.0, 0.0], [math.nan, 1.0]], "finite"),
    ],
)
def test_invalid_points(points: list[list[float]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Polyline(points)


def _naive_point_at(pts: list[tuple[float, float]], s: float) -> tuple[float, float]:
    total = sum(math.dist(a, b) for a, b in itertools.pairwise(pts))
    s = min(max(s, 0.0), total)
    for a, b in itertools.pairwise(pts):
        seg = math.dist(a, b)
        if s <= seg:
            t = s / seg
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        s -= seg
    return pts[-1]


coords = st.tuples(st.integers(-100, 100), st.integers(-100, 100))
polylines = st.lists(coords, min_size=2, max_size=8).filter(
    lambda ps: all(a != b for a, b in itertools.pairwise(ps))
)


@settings(max_examples=200, deadline=None)
@given(points=polylines, frac=st.floats(-0.2, 1.2))
def test_point_at_matches_naive_walk(points: list[tuple[int, int]], frac: float) -> None:
    pts = [(float(x), float(y)) for x, y in points]
    line = Polyline(pts)
    s = frac * line.length
    assert line.point_at(s).tolist() == pytest.approx(_naive_point_at(pts, s), abs=1e-9)


# ------------------------------------------------------------------ offsets

L_BEND = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]  # 90° left turn


def _line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    return abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / math.hypot(*d)


def test_straight_offsets_both_sides() -> None:
    line = [[0.0, 0.0], [10.0, 0.0]]
    assert offset_polyline(line, 2.0).tolist() == [[0.0, -2.0], [10.0, -2.0]]
    assert offset_polyline(line, 2.0, side=DriveSide.left).tolist() == [[0.0, 2.0], [10.0, 2.0]]
    assert offset_polyline(line, -2.0).tolist() == [[0.0, 2.0], [10.0, 2.0]]
    assert offset_polyline(line, 0.0).tolist() == line


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (DriveSide.right, [[0, -1.5], [11.5, -1.5], [11.5, 10]]),  # outside of the bend
        (DriveSide.left, [[0, 1.5], [8.5, 1.5], [8.5, 10]]),  # inside of the bend
    ],
)
def test_miter_on_90_degree_bend(side: DriveSide, expected: list[list[float]]) -> None:
    out = offset_polyline(L_BEND, 1.5, side=side)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    # Every offset segment lies exactly 1.5 m from its source segment's line.
    src = np.array(L_BEND)
    for k in range(2):
        for p in out[k : k + 2]:
            assert _line_distance(p, src[k], src[k + 1]) == pytest.approx(1.5)


def test_lane_width_preserved_through_bend() -> None:
    w = 3.2
    lane0 = offset_polyline(L_BEND, w / 2)
    lane1 = offset_polyline(L_BEND, 3 * w / 2)
    mids = (lane1[:-1] + lane1[1:]) / 2
    assert point_polyline_distance(mids, lane0).tolist() == pytest.approx([w, w])


def test_bevel_fallback_on_sharp_bend() -> None:
    sharp = [[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]]  # ~174° turn: miter factor ~20
    out = offset_polyline(sharp, 1.0)
    assert len(out) == 4
    n_out = np.array([1.0, 10.0]) / math.hypot(1.0, 10.0)  # right normal of (-10, 1)
    np.testing.assert_allclose(out[1], [10.0, -1.0])
    np.testing.assert_allclose(out[2], np.array([10.0, 0.0]) + n_out)
    assert len(offset_polyline(sharp, 1.0, miter_limit=100.0)) == 3


def test_curvature_radii() -> None:
    angles = np.linspace(0.0, math.pi, 7)
    circle = np.column_stack((5 * np.cos(angles), 5 * np.sin(angles)))
    assert curvature_radii(circle).tolist() == pytest.approx([5.0] * 5)
    assert curvature_radii([[0, 0], [1, 0], [2, 0]]).tolist() == [math.inf]
    assert curvature_radii([[0, 0], [1, 0]]).shape == (0,)
