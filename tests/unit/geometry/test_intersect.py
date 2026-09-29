"""Crossings, separation and bounding boxes (plan E.5 rule 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urbanflow.geometry.curves import bezier_connector
from urbanflow.geometry.intersect import (
    bbox_overlap,
    bounding_box,
    first_crossing,
    point_polyline_distance,
    segment_intersections,
    separation_distance,
)

A = [[0.0, 0.0], [10.0, 0.0]]


def test_crossing() -> None:
    assert first_crossing(A, [[4.0, -5.0], [4.0, 5.0]]) == pytest.approx((4.0, 5.0, math.pi / 2))


def test_obtuse_angle_between_directions() -> None:
    s_a, s_b, angle = first_crossing(A, [[6.0, -1.0], [4.0, 1.0]]) or (0, 0, 0)
    assert (s_a, s_b) == pytest.approx((5.0, math.sqrt(2)))
    assert angle == pytest.approx(3 * math.pi / 4)


def test_t_junction() -> None:
    assert first_crossing(A, [[3.0, 5.0], [3.0, 0.0]]) == pytest.approx((3.0, 5.0, math.pi / 2))


def test_touching_endpoints_count() -> None:
    assert first_crossing(A, [[10.0, 0.0], [10.0, 5.0]]) == pytest.approx((10.0, 0.0, math.pi / 2))


@pytest.mark.parametrize(
    "b",
    [
        [[0.0, 1.0], [10.0, 1.0]],  # parallel
        [[2.0, 0.0], [12.0, 0.0]],  # collinear overlap
        [[11.0, -1.0], [11.0, 1.0]],  # misses beyond a's end
    ],
    ids=["parallel", "collinear", "disjoint"],
)
def test_no_crossing(b: list[list[float]]) -> None:
    assert first_crossing(A, b) is None
    assert segment_intersections(A, b).shape == (0, 3)


def test_multi_segment_and_double_crossing() -> None:
    a = [[0.0, 0.0], [4.0, 0.0], [4.0, 8.0]]
    b = [[-1.0, 6.0], [6.0, 6.0], [6.0, -1.0], [2.0, -1.0], [2.0, 1.0]]
    rows = segment_intersections(a, b)
    assert rows[:, 0].tolist() == pytest.approx([2.0, 10.0])  # sorted along a
    assert first_crossing(a, b) == pytest.approx((2.0, 7 + 7 + 4 + 1, math.pi / 2))


def test_crossing_at_shared_vertex_reported_once() -> None:
    rows = segment_intersections([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], [[5.0, -5.0], [5.0, 5.0]])
    np.testing.assert_allclose(rows, [[5.0, 5.0, math.pi / 2]])


def test_point_polyline_distance() -> None:
    line = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    got = point_polyline_distance([[5.0, 3.0], [-3.0, -4.0], [12.0, 5.0], [11.0, -1.0]], line)
    assert got.tolist() == pytest.approx([3.0, 5.0, 2.0, math.sqrt(2)])


@pytest.mark.parametrize("alpha_deg", [20.0, 30.0, 60.0])
def test_separation_of_diverging_lines(alpha_deg: float) -> None:
    alpha = math.radians(alpha_deg)
    a = [[0.0, 0.0], [40.0, 0.0]]
    b = [[0.0, 0.0], [40.0 * math.cos(alpha), 40.0 * math.sin(alpha)]]
    assert separation_distance(a, b, 3.0) == pytest.approx(3.0 / math.sin(alpha))


def test_separation_from_end_for_merging() -> None:
    a = [[-40.0, 0.0], [0.0, 0.0]]
    b = [[-40.0 * math.cos(0.5), -40.0 * math.sin(0.5)], [0.0, 0.0]]
    assert separation_distance(a, b, 3.0, from_end=True) == pytest.approx(3.0 / math.sin(0.5))


def test_separation_is_capped_and_zero_when_apart() -> None:
    assert separation_distance(A, [[0.0, 1.0], [10.0, 1.0]], 3.0) == 10.0
    assert separation_distance(A, [[0.0, 5.0], [10.0, 5.0]], 3.0) == 0.0


def test_separation_of_turning_connector_matches_circle() -> None:
    radius, width = 10.0, 3.0
    left = bezier_connector((0.0, 0.0), (1.0, 0.0), (radius, radius), (0.0, 1.0))
    straight = [[0.0, 0.0], [20.0, 0.0]]
    exact = radius * math.acos(1 - width / radius)  # arc length where R(1 - cos) = width
    assert separation_distance(left, straight, width) == pytest.approx(exact, rel=0.01)


def test_bounding_boxes() -> None:
    assert bounding_box([[1.0, 5.0], [-2.0, 3.0], [4.0, 4.0]]).tolist() == [-2.0, 3.0, 4.0, 5.0]
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [5, 5, 6, 6]], dtype=float)
    assert bbox_overlap(boxes, boxes).tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]
    assert bbox_overlap([0, 0, 1, 1], boxes).shape == (1, 3)
