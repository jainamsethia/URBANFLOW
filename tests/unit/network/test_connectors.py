"""Connector geometry and curvature speed limits (plan E.5 rule 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urbanflow.core.constants import TURN_LATERAL_ACCEL
from urbanflow.network.connectors import build_connector, connector_speed_limit, direction
from urbanflow.network.lanes import DegenerateGeometryError

EAST, NORTH = direction(0.0), direction(math.pi / 2)


def test_direction() -> None:
    np.testing.assert_allclose(direction(math.pi), [-1.0, 0.0], atol=1e-12)


def test_explicit_shape_is_snapped_to_the_lane_ends() -> None:
    line = build_connector((0.5, 0.0), EAST, (10.0, 10.0), NORTH, [(0, 0), (5, 5), (9, 9)])
    np.testing.assert_allclose(line.points, [[0.5, 0.0], [5.0, 5.0], [10.0, 10.0]])


def test_snapping_drops_repeated_points() -> None:
    line = build_connector((0.0, 0.0), EAST, (10.0, 0.0), EAST, [(1, 1), (0, 0), (10, 0)])
    np.testing.assert_allclose(line.points, [[0.0, 0.0], [10.0, 0.0]])
    with pytest.raises(DegenerateGeometryError, match="zero length"):
        build_connector((0.0, 0.0), EAST, (0.0, 0.0), EAST, [(1, 1), (2, 2)])


def test_bezier_endpoints_and_tangents() -> None:
    line = build_connector((0.0, 0.0), EAST, (10.0, 10.0), NORTH)
    np.testing.assert_allclose(line.points[[0, -1]], [[0.0, 0.0], [10.0, 10.0]], atol=1e-12)
    # sampled chords deviate from the exact end tangents by less than one sample angle
    assert abs(line.headings[0]) < math.pi / 16
    assert abs(line.headings[-1] - math.pi / 2) < math.pi / 16
    assert len(line.points) == 9  # ceil((π/2)/(π/16)) + 1


def test_zero_chord_is_degenerate() -> None:
    with pytest.raises(DegenerateGeometryError, match="give the intersection a radius"):
        build_connector((1.0, 1.0), EAST, (1.0, 1.0), EAST)


def test_speed_limit_follows_curvature() -> None:
    radius = 10.0
    line = build_connector((0.0, 0.0), EAST, (radius, radius), NORTH)
    expected = math.sqrt(TURN_LATERAL_ACCEL * radius)
    assert connector_speed_limit(line.points, 13.89, 13.89) == pytest.approx(expected, rel=0.03)
    assert connector_speed_limit(line.points, 3.0, 13.89) == 3.0  # never above the lanes


def test_straight_connector_keeps_the_lane_limits() -> None:
    line = build_connector((0.0, 0.0), EAST, (16.8, 0.0), EAST)
    assert connector_speed_limit(line.points, 13.89, 11.11) == 11.11
    assert connector_speed_limit([(0.0, 0.0), (1.0, 0.0)], 9.0, 12.0) == 9.0
