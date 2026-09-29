"""Bézier connectors approximate circular arcs (plan E.5 rule 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urbanflow.geometry.curves import (
    bezier_connector,
    circular_handle,
    connector_samples,
    turn_angle,
)
from urbanflow.geometry.polyline import Polyline, curvature_radii


def test_endpoints_exact_and_tangents_match_directions() -> None:
    start, end = np.array([2.0, -1.0]), np.array([9.0, 6.0])
    ds, de = np.array([2.0, 0.0]), np.array([0.0, 5.0])
    coarse = bezier_connector(start, ds, end, de)
    assert coarse[0].tolist() == start.tolist() and coarse[-1].tolist() == end.tolist()
    fine = Polyline(bezier_connector(start, ds, end, de, n=2001))
    assert fine.headings[0] == pytest.approx(0.0, abs=1e-3)
    assert fine.headings[-1] == pytest.approx(math.pi / 2, abs=1e-3)


@pytest.mark.parametrize("radius", [5.0, 12.0, 30.0])
def test_quarter_turn_radius_error_below_3_percent(radius: float) -> None:
    pts = bezier_connector((0.0, 0.0), (1.0, 0.0), (radius, radius), (0.0, 1.0))
    assert len(pts) == 9
    radii = curvature_radii(pts)
    assert np.all(np.abs(radii / radius - 1) < 0.03)
    centre_dist = np.hypot(pts[:, 0], pts[:, 1] - radius)
    assert np.all(np.abs(centre_dist / radius - 1) < 0.03)


def test_straight_limit() -> None:
    pts = bezier_connector((0.0, 0.0), (1.0, 0.0), (9.0, 0.0), (1.0, 0.0))
    assert pts.tolist() == [[0.0, 0.0], [9.0, 0.0]]
    assert circular_handle(0.0, 9.0) == 3.0
    assert circular_handle(1e-9, 9.0) == pytest.approx(3.0)


@pytest.mark.parametrize("theta", [0.1, math.pi / 4, math.pi / 2, 2.0, math.pi])
def test_handle_equals_the_plan_formula(theta: float) -> None:
    chord = 7.0
    radius = chord / (2 * math.sin(theta / 2))
    assert circular_handle(theta, chord) == pytest.approx(4 / 3 * math.tan(theta / 4) * radius)


def test_u_turn() -> None:
    pts = bezier_connector((0.0, 0.0), (1.0, 0.0), (0.0, 6.0), (-1.0, 0.0))
    assert len(pts) == 17
    assert circular_handle(math.pi, 6.0) == pytest.approx(4.0)
    assert pts[:, 0].max() == pytest.approx(3.0)  # apex of the semicircle of radius 3


@pytest.mark.parametrize(
    ("theta", "n"),
    [(0.0, 2), (math.pi / 32, 2), (math.pi / 16, 2), (math.pi / 2, 9), (math.pi, 17)],
)
def test_sample_count(theta: float, n: int) -> None:
    assert connector_samples(theta) == n


def test_turn_angle() -> None:
    assert turn_angle((1, 0), (0, 3)) == pytest.approx(math.pi / 2)
    assert turn_angle((1, 0), (0, -3)) == pytest.approx(math.pi / 2)
    assert turn_angle((1, 0), (-1, 0)) == pytest.approx(math.pi)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="distinct"):
        bezier_connector((1, 1), (1, 0), (1, 1), (1, 0))
    with pytest.raises(ValueError, match="d_start"):
        bezier_connector((0, 0), (0, 0), (1, 1), (1, 0))
    with pytest.raises(ValueError, match="at least 2"):
        bezier_connector((0, 0), (1, 0), (1, 1), (0, 1), n=1)
