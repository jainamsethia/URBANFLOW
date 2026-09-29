"""Road trimming and lane polylines (plan E.5 rules 2-3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urbanflow.core.types import DriveSide, IntersectionKind
from urbanflow.geometry import Polyline
from urbanflow.network.lanes import (
    DegenerateGeometryError,
    build_lanes,
    lane_offsets,
    resolved_value,
    trim_distance,
    trim_road,
)
from urbanflow.scenario.schema import IntersectionSpec, LaneSpec, RoadSpec


def _ix(kind: IntersectionKind, radius: float = 8.4) -> IntersectionSpec:
    return IntersectionSpec(id="J", point=(0.0, 0.0), kind=kind, radius=radius)


def _road(n: int = 2, widths: list[float] | None = None) -> RoadSpec:
    lanes = [LaneSpec(width=w, speed_limit=10.0) for w in widths or [3.2] * n]
    return RoadSpec.model_validate(
        {"id": "r", "from": "A", "to": "B", "lanes": lanes, "lane_width": 3.2, "speed_limit": 10}
    )


def test_trim_distance() -> None:
    signal = _ix(IntersectionKind.signalized)
    assert trim_distance((0.0, 0.0), signal) == pytest.approx(8.4)
    assert trim_distance((3.0, 4.0), signal) == pytest.approx(3.4)
    assert trim_distance((30.0, 0.0), signal) == 0.0  # outside the radius (W801)
    assert trim_distance((0.0, 0.0), _ix(IntersectionKind.boundary)) == 0.0


def test_trim_cuts_along_arc_length() -> None:
    # CityFlow cuts along the last segment and overshoots; the cut must follow the polyline
    ref = Polyline([(0.0, 0.0), (10.0, 0.0), (10.0, 4.0)])
    trimmed = trim_road(ref, 1.0, 7.0)
    assert trimmed.length == pytest.approx(ref.length - 8.0)
    np.testing.assert_allclose(trimmed.points, [[1.0, 0.0], [7.0, 0.0]])


def test_lane_offsets_stack_from_the_median() -> None:
    assert lane_offsets([3.2, 3.5, 3.0]) == pytest.approx([1.6, 4.95, 8.2])


@pytest.mark.parametrize(("side", "sign"), [(DriveSide.right, -1.0), (DriveSide.left, 1.0)])
def test_lanes_offset_toward_the_curb(side: DriveSide, sign: float) -> None:
    lanes = build_lanes(_road(widths=[3.2, 3.6]), Polyline([(0.0, 0.0), (50.0, 0.0)]), side)
    assert [lane.index for lane in lanes] == [0, 1]
    np.testing.assert_allclose(lanes[0].line.points[:, 1], sign * 1.6)
    np.testing.assert_allclose(lanes[1].line.points[:, 1], sign * 5.0)
    assert [lane.width for lane in lanes] == [3.2, 3.6]
    assert lanes[1].offset == pytest.approx(5.0)


def test_miter_keeps_lane_width_on_a_right_angle_bend() -> None:
    ref = Polyline([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)])  # left bend
    lane0, lane1 = build_lanes(_road(), ref, DriveSide.right)
    assert len(lane0.line.points) == 3
    corner = lane1.line.points[1] - lane0.line.points[1]
    assert np.linalg.norm(corner) == pytest.approx(3.2 * math.sqrt(2))


def test_lane_speed_limits_fall_back_to_the_road() -> None:
    road = RoadSpec.model_validate(
        {"id": "r", "from": "A", "to": "B", "lanes": [{}], "lane_width": 3.0, "speed_limit": 9}
    )
    (lane,) = build_lanes(road, Polyline([(0.0, 0.0), (20.0, 0.0)]), DriveSide.right)
    assert (lane.width, lane.speed_limit) == (3.0, 9.0)


def test_reversed_lane_is_degenerate() -> None:
    # a 2 m wide hairpin to the right: the 1.6 m offset lane folds back on the middle segment
    ref = Polyline([(0.0, 0.0), (40.0, 0.0), (40.0, -2.0), (0.0, -2.0)])
    with pytest.raises(DegenerateGeometryError, match="lane 0 reverses direction on segment 1"):
        build_lanes(_road(1), ref, DriveSide.right)
    assert len(build_lanes(_road(1), ref, DriveSide.left)) == 1  # outer side is fine


@pytest.mark.parametrize("end_y", [1.0, -1.0])  # ~177 deg turn; the lane outside / inside
def test_bevelled_vertices_keep_segments_parallel(end_y: float) -> None:
    ref = Polyline([(0.0, 0.0), (20.0, 0.0), (0.0, end_y)])
    (lane,) = build_lanes(_road(1), ref, DriveSide.right)
    pts = lane.line.points
    assert len(pts) == 4  # the vertex is bevelled into two points
    np.testing.assert_allclose(lane.line.headings[[0, 2]], ref.headings, atol=1e-12)
    np.testing.assert_allclose(pts[1] - pts[0], [20.0, 0.0], atol=1e-12)


def test_resolved_value() -> None:
    assert resolved_value(3.0, "x") == 3.0
    with pytest.raises(TypeError, match=r"needs Scenario\.resolved"):
        resolved_value(None, "lane width")
