"""Layer 1: pure-NumPy polylines, offsets, connector curves and crossings (plan E.5)."""

from urbanflow.geometry.curves import (
    bezier_connector,
    circular_handle,
    connector_samples,
    turn_angle,
)
from urbanflow.geometry.intersect import (
    bbox_overlap,
    bounding_box,
    first_crossing,
    point_polyline_distance,
    segment_intersections,
    separation_distance,
)
from urbanflow.geometry.polyline import Polyline, as_points, curvature_radii, offset_polyline

__all__ = [
    "Polyline",
    "as_points",
    "bbox_overlap",
    "bezier_connector",
    "bounding_box",
    "circular_handle",
    "connector_samples",
    "curvature_radii",
    "first_crossing",
    "offset_polyline",
    "point_polyline_distance",
    "segment_intersections",
    "separation_distance",
    "turn_angle",
]
