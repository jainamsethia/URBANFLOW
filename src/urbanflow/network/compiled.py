"""The compiled network: immutable integer-indexed arrays the engine reads every step (E.3).

Index spaces are 0-based and contiguous: roads ``R``; links ``K = n_lanes + n_conn`` with
lanes first (lane ``k`` of road ``r`` is link ``road_lane_start[r] + k``, lane 0 = median)
and connectors after them (connector ``c`` is link ``n_lanes + c``); movements ``M``;
intersections ``I``; conflicts ``C``. Every string id has a ``dict`` map and a reverse
tuple. Coordinates are local: world = local + ``origin`` (the world bbox minimum), which
keeps float32 frame coordinates precise. Every array is read-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from urbanflow.core.errors import NotFoundError, Severity, ValidationIssue, suggest
from urbanflow.core.types import BoolArray, DriveSide, FloatArray, IntArray, UIntArray
from urbanflow.scenario.schema import VehicleTypeSpec

__all__ = ["CompileReport", "CompiledNetwork"]


@dataclass(frozen=True, slots=True)
class CompileReport:
    """Diagnostics of a successful compile (errors raise ``ScenarioValidationError``)."""

    warnings: tuple[ValidationIssue, ...] = ()
    """W302 (conflicting protected movements) and W304 (connectors crossing twice)."""

    def __post_init__(self) -> None:
        if any(i.severity is not Severity.warning for i in self.warnings):
            raise ValueError("a CompileReport holds warnings only")


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class CompiledNetwork:
    """Immutable compiled network (plan E.3). Build it with ``compile_network(scenario)``."""

    # ------------------------------------------------------------------ identity and maps
    scenario_hash: str
    """``Scenario.content_hash`` of the compiled scenario."""
    geometry_crc: int
    """crc32 of ``scenario_hash``; frames carry it to detect geometry mismatches."""
    drive_side: DriveSide
    road_ids: tuple[str, ...]
    road_index: Mapping[str, int]
    link_ids: tuple[str, ...]
    """Lane ids ``"{road}_{k}"``, then connector ids ``"{lane_in}->{lane_out}"``."""
    link_index: Mapping[str, int]
    mov_ids: tuple[str, ...]
    mov_index: Mapping[str, int]
    int_ids: tuple[str, ...]
    int_index: Mapping[str, int]
    vehicle_types: tuple[VehicleTypeSpec, ...]
    """Resolved vehicle types in type-index order (sorted by id)."""
    report: CompileReport
    """Compile warnings (as ValidationIssues with JSON paths)."""

    # ------------------------------------------------------------------ roads
    road_from: IntArray
    road_to: IntArray
    road_lane_start: IntArray
    road_n_lanes: UIntArray
    road_length: FloatArray
    """Trimmed reference length, m."""
    road_speed_limit: FloatArray
    road_capacity_vph: FloatArray
    """``n * 3600 / (T + (l + s0) / v)`` with the built-in car defaults."""

    # ------------------------------------------------------------------ links
    link_kind: UIntArray
    """``LinkKind`` code: 0 lane, 1 connector."""
    link_length: FloatArray
    link_speed_limit: FloatArray
    link_width: NDArray[np.float32]
    link_road: IntArray
    """Road of a lane; -1 for connectors."""
    link_lane_index: IntArray
    """Index within its road; -1 for connectors."""
    link_intersection: IntArray
    """A lane's downstream intersection; a connector's own intersection."""
    link_movement: IntArray
    """A connector's movement; -1 for lanes."""
    link_is_exit: BoolArray
    """The lane ends at a boundary intersection (rendering only; arrival is route-based)."""
    lane_left: IntArray
    """Adjacent lane link toward the median, or -1 (``n_lanes`` entries)."""
    lane_right: IntArray
    """Adjacent lane link toward the curb, or -1."""
    lane_out_ptr: IntArray
    lane_out_conn: IntArray
    """CSR: connector link ids leaving each lane, ascending."""
    conn_from_lane: IntArray
    """Lane link id at the start of each connector (``n_conn`` entries)."""
    conn_to_lane: IntArray
    road_pair_mask: Mapping[tuple[int, int], int]
    """Bit ``k`` set iff lane ``k`` of road ``r`` has a connector to road ``r'``."""
    road_pair_conns: Mapping[tuple[int, int], tuple[int, ...]]
    """Connector link ids from road ``r`` to road ``r'``."""
    lane_detector_start: FloatArray
    """``max(0, L - DETECTOR_LENGTH)`` of every lane, m."""

    # ------------------------------------------------------------------ movements
    mov_intersection: IntArray
    mov_from_road: IntArray
    mov_to_road: IntArray
    mov_turn: UIntArray
    """``TurnKind`` code."""
    mov_static_rank: UIntArray
    """3/2/1 at priority intersections, 1 uncontrolled, 0 signalised (rank from state)."""
    mov_conn_ptr: IntArray
    mov_conn: IntArray
    """CSR: connector link ids of each movement, in connection order."""

    # ------------------------------------------------------------------ intersections
    int_kind: UIntArray
    """``IntersectionKind`` code."""
    int_point: FloatArray
    """Centres, local coordinates, shape ``(I, 2)``."""
    int_radius: FloatArray
    int_program: IntArray
    """Index of the signal program (signalised intersections in order), or -1."""
    int_in_ptr: IntArray
    int_in_lanes: IntArray
    """CSR: incoming lanes, clockwise by approach bearing from north, then lane index."""
    int_out_ptr: IntArray
    int_out_lanes: IntArray

    # ------------------------------------------------------------------ conflicts
    conf_a: IntArray
    conf_b: IntArray
    conf_kind: UIntArray
    """0 crossing, 1 merging (same to_lane), 2 diverging (same from_lane)."""
    conf_zone_a: FloatArray
    conf_zone_b: FloatArray
    conn_conf_ptr: IntArray
    conn_conf: IntArray
    conn_conf_side: UIntArray

    # ------------------------------------------------------------------ polylines
    pts: FloatArray
    """All link polylines concatenated, local coordinates, shape ``(P, 2)``."""
    link_pt_start: IntArray
    """Point offsets, ``K + 1`` entries."""
    s_global: FloatArray
    """Globally monotone arc key ``link_base[l] + cumulative arc``."""
    link_base: FloatArray
    """``link_base[l + 1] = link_base[l] + link_length[l] + LINK_BASE_GAP``."""
    seg_heading: FloatArray
    """Heading of the segment starting at each point (the last point repeats its link's)."""
    origin: FloatArray
    """World coordinates of the local origin (world bbox minimum)."""
    bbox: FloatArray
    """``[xmin, ymin, xmax, ymax]`` in local coordinates."""

    def __post_init__(self) -> None:
        for arr in self.arrays().values():
            arr.setflags(write=False)

    def __repr__(self) -> str:
        return (
            f"CompiledNetwork(roads={self.n_roads}, lanes={self.n_lanes}, "
            f"connectors={self.n_conn}, movements={self.n_movements}, "
            f"intersections={self.n_intersections}, conflicts={self.n_conflicts}, "
            f"hash={self.scenario_hash[:12]})"
        )

    # ------------------------------------------------------------------ sizes
    @property
    def n_roads(self) -> int:
        return len(self.road_ids)

    @property
    def n_links(self) -> int:
        return len(self.link_ids)

    @property
    def n_lanes(self) -> int:
        return len(self.lane_left)

    @property
    def n_conn(self) -> int:
        return len(self.conn_from_lane)

    @property
    def n_movements(self) -> int:
        return len(self.mov_ids)

    @property
    def n_intersections(self) -> int:
        return len(self.int_ids)

    @property
    def n_conflicts(self) -> int:
        return len(self.conf_a)

    # ------------------------------------------------------------------ queries
    def arrays(self) -> dict[str, NDArray[Any]]:
        """Every numpy array field by name, in declaration order."""
        out: dict[str, NDArray[Any]] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                out[f.name] = value
        return out

    def read_only(self) -> bool:
        """True when no array can be written (always, unless someone flips the flags)."""
        return not any(arr.flags.writeable for arr in self.arrays().values())

    def lane_of(self, road: int | str, k: int) -> int:
        """Link id of lane ``k`` (0 = median) of ``road`` (index or id)."""
        if isinstance(road, str):
            if road not in self.road_index:
                raise NotFoundError(f'unknown road "{road}"{suggest(road, self.road_index)}')
            road = self.road_index[road]
        if not 0 <= road < self.n_roads:
            raise NotFoundError(f"road index {road} out of range ({self.n_roads} roads)")
        n = int(self.road_n_lanes[road])
        if not 0 <= k < n:
            raise NotFoundError(
                f'lane {k} does not exist: road "{self.road_ids[road]}" has {n} lanes (0-{n - 1})'
            )
        return int(self.road_lane_start[road]) + k

    def link_points(self, link: int) -> FloatArray:
        """Polyline of ``link`` (local coordinates, a read-only view)."""
        return self.pts[self.link_pt_start[link] : self.link_pt_start[link + 1]]

    def point_at(self, links: ArrayLike, pos: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Local points and headings at ``pos`` metres along ``links`` (vectorised).

        ``pos`` is clamped to ``[0, link_length]``. One ``searchsorted`` over the global
        arc key serves all queries: ``s = link_base[l] + pos`` locates the segment, clipped
        to the link's own segments.
        """
        lk = np.asarray(links, dtype=np.intp)
        s = self.link_base[lk] + np.clip(np.asarray(pos, dtype=np.float64), 0, self.link_length[lk])
        i = np.searchsorted(self.s_global, s, side="right") - 1
        i = np.clip(i, self.link_pt_start[lk], self.link_pt_start[lk + 1] - 2)
        t = (s - self.s_global[i]) / (self.s_global[i + 1] - self.s_global[i])
        xy: FloatArray = self.pts[i] + t[..., None] * (self.pts[i + 1] - self.pts[i])
        return xy, self.seg_heading[i]
