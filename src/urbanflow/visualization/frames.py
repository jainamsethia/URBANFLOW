"""Render frames and vehicle-registry deltas (plan K.10.1, K.10.3, B.2 #2/#3/#23).

A :class:`Frame` is the render state of one step: running vehicles sorted by uid with their
front-bumper ``xy`` in *local* coordinates (world = local + ``CompiledNetwork.origin``, the
same frame as :class:`~urbanflow.visualization.geometry.RenderGeometry`), heading, speed,
link and flag bits, plus optional per-movement signal states and per-link lane values.

Flag bits: 0 halting, 1 braking, 2 lane-change left, 3 lane-change right, 4 dwelling,
5 held at a stop line / zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from urbanflow.core import constants as C
from urbanflow.core.types import DriveSide, FloatArray, IntArray

if TYPE_CHECKING:
    from urbanflow.engine.engine import Engine
    from urbanflow.network.compiled import CompiledNetwork

__all__ = [
    "LANE_METRICS",
    "Frame",
    "LaneMetric",
    "RegistryTracker",
    "build_frame",
    "lane_values",
    "vehicle_xy",
]

FLAG_HALTING: Final = 1 << 0
FLAG_BRAKING: Final = 1 << 1
FLAG_LC_LEFT: Final = 1 << 2
FLAG_LC_RIGHT: Final = 1 << 3
FLAG_DWELLING: Final = 1 << 4
FLAG_HELD: Final = 1 << 5


class LaneMetric(IntEnum):
    """Per-lane values carried by frames (codes are part of the UFB1 header, B.2 #3)."""

    none = 0
    density = 1
    speed = 2
    queue = 3
    congestion = 4
    occupancy = 5
    utilization = 6
    flow = 7


LANE_METRICS: Final[dict[LaneMetric, tuple[str, str, tuple[float, float] | None]]] = {
    LaneMetric.density: ("Density", "veh/km", (0.0, 150.0)),
    LaneMetric.speed: ("Mean speed", "m/s", None),
    LaneMetric.queue: ("Queue", "veh", None),
    LaneMetric.congestion: ("Congestion", "1 - v/v0", (0.0, 1.0)),
    LaneMetric.occupancy: ("Occupancy", "", (0.0, 1.0)),
    LaneMetric.utilization: ("Utilization", "", (0.0, 1.0)),
    LaneMetric.flow: ("Flow", "veh/h", (0.0, 2000.0)),
}
"""``code -> (label, unit, domain)``; ``None`` domain = scale to the network maximum."""


@dataclass(frozen=True, slots=True)
class Frame:
    """Render state of one step (arrays sorted by ``uid``)."""

    step: int
    time: float
    uid: NDArray[Any]
    xy: FloatArray
    """``(n, 2)`` float32 local coordinates of front bumpers."""
    heading: FloatArray
    speed: FloatArray
    link: NDArray[Any]
    flags: NDArray[Any]
    signals: NDArray[Any] | None = None
    """Per-movement state codes (r 0, y 1, g 2, G 3, 255 unsignalised)."""
    lane_values: FloatArray | None = None
    lane_metric: LaneMetric = LaneMetric.none

    @property
    def n(self) -> int:
        """Number of vehicles."""
        return int(self.uid.size)


def vehicle_xy(
    net: CompiledNetwork, links: IntArray, pos: FloatArray, lat_offset: NDArray[Any]
) -> tuple[FloatArray, FloatArray]:
    """World ``(xy (N, 2), heading (N,))`` of front bumpers at ``pos`` along ``links``.

    ``lat_offset`` (m, positive toward the median) shifts the point along the unit normal;
    headings are radians counter-clockwise from +x.
    """
    xy, heading = net.point_at(links, pos)
    toward_median = 1.0 if net.drive_side is DriveSide.right else -1.0  # left of travel
    normal = np.column_stack((-np.sin(heading), np.cos(heading))) * toward_median
    world: FloatArray = xy + np.asarray(lat_offset, dtype=np.float64)[:, None] * normal
    return world + net.origin, heading


def lane_values(engine: Engine, run: IntArray, metric: LaneMetric) -> FloatArray:
    """Per-link values of ``metric`` from the running vehicles (float32; NaN = no data)."""
    net, veh = engine.network, engine.vehicles
    n = net.n_links
    link = veh.link[run].astype(np.intp)
    count = np.bincount(link, minlength=n).astype(np.float64)
    length = net.link_length.astype(np.float64)
    speed = np.bincount(link, veh.speed[run], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(count > 0, speed / count, np.nan)
        density = count / (length / C.METRES_PER_KM)
        if metric is LaneMetric.density:
            out = density
        elif metric is LaneMetric.speed:
            out = mean_speed
        elif metric is LaneMetric.queue:
            from urbanflow.engine.signalling import stopline_queue

            q = stopline_queue(
                link, veh.pos[run], veh.uid[run], veh.halting[run], length[: net.n_lanes]
            )
            out = np.full(n, np.nan)
            out[: net.n_lanes] = q
        elif metric is LaneMetric.congestion:
            ratio = np.bincount(link, veh.speed[run] / np.maximum(veh.v0[run], 1e-9), minlength=n)
            out = np.where(count > 0, 1.0 - ratio / count, np.nan)
        elif metric is LaneMetric.occupancy:
            occ = np.bincount(link, veh.length[run].astype(np.float64), minlength=n)
            out = np.minimum(1.0, occ / length)
        elif metric is LaneMetric.utilization:
            cap = np.maximum(1.0, length / C.VEH_SPACING_REF_M)
            out = np.minimum(1.0, count / cap)
        elif metric is LaneMetric.flow:
            out = np.where(
                count > 0, C.SECONDS_PER_HOUR / C.METRES_PER_KM * density * mean_speed, 0.0
            )
        else:
            out = np.full(n, np.nan)
    return out.astype(np.float32)


def build_frame(
    engine: Engine, *, signals: bool = True, lane_metric: LaneMetric = LaneMetric.none
) -> Frame:
    """The :class:`Frame` of the engine's current state."""

    net, veh = engine.network, engine.vehicles
    run = np.flatnonzero(veh.active)
    run = run[np.argsort(veh.uid[run], kind="stable")]
    links = veh.link[run].astype(np.intp)
    xy, heading = vehicle_xy(net, links, veh.pos[run], veh.lat_offset[run])
    lat = veh.lat_offset[run]
    flags = (
        veh.halting[run] * FLAG_HALTING
        | (veh.accel[run] < -C.BRAKE_FLAG_DECEL) * FLAG_BRAKING
        | (lat < 0) * FLAG_LC_LEFT  # moving toward the median: still drawn curb-side
        | (lat > 0) * FLAG_LC_RIGHT
        | (veh.dwell_left[run] > 0) * FLAG_DWELLING
        | veh.held[run] * FLAG_HELD
    ).astype(np.uint8)
    metric = LaneMetric(lane_metric)
    return Frame(
        step=engine.step_count,
        time=engine.time,
        uid=veh.uid[run].astype(np.uint32),
        xy=(xy - net.origin).astype(np.float32),
        heading=heading.astype(np.float32),
        speed=veh.speed[run].astype(np.float32),
        link=links.astype(np.uint32),
        flags=flags,
        signals=engine.signals.movement_state.copy() if signals else None,
        lane_values=None if metric is LaneMetric.none else lane_values(engine, run, metric),
        lane_metric=metric,
    )


class RegistryTracker:
    """Diffs the uid sets of consecutive emitted frames (K.10.3): a client learns about new
    vehicles from ``added`` before it sees them in a frame, and forgets ``removed`` ones.
    Works for live runs, replay seeks and dropped frames alike."""

    def __init__(self) -> None:
        self._last: NDArray[Any] = np.zeros(0, dtype=np.uint32)

    def delta(
        self, uids: NDArray[Any], *, reset: bool = False
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """``(added, removed)`` uids relative to the previous call (everything is added
        after ``reset``)."""
        current = np.asarray(uids, dtype=np.uint32)
        if reset:
            added, removed = current.copy(), np.zeros(0, dtype=np.uint32)
        else:
            added = np.setdiff1d(current, self._last, assume_unique=True)
            removed = np.setdiff1d(self._last, current, assume_unique=True)
        self._last = current.copy()
        return added.astype(np.uint32), removed.astype(np.uint32)
