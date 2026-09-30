"""Engine side of signal control: the controllers' per-step lane data (plan H.3, F.1 step 1).

:class:`LaneStats` implements :class:`~urbanflow.signals.controllers.base.LaneData` from the
vehicle table at the start of a step. Each field is computed on first access
(``functools.cached_property``) and the engine builds one instance per step, so every
field costs at most one vectorised pass per step and nothing when no controller reads it.
Detector occupancy is the exception: its last-seen timers need every step, so the engine
evaluates :func:`detector_occupancy` each step while the network has signals.

ponytail: detectors are sampled once per step, so a body that passes a whole detector
within one step (speed * dt > detector + vehicle length) is missed; per-step swept
intervals would catch it if large dt and short detectors ever matter.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.types import BoolArray, FloatArray, IntArray, UIntArray, VehicleClass
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = ["LaneStats", "detector_occupancy", "stopline_queue"]


def detector_occupancy(net: CompiledNetwork, veh: VehicleTable, run: IntArray) -> BoolArray:
    """Per lane: some vehicle body overlaps ``[lane_detector_start, L]``.

    A body on the lane overlaps iff its front is past the detector start; a connector
    vehicle whose rear still hangs back over its from-lane end occupies that lane's
    detector.
    """
    n_lanes = net.n_lanes
    link = veh.link[run].astype(np.intp)
    pos = veh.pos[run]
    occupied = np.zeros(n_lanes, dtype=bool)
    on_lane = link < n_lanes
    lanes = link[on_lane]
    occupied[lanes[pos[on_lane] >= net.lane_detector_start[lanes]]] = True
    hang = ~on_lane & (pos < veh.length[run])
    occupied[net.conn_from_lane[link[hang] - n_lanes]] = True
    return occupied


def stopline_queue(
    link: IntArray, pos: FloatArray, uid: UIntArray, halting: BoolArray, lane_length: FloatArray
) -> IntArray:
    """Per lane: the stop-line queue ``Q_l`` (plan J.3), vectorised.

    Inputs are per running vehicle (any order; connector vehicles, ``link >= n_lanes``, are
    ignored) and ``lane_length`` has one entry per lane. Lane vehicles are sorted front
    first per lane (ties by uid); the queue is the leading run of halting vehicles, counted
    only when the front vehicle is within ``QUEUE_FRONT_TOLERANCE_M`` of the stop line.
    """
    n_lanes = lane_length.size
    keep = np.flatnonzero(link < n_lanes)
    if keep.size == 0:
        return np.zeros(n_lanes, dtype=np.int64)
    order = keep[np.lexsort((uid[keep], -pos[keep], link[keep]))]
    lane = link[order]
    start = np.r_[True, lane[1:] != lane[:-1]]
    moving = (~halting[order]).astype(np.int64)
    c = np.cumsum(moving)
    base = np.maximum.accumulate(np.where(start, c - moving, 0))
    in_queue = (c - base) == 0
    first = np.flatnonzero(start)
    valid = lane_length[lane[first]] - pos[order][first] <= C.QUEUE_FRONT_TOLERANCE_M
    group = np.cumsum(start) - 1
    return np.bincount(lane[in_queue & valid[group]], minlength=n_lanes)


class LaneStats:
    """Per-step :class:`~urbanflow.signals.controllers.base.LaneData` (lazy fields).

    ``run`` are the running handles at the start of the step; ``occupied`` and
    ``last_seen`` are this step's detector occupancy and seconds since each lane's detector
    was last occupied.
    """

    def __init__(
        self,
        net: CompiledNetwork,
        veh: VehicleTable,
        types: VehicleTypes,
        run: IntArray,
        occupied: BoolArray,
        last_seen: FloatArray,
    ) -> None:
        self.net = net
        self._veh, self._types, self._run = veh, types, run
        self._occupied, self._last_seen = occupied, last_seen

    @cached_property
    def _link(self) -> IntArray:
        return self._veh.link[self._run].astype(np.intp)

    @cached_property
    def _halting(self) -> BoolArray:
        return self._veh.halting[self._run]

    @cached_property
    def count(self) -> IntArray:
        """Per link: running vehicles whose front is on it."""
        return np.bincount(self._link, minlength=self.net.n_links)

    @cached_property
    def halting(self) -> IntArray:
        """Per link: halting vehicles."""
        return np.bincount(self._link[self._halting], minlength=self.net.n_links)

    @cached_property
    def queue(self) -> IntArray:
        """Per lane: stop-line queue (J.3)."""
        veh, run, net = self._veh, self._run, self.net
        lane_length = net.link_length[: net.n_lanes]
        return stopline_queue(self._link, veh.pos[run], veh.uid[run], self._halting, lane_length)

    @cached_property
    def waiting(self) -> FloatArray:
        """Per lane: sum of the vehicles' ``link_waiting_time``, s."""
        lanes = self._link < self.net.n_lanes
        weights = self._veh.link_waiting_time[self._run][lanes].astype(np.float64)
        return np.bincount(self._link[lanes], weights, minlength=self.net.n_lanes)

    @property
    def detector_occupied(self) -> BoolArray:
        """Per lane: a body is inside ``[L - DETECTOR_LENGTH, L]``."""
        return self._occupied

    @property
    def detector_last_seen(self) -> FloatArray:
        """Per lane: seconds since the detector was last occupied (0 now, inf never)."""
        return self._last_seen

    @cached_property
    def approaching(self) -> IntArray:
        """Per lane: moving vehicles within ``APPROACH_DISTANCE_M`` of the stop line."""
        net, link = self.net, self._link
        lanes = link < net.n_lanes
        near = net.link_length[np.where(lanes, link, 0)] - self._veh.pos[self._run]
        hit = lanes & ~self._halting & (near <= C.APPROACH_DISTANCE_M)
        return np.bincount(link[hit], minlength=net.n_lanes)

    @cached_property
    def _planning(self) -> BoolArray:
        """Lane vehicles with a planned connector."""
        return (self._link < self.net.n_lanes) & (self._veh.next_conn[self._run] >= 0)

    @cached_property
    def planned(self) -> IntArray:
        """Per connector index: lane vehicles planning it."""
        conn = self._veh.next_conn[self._run][self._planning] - self.net.n_lanes
        return np.bincount(conn, minlength=self.net.n_conn)

    @cached_property
    def planned_halting(self) -> IntArray:
        """Per connector index: halting lane vehicles planning it."""
        sel = self._planning & self._halting
        conn = self._veh.next_conn[self._run][sel] - self.net.n_lanes
        return np.bincount(conn, minlength=self.net.n_conn)

    @cached_property
    def emergency(self) -> tuple[IntArray, IntArray, FloatArray]:
        """Running emergency vehicles: ``(link, connector, distance)`` (see LaneData)."""
        veh, run, net = self._veh, self._run, self.net
        emergency = self._types.vclass[veh.type_idx[run]] == VehicleClass.emergency.code
        link = self._link[emergency]
        pos = veh.pos[run][emergency]
        on_lane = link < net.n_lanes
        conn = np.where(on_lane, veh.next_conn[run][emergency], link).astype(np.intp)
        dist = np.where(on_lane, net.link_length[link] - pos, -pos)
        return link, conn, dist
