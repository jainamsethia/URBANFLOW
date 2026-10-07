"""Read views, per-step vector state and vehicle control of a simulation (plan AA 5.4, 5.5).

* :class:`VehicleView`, :class:`LaneView` and :class:`RoadView` are frozen snapshot copies:
  they never change after creation.
* Vector forms (``sim.state.*``, ``sim.lanes.vehicle_counts()`` ...) are read-only arrays
  computed once per step and cached until the next ``step()`` or ``reset()``; ``.copy()``
  them to keep them.
* ``sim.state.raw`` exposes the vehicle table as zero-copy, read-only, capacity-sized live
  views plus the ``active`` mask; after a step they may be stale (arrays grow).

Vehicles are identified by id or uid; engine slots (handles) never leave this module.
Coordinates are world coordinates (the scenario's): the link polyline point at the front
bumper plus ``lat_offset`` along the unit normal toward the median.

:class:`SignalView`, :class:`PhaseInfo`, :class:`MovementInfo` and
:class:`IntersectionView` are snapshot copies too; ``sim.signals`` (``control.py``) and
``sim.intersections`` build them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from urbanflow.core import constants as C
from urbanflow.core.errors import NotFoundError, suggest
from urbanflow.core.types import (
    FloatArray,
    IntArray,
    IntersectionKind,
    SignalState,
    Stage,
    TurnKind,
    UIntArray,
    VehicleClass,
    VehicleStatus,
)
from urbanflow.engine.signalling import stopline_queue
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.scenario.schema import DepartLane, DepartSpeed
from urbanflow.signals import FixedTime, Preemption, SignalProgram, controller_name
from urbanflow.signals.controllers import realised_cycle, stage_steps
from urbanflow.vehicles.table import COLUMNS
from urbanflow.visualization.frames import vehicle_xy
from urbanflow.visualization.geometry import render_geometry

if TYPE_CHECKING:
    from urbanflow.engine import Engine

__all__ = [
    "IntersectionCollection",
    "IntersectionView",
    "LaneCollection",
    "LaneView",
    "MovementInfo",
    "NetworkInfo",
    "PhaseInfo",
    "RoadCollection",
    "RoadView",
    "SignalView",
    "StateArrays",
    "StepCache",
    "VehicleCollection",
    "VehicleView",
    "phase_infos",
    "signal_view",
    "vehicle_xy",
]

StepCache = dict[str, Any]
"""Per-step cache shared by the views of one simulation; the facade clears it in place."""

_RUNNING: Final = VehicleStatus.running.code
_WAITING: Final = VehicleStatus.waiting_insert.code


def _cached[T](cache: StepCache, key: str, build: Callable[[], T]) -> T:
    if key not in cache:
        cache[key] = build()
    return cast(T, cache[key])


def _frozen[A: NDArray[Any]](arr: A) -> A:
    arr.setflags(write=False)
    return arr


# ------------------------------------------------------------------------------- vehicles
@dataclass(frozen=True, slots=True)
class VehicleView:
    """Snapshot of one vehicle (plan AA 5.4). Location fields are None unless running."""

    id: str
    uid: int
    type: str
    vclass: VehicleClass
    status: VehicleStatus
    length: float
    width: float
    max_speed: float
    max_accel: float
    decel: float
    emergency_decel: float
    min_gap: float
    speed_factor: float
    politeness: float
    headway: float
    """Desired time gap T, s."""
    reaction_time: float
    """= dt: decisions use the previous step's state (G.2 A2), s."""
    model_params: Mapping[str, float]
    """Parameters of the car-following model for this vehicle's type."""
    road: str | None
    """Road of the lane the vehicle is on (None on a connector)."""
    lane: str | None
    connector: str | None
    intersection: str | None
    """The intersection whose connector the vehicle is on."""
    link_index: int | None
    lane_index: int | None
    position: float | None
    """Front-bumper position along the link, m."""
    x: float | None
    y: float | None
    heading: float | None
    """Radians, counter-clockwise from +x."""
    lateral_offset: float
    speed: float
    acceleration: float
    desired_speed: float | None
    """Speed override if set, else the desired speed on the current link, m/s."""
    route: tuple[str, ...]
    route_index: int
    next_road: str | None
    destination: str
    depart_time: float
    insert_time: float | None
    travel_time: float | None
    """Time since insertion, s."""
    waiting_time: float
    stops: int
    distance: float
    speed_override: float | None


def _status(status: VehicleStatus | str) -> VehicleStatus:
    try:
        return VehicleStatus(status)
    except ValueError:
        names = [s.value for s in VehicleStatus]
        raise NotFoundError(
            f'unknown vehicle status "{status}"{suggest(str(status), names)} '
            f"(available: {', '.join(names)})"
        ) from None


class VehicleCollection:
    """``sim.vehicles``: read access by id and the vehicle control commands (AA 5.4).

    ``len()`` and iteration cover running vehicles (iteration in uid order); lookups by id
    also find vehicles waiting to be inserted. Control calls take effect in the next step.
    """

    def __init__(
        self, engine: Engine, cache: StepCache, arrived_uids: Callable[[], IntArray]
    ) -> None:
        self._engine = engine
        self._cache = cache
        self._arrived_uids = arrived_uids

    # ------------------------------------------------------------------ lookup
    def _handle(self, vehicle_id: str) -> int:
        veh = self._engine.vehicles
        h = veh.id_to_handle.get(vehicle_id)
        if h is not None:
            return h
        if vehicle_id in veh.uid_to_id:
            raise NotFoundError(
                f'vehicle "{vehicle_id}" is no longer in the simulation (arrived or removed)'
            )
        return veh.handle_of(vehicle_id)  # raises NotFoundError with a hint

    def _by_uid(self) -> IntArray:
        def build() -> IntArray:
            veh = self._engine.vehicles
            run = veh.running()
            return _frozen(run[np.argsort(veh.uid[run], kind="stable")])

        return _cached(self._cache, "vehicles_by_uid", build)

    def _view(self, h: int) -> VehicleView:
        eng = self._engine
        net, veh, types = eng.network, eng.vehicles, eng.types
        t = int(veh.type_idx[h])
        status = VehicleStatus.from_code(int(veh.status[h]))
        running = status is VehicleStatus.running
        link = int(veh.link[h]) if running else -1
        on_lane = 0 <= link < net.n_lanes
        on_conn = link >= net.n_lanes
        route = tuple(net.road_ids[r] for r in eng.routes.get(int(veh.route_id[h])).tolist())
        cursor = int(veh.route_cursor[h])
        x = y = heading = None
        if running:
            xy, hd = vehicle_xy(
                net, veh.link[h : h + 1], veh.pos[h : h + 1], veh.lat_offset[h : h + 1]
            )
            x, y, heading = float(xy[0, 0]), float(xy[0, 1]), float(hd[0])
        override = float(veh.speed_override[h])
        override_or_none = None if np.isnan(override) else override
        model = eng.car_following.Params.model_fields
        return VehicleView(
            id=str(veh.ids[h]),
            uid=int(veh.uid[h]),
            type=types.ids[t],
            vclass=VehicleClass.from_code(int(types.vclass[t])),
            status=status,
            length=float(types.length[t]),
            width=float(types.width[t]),
            max_speed=float(types.max_speed[t]),
            max_accel=float(types.accel[t]),
            decel=float(types.decel[t]),
            emergency_decel=float(types.emergency_decel[t]),
            min_gap=float(types.min_gap[t]),
            speed_factor=float(veh.speed_factor[h]),
            politeness=float(types.politeness[t]),
            headway=float(types.headway[t]),
            reaction_time=eng.config.dt,
            model_params={p: float(getattr(types.params, p)[t]) for p in model},
            road=net.road_ids[int(net.link_road[link])] if on_lane else None,
            lane=net.link_ids[link] if on_lane else None,
            connector=net.link_ids[link] if on_conn else None,
            intersection=net.int_ids[int(net.link_intersection[link])] if on_conn else None,
            link_index=link if running else None,
            lane_index=int(net.link_lane_index[link]) if on_lane else None,
            position=float(veh.pos[h]) if running else None,
            x=x,
            y=y,
            heading=heading,
            lateral_offset=float(veh.lat_offset[h]),
            speed=float(veh.speed[h]),
            acceleration=float(veh.accel[h]),
            desired_speed=(
                (float(veh.v0[h]) if override_or_none is None else override_or_none)
                if running
                else None
            ),
            route=route,
            route_index=cursor,
            next_road=route[cursor + 1] if cursor + 1 < len(route) else None,
            destination=route[-1],
            depart_time=float(veh.depart_time[h]),
            insert_time=float(veh.insert_time[h]) if running else None,
            travel_time=eng.time - float(veh.insert_time[h]) if running else None,
            waiting_time=float(veh.waiting_time[h]),
            stops=int(veh.stops[h]),
            distance=float(veh.distance[h]),
            speed_override=override_or_none,
        )

    def __getitem__(self, vehicle_id: str) -> VehicleView:
        """The view of a running or waiting vehicle; ``NotFoundError`` with a hint."""
        return self._view(self._handle(vehicle_id))

    def get(self, vehicle_id: str, default: VehicleView | None = None) -> VehicleView | None:
        """Like ``[]`` but returns ``default`` for unknown or departed vehicles."""
        h = self._engine.vehicles.id_to_handle.get(vehicle_id)
        return default if h is None else self._view(h)

    def __contains__(self, vehicle_id: object) -> bool:
        return vehicle_id in self._engine.vehicles.id_to_handle

    def __len__(self) -> int:
        return len(self._by_uid())

    def __iter__(self) -> Iterator[VehicleView]:
        for h in self._by_uid().tolist():
            yield self._view(h)

    def ids(self, status: VehicleStatus | str = VehicleStatus.running) -> list[str]:
        """Ids of the vehicles with ``status``, in uid order."""
        veh = self._engine.vehicles
        wanted = _status(status)
        if wanted is VehicleStatus.pending:
            return []
        if wanted in (VehicleStatus.running, VehicleStatus.waiting_insert):
            live = np.fromiter(veh.id_to_handle.values(), dtype=np.intp)
            live = live[veh.status[live] == wanted.code]
            uids = np.sort(veh.uid[live])
        else:
            arrived = self._arrived_uids()
            if wanted is VehicleStatus.arrived:
                uids = np.sort(arrived)
            else:  # removed: every uid neither live nor arrived
                gone = np.ones(veh.next_uid, dtype=bool)
                gone[veh.uid[np.fromiter(veh.id_to_handle.values(), dtype=np.intp)]] = False
                gone[arrived] = False
                uids = np.flatnonzero(gone)
        return [veh.uid_to_id[u] for u in uids.tolist()]

    def count(self, status: VehicleStatus | str = VehicleStatus.running) -> int:
        """Number of vehicles with ``status`` (arrived and removed are cumulative)."""
        eng = self._engine
        counts = {
            VehicleStatus.pending: 0,
            VehicleStatus.waiting_insert: eng.backlog,
            VehicleStatus.running: len(self),
            VehicleStatus.arrived: eng.arrived,
            VehicleStatus.removed: eng.removed,
        }
        return counts[_status(status)]

    def uid(self, vehicle_id: str) -> int:
        """The uid of any vehicle spawned in this run."""
        veh = self._engine.vehicles
        if vehicle_id not in veh.id_to_handle and vehicle_id in veh.uid_to_id:
            return veh.uid_to_id.index(vehicle_id)  # ponytail: O(n) scan for departed ids
        return int(veh.uid[veh.handle_of(vehicle_id)])  # NotFoundError with a hint

    def id_of(self, uid: int) -> str:
        """The id of the vehicle with ``uid`` (any vehicle spawned in this run)."""
        ids = self._engine.vehicles.uid_to_id
        if not 0 <= uid < len(ids):
            raise NotFoundError(f"unknown vehicle uid {uid} ({len(ids)} vehicles spawned)")
        return ids[uid]

    def to_columns(self) -> dict[str, NDArray[Any] | list[str]]:
        """Running vehicles as columns in uid order (copies; ``pandas.DataFrame``-ready)."""
        eng = self._engine
        net, veh, types = eng.network, eng.vehicles, eng.types
        h = self._by_uid()
        link = veh.link[h]
        xy, heading = vehicle_xy(net, link, veh.pos[h], veh.lat_offset[h])
        return {
            "id": [str(veh.ids[i]) for i in h.tolist()],
            "uid": veh.uid[h],
            "type": [types.ids[t] for t in veh.type_idx[h].tolist()],
            "link": [net.link_ids[k] for k in link.tolist()],
            "link_index": link,
            "position": veh.pos[h],
            "speed": veh.speed[h],
            "acceleration": veh.accel[h],
            "x": xy[:, 0].copy(),
            "y": xy[:, 1].copy(),
            "heading": heading,
            "waiting_time": veh.waiting_time[h],
            "stops": veh.stops[h],
            "distance": veh.distance[h],
            "depart_time": veh.depart_time[h],
            "insert_time": veh.insert_time[h],
        }

    # ------------------------------------------------------------------ control
    def add(
        self,
        *,
        route: list[str] | tuple[str, ...] | None = None,
        origin: str | None = None,
        destination: str | None = None,
        via: list[str] | tuple[str, ...] = (),
        vehicle_type: str = "car",
        id: str | None = None,
        depart_lane: DepartLane = "best",
        depart_speed: DepartSpeed = "max",
    ) -> str:
        """Spawn a vehicle now; it waits in its first road's queue and is inserted during a
        later step. Give ``route`` (road ids) or ``origin`` and ``destination`` (and
        ``via``). Returns its id (``api.<n>`` unless ``id`` is given)."""
        self._cache.clear()  # counts and orders change now, not only after the next step
        return self._engine.commands.add_vehicle(
            route=route,
            origin=origin,
            destination=destination,
            via=via,
            vehicle_type=vehicle_type,
            id=id,
            depart_lane=depart_lane,
            depart_speed=depart_speed,
        )

    def remove(self, vehicle_id: str) -> None:
        """Take a running or waiting vehicle out now (status ``removed``)."""
        self._engine.commands.remove_vehicle(vehicle_id)
        self._cache.clear()

    def set_speed(
        self, vehicle_id: str, speed: float | None, *, duration: float | None = None
    ) -> None:
        """Override the desired speed (m/s) for ``duration`` s, or until cleared with None.

        Car-following and the safe-speed cap still apply.
        """
        self._engine.commands.set_speed(vehicle_id, speed, duration)

    def set_route(self, vehicle_id: str, roads: list[str] | tuple[str, ...]) -> None:
        """Replace the rest of the route by ``roads`` (road ids). It must start at the
        current road (the first road while waiting; the next road on a connector) and be
        connected; a vehicle committed to its next stop line must keep going through the
        road its connector leads to. Otherwise ``CommandError``."""
        self._engine.commands.set_route(vehicle_id, roads)
        self._cache.clear()


# ------------------------------------------------------------------------------- lanes, roads
@dataclass(frozen=True, slots=True)
class _LaneStats:
    """Per-lane aggregates of one step (lanes only; connectors are not lanes)."""

    handles: IntArray
    """Running vehicles on lanes sorted by (lane, front to back, uid)."""
    ptr: IntArray
    """CSR offsets of each lane's vehicles in ``handles``."""
    count: IntArray
    halting: IntArray
    speed_sum: FloatArray
    length_sum: FloatArray
    queue: IntArray


def _mean(total: FloatArray, count: IntArray) -> FloatArray:
    return _frozen(np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0))


def _lane_stats(engine: Engine) -> _LaneStats:
    net, veh = engine.network, engine.vehicles
    n = net.n_lanes
    run = veh.running()
    run = run[veh.link[run] < n]
    order = np.lexsort((veh.uid[run], -veh.pos[run], veh.link[run]))
    h = run[order]
    lane = veh.link[h].astype(np.intp)
    count = np.bincount(lane, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(count)
    halting = veh.halting[h]
    weights = halting.astype(np.float64)
    return _LaneStats(
        handles=_frozen(h),
        ptr=_frozen(ptr),
        count=_frozen(count),
        halting=_frozen(np.bincount(lane, weights=weights, minlength=n).astype(np.int64)),
        speed_sum=np.bincount(lane, weights=veh.speed[h], minlength=n),
        length_sum=np.bincount(lane, weights=veh.length[h], minlength=n),
        queue=_frozen(stopline_queue(lane, veh.pos[h], veh.uid[h], halting, net.link_length[:n])),
    )


class _Indexed[V]:
    """Ids in compiled index order, lookups by id or index, iteration over views."""

    kind: str = ""

    def __init__(self, engine: Engine, cache: StepCache, ids: tuple[str, ...]) -> None:
        self._engine = engine
        self._cache = cache
        self.ids = ids
        """Ids in compiled index order (the order of the vector forms)."""
        self._index = {name: i for i, name in enumerate(ids)}

    def index(self, item_id: str) -> int:
        """Compiled index of ``item_id``; ``NotFoundError`` with a hint if unknown."""
        try:
            return self._index[item_id]
        except KeyError:
            hint = suggest(item_id, self._index)
            raise NotFoundError(f'unknown {self.kind} "{item_id}"{hint}') from None

    def __getitem__(self, key: str | int) -> V:
        if isinstance(key, str):
            return self._make(self.index(key))
        if not 0 <= key < len(self.ids):
            raise NotFoundError(f"{self.kind} index {key} out of range ({len(self.ids)})")
        return self._make(int(key))

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._index

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[V]:
        for i in range(len(self.ids)):
            yield self._make(i)

    def _make(self, i: int) -> V:
        raise NotImplementedError

    def _stats(self) -> _LaneStats:
        return _cached(self._cache, "lane_stats", lambda: _lane_stats(self._engine))


@dataclass(frozen=True, slots=True)
class LaneView:
    """Snapshot of one lane (plan AA 5.4)."""

    id: str
    road: str
    index: int
    """Lane index within its road (0 = median)."""
    length: float
    width: float
    speed_limit: float
    vehicle_count: int
    halting_count: int
    mean_speed: float | None
    """None when the lane is empty."""
    occupancy: float
    queue_length: int
    """Stop-line queue, vehicles (J.3)."""
    vehicle_ids: tuple[str, ...]
    """Front to back."""


class LaneCollection(_Indexed[LaneView]):
    """``sim.lanes``: every lane in compiled order, with per-step vector forms."""

    kind = "lane"

    def __init__(self, engine: Engine, cache: StepCache) -> None:
        net = engine.network
        super().__init__(engine, cache, net.link_ids[: net.n_lanes])

    def vehicle_counts(self) -> IntArray:
        """Running vehicles whose front is on each lane."""
        return self._stats().count

    def halting_counts(self) -> IntArray:
        """Vehicles slower than ``config.halting_speed`` on each lane."""
        return self._stats().halting

    def mean_speeds(self) -> FloatArray:
        """Mean speed per lane, m/s (NaN when empty)."""
        stats = self._stats()
        return _cached(self._cache, "lane_speed", lambda: _mean(stats.speed_sum, stats.count))

    def occupancy(self) -> FloatArray:
        """Space occupancy ``min(1, sum of vehicle lengths / lane length)`` (J.2)."""

        def build() -> FloatArray:
            length = self._engine.network.link_length[: len(self.ids)]
            return _frozen(np.minimum(1.0, self._stats().length_sum / length))

        return _cached(self._cache, "lane_occupancy", build)

    def waiting_times(self) -> FloatArray:
        """Per lane: summed time the vehicles on it have spent halting on it, s."""

        def build() -> FloatArray:
            veh, n = self._engine.vehicles, len(self.ids)
            run = np.flatnonzero(veh.active)
            link = veh.link[run]
            lanes = link < n
            weights = veh.link_waiting_time[run][lanes].astype(np.float64)
            return _frozen(np.bincount(link[lanes], weights, minlength=n))

        return _cached(self._cache, "lane_waiting", build)

    def queue_lengths(self) -> IntArray:
        """Stop-line queue per lane: contiguous halting vehicles from the front (J.3)."""
        return self._stats().queue

    def _make(self, i: int) -> LaneView:
        net, veh = self._engine.network, self._engine.vehicles
        stats = self._stats()
        mean = float(self.mean_speeds()[i])
        on_lane = stats.handles[stats.ptr[i] : stats.ptr[i + 1]].tolist()
        return LaneView(
            id=self.ids[i],
            road=net.road_ids[int(net.link_road[i])],
            index=int(net.link_lane_index[i]),
            length=float(net.link_length[i]),
            width=float(net.link_width[i]),
            speed_limit=float(net.link_speed_limit[i]),
            vehicle_count=int(stats.count[i]),
            halting_count=int(stats.halting[i]),
            mean_speed=None if np.isnan(mean) else mean,
            occupancy=float(self.occupancy()[i]),
            queue_length=int(stats.queue[i]),
            vehicle_ids=tuple(str(veh.ids[h]) for h in on_lane),
        )


@dataclass(frozen=True, slots=True)
class RoadView:
    """Snapshot of one road (plan AA 5.4)."""

    id: str
    from_intersection: str
    to_intersection: str
    lanes: tuple[str, ...]
    """Lane ids, median first."""
    n_lanes: int
    length: float
    speed_limit: float
    capacity_vph: float
    utilization: float
    """``min(1, sum n / sum c)`` with jam capacity ``c = max(1, L / 7.5 m)`` per lane."""
    vehicle_count: int
    halting_count: int
    mean_speed: float | None
    travel_time: float
    """Current estimate ``L / max(mean speed, 1 m/s)``; free-flow ``L / limit`` if empty."""


@dataclass(frozen=True, slots=True)
class _RoadStats:
    count: IntArray
    halting: IntArray
    mean_speed: FloatArray
    occupancy: FloatArray
    queue: IntArray
    utilization: FloatArray


class RoadCollection(_Indexed[RoadView]):
    """``sim.roads``: every road in compiled order, with per-step vector forms (sums and
    count-weighted means over its lanes)."""

    kind = "road"

    def __init__(self, engine: Engine, cache: StepCache) -> None:
        super().__init__(engine, cache, engine.network.road_ids)

    def _road_stats(self) -> _RoadStats:
        def build() -> _RoadStats:
            net, lanes = self._engine.network, self._stats()
            n = net.n_roads
            road = net.link_road[: net.n_lanes]
            length = net.link_length[: net.n_lanes]

            def total(values: NDArray[Any]) -> FloatArray:
                return np.bincount(road, weights=values, minlength=n)

            count = _frozen(total(lanes.count).astype(np.int64))
            jam = total(np.maximum(1.0, length / C.VEH_SPACING_REF_M))
            return _RoadStats(
                count=count,
                halting=_frozen(total(lanes.halting).astype(np.int64)),
                mean_speed=_mean(total(lanes.speed_sum), count),
                occupancy=_frozen(np.minimum(1.0, total(lanes.length_sum) / total(length))),
                queue=_frozen(total(lanes.queue).astype(np.int64)),
                utilization=_frozen(np.minimum(1.0, count / jam)),
            )

        return _cached(self._cache, "road_stats", build)

    def vehicle_counts(self) -> IntArray:
        """Running vehicles on each road's lanes."""
        return self._road_stats().count

    def halting_counts(self) -> IntArray:
        return self._road_stats().halting

    def mean_speeds(self) -> FloatArray:
        """Mean speed per road, m/s (NaN when empty)."""
        return self._road_stats().mean_speed

    def occupancy(self) -> FloatArray:
        """``min(1, sum of vehicle lengths / sum of lane lengths)``."""
        return self._road_stats().occupancy

    def queue_lengths(self) -> IntArray:
        """Sum of the lanes' stop-line queues, vehicles."""
        return self._road_stats().queue

    def _make(self, i: int) -> RoadView:
        net = self._engine.network
        stats = self._road_stats()
        start, n = int(net.road_lane_start[i]), int(net.road_n_lanes[i])
        length, limit = float(net.road_length[i]), float(net.road_speed_limit[i])
        mean = float(stats.mean_speed[i])
        empty = np.isnan(mean)
        return RoadView(
            id=self.ids[i],
            from_intersection=net.int_ids[int(net.road_from[i])],
            to_intersection=net.int_ids[int(net.road_to[i])],
            lanes=net.link_ids[start : start + n],
            n_lanes=n,
            length=length,
            speed_limit=limit,
            capacity_vph=float(net.road_capacity_vph[i]),
            utilization=float(stats.utilization[i]),
            vehicle_count=int(stats.count[i]),
            halting_count=int(stats.halting[i]),
            mean_speed=None if empty else mean,
            travel_time=length / limit if empty else length / max(mean, C.MIN_EFFECTIVE_SPEED),
        )


# ------------------------------------------------------------------------------- state
class StateArrays:
    """``sim.state``: running vehicles as read-only arrays in slot order (AA 5.4, 5.5).

    Rows are aligned across fields; key them by :attr:`uids`. Arrays are copies computed
    once per step; :attr:`raw` is the live table.
    """

    def __init__(self, engine: Engine, cache: StepCache) -> None:
        self._engine = engine
        self._cache = cache

    def _running(self) -> IntArray:
        return _cached(self._cache, "state_running", lambda: self._engine.vehicles.running())

    def _column(self, column: str) -> NDArray[Any]:
        def build() -> NDArray[Any]:
            return _frozen(getattr(self._engine.vehicles, column)[self._running()])

        return _cached(self._cache, f"state_{column}", build)

    def _geometry(self) -> tuple[FloatArray, FloatArray]:
        def build() -> tuple[FloatArray, FloatArray]:
            net, veh = self._engine.network, self._engine.vehicles
            run = self._running()
            xy, heading = vehicle_xy(net, veh.link[run], veh.pos[run], veh.lat_offset[run])
            return _frozen(xy), _frozen(heading)

        return _cached(self._cache, "state_geometry", build)

    @property
    def uids(self) -> UIntArray:
        return self._column("uid")

    @property
    def type_idx(self) -> UIntArray:
        """Vehicle type index (``sim.network.compiled.vehicle_types`` order)."""
        return self._column("type_idx")

    @property
    def links(self) -> IntArray:
        """Compiled link index (lanes, then connectors)."""
        return self._column("link")

    @property
    def positions(self) -> FloatArray:
        """Front-bumper position along the link, m."""
        return self._column("pos")

    @property
    def speeds(self) -> FloatArray:
        return self._column("speed")

    @property
    def accels(self) -> FloatArray:
        return self._column("accel")

    @property
    def xy(self) -> FloatArray:
        """World coordinates of the front bumpers, shape ``(N, 2)``."""
        return self._geometry()[0]

    @property
    def headings(self) -> FloatArray:
        """Radians, counter-clockwise from +x."""
        return self._geometry()[1]

    @property
    def waiting_times(self) -> FloatArray:
        """Trip-cumulative halting time, s."""
        return self._column("waiting_time")

    @property
    def distances(self) -> FloatArray:
        """Distance travelled since insertion, m."""
        return self._column("distance")

    @property
    def ids(self) -> list[str]:
        """Vehicle ids aligned with the arrays (built on first use)."""
        veh = self._engine.vehicles
        return _cached(
            self._cache, "state_ids", lambda: [str(veh.ids[h]) for h in self._running().tolist()]
        )

    @property
    def raw(self) -> Mapping[str, NDArray[Any]]:
        """Every vehicle-table column as a zero-copy, read-only, capacity-sized live view.

        Rows are engine slots; ``raw["active"]`` marks the running ones. Views may be stale
        after the next step (the table grows by reallocation).
        """
        veh = self._engine.vehicles
        out: dict[str, NDArray[Any]] = {}
        for name in COLUMNS:
            view = getattr(veh, name).view()
            view.flags.writeable = False
            out[name] = view
        return out


# ------------------------------------------------------------------------------- signals
_STATE_CHARS: Final = tuple(s.value for s in SignalState)  # code -> "r", "y", "g", "G"


@dataclass(frozen=True, slots=True)
class SignalView:
    """Snapshot of the signal of one intersection (plan H.2, AA 5.4)."""

    phase_index: int
    """Current phase; during yellow and all-red, the phase being left."""
    phase_id: str
    stage: Stage
    """``green``, ``yellow`` or ``all_red``."""
    target: int
    """The phase a transition leads to; -1 in green."""
    stage_elapsed: float
    """Time the current stage has been applied, s."""
    green_elapsed: float
    """Time since the current phase's green started (counts on through its transition), s."""
    remaining: float | None
    """``fixed_time``: seconds until the current stage ends (the stage lengths quantised to
    whole steps); None for other controllers."""
    min_green: float
    """Of the current phase, s."""
    max_green: float
    cycle: float | None
    """``fixed_time``: the realised (quantised) cycle ``C_q``, s; None otherwise."""
    held: bool
    """A manual hold (``hold_phase``) is active."""
    movement_states: Mapping[str, str]
    """Movement id -> ``"G"``, ``"g"``, ``"y"`` or ``"r"``, in canonical order."""
    state_string: str
    """SUMO-style: one character per movement, in canonical order."""
    controller: str
    """Name of the active controller (``external`` during a manual hold)."""
    preempting: bool = False
    """A ``preemption`` controller is serving an emergency vehicle."""


@dataclass(frozen=True, slots=True)
class PhaseInfo:
    """One phase of a signal program (AA 5.4)."""

    index: int
    id: str
    green: Mapping[str, str]
    """Movement id -> ``"G"`` (protected) or ``"g"`` (permissive); the others are red."""
    duration: float
    """Fixed-time green, s."""
    min_green: float
    max_green: float


def signal_view(engine: Engine, j: int) -> SignalView:
    """The :class:`SignalView` of signalised intersection ``j`` (a global index)."""
    sig, net = engine.signals, engine.network
    prog = sig.program(j)
    p = int(sig.phase[j])
    stage = Stage.from_code(int(sig.stage[j]))
    controller = engine.controllers[j]
    preempting = isinstance(controller, Preemption) and controller.active
    timing = controller.inner if isinstance(controller, Preemption) else controller
    remaining = cycle = None
    if isinstance(timing, FixedTime) and not preempting:
        dt = engine.config.dt
        timed = timing.timing or prog
        length = {
            Stage.green: max(float(timed.duration[p]), float(timed.min_green[p])),
            Stage.yellow: prog.yellow,
            Stage.all_red: prog.all_red,
        }[stage]
        remaining = max(0.0, stage_steps(length, dt) * dt - float(sig.stage_elapsed[j]))
        cycle = realised_cycle(timed, dt)
    codes = sig.movement_state[prog.movements].tolist()
    return SignalView(
        phase_index=p,
        phase_id=prog.phase_ids[p],
        stage=stage,
        target=int(sig.target[j]),
        stage_elapsed=float(sig.stage_elapsed[j]),
        green_elapsed=float(sig.green_elapsed[j]),
        remaining=remaining,
        min_green=float(prog.min_green[p]),
        max_green=float(prog.max_green[p]),
        cycle=cycle,
        held=bool(engine.held[j]),
        movement_states={
            net.mov_ids[m]: _STATE_CHARS[c]
            for m, c in zip(prog.movements.tolist(), codes, strict=True)
        },
        state_string="".join(_STATE_CHARS[c] for c in codes),
        controller=controller_name(controller),
        preempting=preempting,
    )


def phase_infos(net: CompiledNetwork, prog: SignalProgram) -> tuple[PhaseInfo, ...]:
    """The phases of ``prog`` in program order."""
    movements = [net.mov_ids[m] for m in prog.movements.tolist()]
    return tuple(
        PhaseInfo(
            index=p,
            id=prog.phase_ids[p],
            green={
                m: _STATE_CHARS[c]
                for m, c in zip(movements, prog.phase_state[p].tolist(), strict=True)
                if c >= SignalState.g.code
            },
            duration=float(prog.duration[p]),
            min_green=float(prog.min_green[p]),
            max_green=float(prog.max_green[p]),
        )
        for p in range(prog.n_phases)
    )


# ------------------------------------------------------------------------------- intersections
@dataclass(frozen=True, slots=True)
class MovementInfo:
    """One movement of an intersection (AA 5.4)."""

    id: str
    from_road: str
    to_road: str
    turn: TurnKind
    rank: int
    """Right of way now: the signal state code at a signalised intersection (G 3, g 2,
    y 1, r 0), else the static rank (priority 3/2/1, uncontrolled 1)."""
    connections: tuple[str, ...]
    """Connector ids, in connection order."""
    state: str | None
    """``"G"``, ``"g"``, ``"y"`` or ``"r"``; None at an unsignalised intersection."""


@dataclass(frozen=True, slots=True)
class IntersectionView:
    """Snapshot of one intersection or boundary node (AA 5.4)."""

    id: str
    kind: IntersectionKind
    point: tuple[float, float]
    """Centre, world coordinates."""
    movements: tuple[MovementInfo, ...]
    """In canonical order."""
    signal: SignalView | None
    """None unless the intersection is signalised."""


class IntersectionCollection(_Indexed[IntersectionView]):
    """``sim.intersections``: every node (boundaries included) in compiled order."""

    kind = "intersection"

    def __init__(self, engine: Engine, cache: StepCache) -> None:
        super().__init__(engine, cache, engine.network.int_ids)

    def _make(self, i: int) -> IntersectionView:
        eng = self._engine
        net = eng.network
        signalised = i in eng.controllers
        movements = []
        for m in np.flatnonzero(net.mov_intersection == i).tolist():
            code = int(eng.signals.movement_state[m])
            conns = net.mov_conn[net.mov_conn_ptr[m] : net.mov_conn_ptr[m + 1]].tolist()
            movements.append(
                MovementInfo(
                    id=net.mov_ids[m],
                    from_road=net.road_ids[int(net.mov_from_road[m])],
                    to_road=net.road_ids[int(net.mov_to_road[m])],
                    turn=TurnKind.from_code(int(net.mov_turn[m])),
                    rank=code if signalised else int(net.mov_static_rank[m]),
                    connections=tuple(net.link_ids[c] for c in conns),
                    state=_STATE_CHARS[code] if signalised else None,
                )
            )
        x, y = (net.int_point[i] + net.origin).tolist()
        return IntersectionView(
            id=self.ids[i],
            kind=IntersectionKind.from_code(int(net.int_kind[i])),
            point=(x, y),
            movements=tuple(movements),
            signal=signal_view(eng, i) if signalised else None,
        )


# ------------------------------------------------------------------------------- network
_KINDS: Final = ("lane", "connection", "link", "road", "intersection", "movement")


class NetworkInfo:
    """``sim.network``: ids, static arrays, render geometry and the road graph (AA 5.4)."""

    def __init__(self, network: CompiledNetwork, graph: nx.DiGraph) -> None:
        self._net = network
        self._graph = graph
        n = network.n_lanes
        self.lane_ids: tuple[str, ...] = network.link_ids[:n]
        self.connection_ids: tuple[str, ...] = network.link_ids[n:]
        """Connector ids ``"{lane_in}->{lane_out}"`` (link index = n_lanes + position)."""
        self.road_ids: tuple[str, ...] = network.road_ids
        self.intersection_ids: tuple[str, ...] = network.int_ids
        self.movement_ids: tuple[str, ...] = network.mov_ids
        self.lane_length: FloatArray = network.link_length[:n]
        self.lane_speed_limit: FloatArray = network.link_speed_limit[:n]
        self.lane_road: IntArray = network.link_road[:n]
        self.link_kind: UIntArray = network.link_kind
        """``LinkKind`` code of every link: 0 lane, 1 connector."""

    @property
    def compiled(self) -> CompiledNetwork:
        """The read-only compiled network (plan E.3)."""
        return self._net

    @property
    def n_lanes(self) -> int:
        return self._net.n_lanes

    def index(self, kind: str, item_id: str) -> int:
        """Index of ``item_id`` among the ids of ``kind``: lane, connection, link (lanes
        then connectors), road, intersection or movement."""
        net = self._net
        ids = {
            "lane": self.lane_ids,
            "connection": self.connection_ids,
            "link": net.link_ids,
            "road": net.road_ids,
            "intersection": net.int_ids,
            "movement": net.mov_ids,
        }.get(kind)
        if ids is None:
            raise NotFoundError(
                f'unknown kind "{kind}"{suggest(kind, _KINDS)} (available: {", ".join(_KINDS)})'
            )
        maps = {"road": net.road_index, "intersection": net.int_index, "movement": net.mov_index}
        i = maps.get(kind, net.link_index).get(item_id, -1)
        if kind == "connection":
            i -= net.n_lanes
        if not 0 <= i < len(ids):
            raise NotFoundError(f'unknown {kind} "{item_id}"{suggest(item_id, ids)}')
        return i

    def geometry(self) -> dict[str, Any]:
        """Render geometry (local coordinates plus ``origin``) as a JSON-ready dict."""
        return render_geometry(self._net).model_dump(mode="json")

    def graph(self) -> nx.DiGraph:
        """A copy of the road graph (nodes: road indices; edges: movements)."""
        return self._graph.copy()
