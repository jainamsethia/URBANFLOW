"""The metrics manager (plan J): streaming accumulators, sampled tables and the summary.

One :class:`MetricsManager` per simulation; :meth:`update` runs after every engine step.

Tables (plain column dicts of numpy arrays; ``time`` is the sample time, ``window_s`` the
window length since the previous sample):

* ``timeseries`` (global, one row per sample): active, backlog, generated, arrived,
  inserted_w, arrived_w, halting, speed_mean (instantaneous), vkt_w, vht_w,
  space_mean_speed, throughput_vph, waiting_w (veh*s), queue_mean / queue_max (time-mean
  and max of the network's total stop-line queue over the window, veh).
* ``intersections`` (one row per sample and non-boundary intersection): crossings_w,
  throughput_vph (stop-line crossings), queue_mean / queue_max (sum of incoming lanes' J.3
  queues), phase (-1 unsignalised).
* ``trips`` (one row per arrived vehicle): ids, origin/destination roads, times, delay
  (travel time - free-flow time), waiting time, stops, distance.

The summary (J.6, flattened keys, B.2 #13) excludes trips departing before
``metrics.warmup`` and windows before it (J.1).

ponytail: a fixed built-in collector set (no collector plugins, no lane table, no energy
proxy); add the J.7 collector protocol when custom metrics are needed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.events import EventType
from urbanflow.core.types import FloatArray, IntArray, VehicleStatus
from urbanflow.engine.signalling import stopline_queue

if TYPE_CHECKING:
    from urbanflow.engine.engine import Engine

__all__ = ["METRIC_DIRECTIONS", "MetricsManager", "Table"]

Table = dict[str, np.ndarray]
"""A table: column name -> 1-D array (string columns are object-free ``<U`` arrays)."""

_ARRIVED: Final = EventType.vehicle_arrived.code
_INSERTED: Final = EventType.vehicle_inserted.code
_ENTERED: Final = EventType.vehicle_entered_link.code
_WAITING: Final = VehicleStatus.waiting_insert.code
_RUNNING: Final = VehicleStatus.running.code

GLOBAL_COLUMNS: Final = (
    "time", "window_s", "active", "backlog", "generated", "arrived", "inserted_w",
    "arrived_w", "halting", "speed_mean", "vkt_w", "vht_w", "space_mean_speed",
    "throughput_vph", "waiting_w", "queue_mean", "queue_max",
)  # fmt: skip
INTERSECTION_COLUMNS: Final = (
    "time", "window_s", "intersection", "crossings_w", "throughput_vph", "queue_mean",
    "queue_max", "phase",
)  # fmt: skip
TRIP_COLUMNS: Final = (
    "uid", "vehicle_id", "type", "origin", "destination", "depart_time", "insert_time",
    "arrive_time", "travel_time", "insertion_delay", "total_time", "delay", "waiting_time",
    "stops", "distance_m",
)  # fmt: skip

METRIC_DIRECTIONS: Final[Mapping[str, Literal["min", "max"]]] = {
    "travel_time.mean": "min",
    "travel_time.p95": "min",
    "delay.mean": "min",
    "waiting_time_mean": "min",
    "stops_mean": "min",
    "att_censored": "min",
    "queue.mean_total_veh": "min",
    "throughput_vph": "max",
    "space_mean_speed": "max",
    "vehicles.arrived": "max",
}
"""Better direction of the summary metrics used in comparisons (J.6)."""


class MetricsManager:
    """``sim.metrics``: accumulates every step; tables, summary, latest values, history."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        net = engine.network
        self._junctions: IntArray = np.flatnonzero(
            np.asarray(net.int_kind) != _boundary_code()
        ).astype(np.intp)
        self._lane_int: IntArray = net.link_intersection[: net.n_lanes].astype(np.intp)
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Forget everything (called on simulation reset)."""
        n_int = self._engine.network.n_intersections
        self._counts = np.zeros(len(EventType), dtype=np.int64)
        self._trips: list[dict[str, np.ndarray]] = []
        self._d_done = 0.0  # distance of arrived vehicles, m
        self._global: list[tuple[float, ...]] = []
        self._ints: list[np.ndarray] = []
        # window accumulators
        self._w_steps = 0
        self._w_start = self._engine.time
        self._w_d0 = 0.0
        self._w_vht = 0.0
        self._w_wait = 0.0
        self._w_arr = 0
        self._w_ins = 0
        self._w_qsum = 0.0
        self._w_qmax = 0
        self._w_cross = np.zeros(n_int, dtype=np.int64)
        self._w_iq_sum = np.zeros(n_int, dtype=np.float64)
        self._w_iq_max = np.zeros(n_int, dtype=np.int64)
        # post-warm-up totals for the summary
        self._s_vht = 0.0
        self._s_d0: float | None = None
        self._s_qsum = 0.0
        self._s_steps = 0
        self._s_lane_qmax = 0

    def state_dict(self) -> dict[str, Any]:
        """A deep copy of every accumulator (in-memory snapshots)."""
        return copy.deepcopy({k: v for k, v in vars(self).items() if k != "_engine"})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Replace the accumulators by a copy of :meth:`state_dict` output."""
        vars(self).update(copy.deepcopy(dict(state)))

    # ------------------------------------------------------------------ per step
    def update(self) -> None:
        """Account the step that just finished."""
        eng = self._engine
        net, veh, ev, dt = eng.network, eng.vehicles, eng.events, eng.config.dt
        kind = ev.type
        self._counts += np.bincount(kind, minlength=len(EventType))[: len(EventType)]
        arrived = kind == _ARRIVED
        n_arr = int(np.count_nonzero(arrived))
        if n_arr:
            h = ev.handle[arrived].astype(np.intp)  # rows stay intact until the next step
            self._record_trips(h, ev.time[arrived])
            self._d_done += float(veh.distance[h].sum())
        entered = (kind == _ENTERED) & (ev.link >= net.n_lanes)
        if entered.any():
            ints = net.link_intersection[ev.link[entered]]
            self._w_cross += np.bincount(ints, minlength=self._w_cross.size)

        run = np.flatnonzero(veh.active)
        link = veh.link[run].astype(np.intp)
        halt = veh.halting[run]
        lane_len = net.link_length[: net.n_lanes]
        q = stopline_queue(link, veh.pos[run], veh.uid[run], halt, lane_len)
        q_int = np.bincount(self._lane_int, weights=q, minlength=self._w_cross.size)
        q_total = int(q.sum())
        vht = run.size * dt / C.SECONDS_PER_HOUR

        self._w_steps += 1
        self._w_vht += vht
        self._w_wait += float(np.count_nonzero(halt)) * dt
        self._w_arr += n_arr
        self._w_ins += int(np.count_nonzero(kind == _INSERTED))
        self._w_qsum += q_total
        self._w_qmax = max(self._w_qmax, q_total)
        self._w_iq_sum += q_int
        np.maximum(self._w_iq_max, q_int.astype(np.int64), out=self._w_iq_max)

        if eng.time > eng.config.metrics.warmup - C.TIME_EPS:
            d_now = self._distance()
            if self._s_d0 is None:
                self._s_d0 = d_now
            self._s_vht += vht
            self._s_qsum += q_total
            self._s_steps += 1
            self._s_lane_qmax = max(self._s_lane_qmax, int(q.max()) if q.size else 0)

        every = max(1, round(eng.config.metrics.interval / dt))
        if eng.step_count % every == 0:
            self._sample(run, halt)

    def _distance(self) -> float:
        """``D(t)``: distance driven by running plus arrived vehicles, m."""
        veh = self._engine.vehicles
        return float(veh.distance[veh.active].sum()) + self._d_done

    def _record_trips(self, h: IntArray, arrive: FloatArray) -> None:
        eng = self._engine
        veh, net, routes, types = eng.vehicles, eng.network, eng.routes.routes, eng.types
        rid = veh.route_id[h].tolist()
        depart, insert = veh.depart_time[h].copy(), veh.insert_time[h].copy()
        travel = arrive - insert
        self._trips.append(
            {
                "uid": veh.uid[h].astype(np.int64),
                "vehicle_id": np.array([veh.ids[i] or "" for i in h.tolist()]),
                "type": np.array([types.ids[t] for t in veh.type_idx[h].tolist()]),
                "origin": np.array([net.road_ids[int(routes[r][0])] for r in rid]),
                "destination": np.array([net.road_ids[int(routes[r][-1])] for r in rid]),
                "depart_time": depart,
                "insert_time": insert,
                "arrive_time": arrive.astype(np.float64),
                "travel_time": travel,
                "insertion_delay": insert - depart,
                "total_time": arrive - depart,
                "delay": travel - veh.ff_time[h],
                "waiting_time": veh.waiting_time[h].astype(np.float64),
                "stops": veh.stops[h].astype(np.int64),
                "distance_m": veh.distance[h].astype(np.float64),
            }
        )

    def _sample(self, run: IntArray, halt: np.ndarray) -> None:
        eng = self._engine
        veh = eng.vehicles
        t = eng.time
        window = max(t - self._w_start, C.TIME_EPS)
        steps = max(self._w_steps, 1)
        d_now = self._distance()
        vkt = (d_now - self._w_d0) / C.METRES_PER_KM
        speed = veh.speed[run]
        hours = window / C.SECONDS_PER_HOUR
        self._global.append(
            (
                t,
                window,
                float(run.size),
                float(eng.backlog),
                float(eng.generated),
                float(eng.arrived),
                float(self._w_ins),
                float(self._w_arr),
                float(np.count_nonzero(halt)),
                float(speed.mean()) if speed.size else math.nan,
                vkt,
                self._w_vht,
                vkt * C.METRES_PER_KM / (self._w_vht * C.SECONDS_PER_HOUR)
                if self._w_vht > 0
                else math.nan,
                self._w_arr / hours,
                self._w_wait,
                self._w_qsum / steps,
                float(self._w_qmax),
            )
        )
        j = self._junctions
        rows = np.empty((j.size, len(INTERSECTION_COLUMNS)), dtype=np.float64)
        rows[:, 0], rows[:, 1], rows[:, 2] = t, window, j
        rows[:, 3] = self._w_cross[j]
        rows[:, 4] = self._w_cross[j] / hours
        rows[:, 5] = self._w_iq_sum[j] / steps
        rows[:, 6] = self._w_iq_max[j]
        rows[:, 7] = eng.signals.phase[j]
        self._ints.append(rows)
        self._w_steps, self._w_start, self._w_d0 = 0, t, d_now
        self._w_vht = self._w_wait = self._w_qsum = 0.0
        self._w_arr = self._w_ins = self._w_qmax = 0
        self._w_cross[:] = 0
        self._w_iq_sum[:] = 0.0
        self._w_iq_max[:] = 0

    # ------------------------------------------------------------------ tables
    def timeseries(self) -> Table:
        """The global timeseries table."""
        data = np.array(self._global, dtype=np.float64).reshape(-1, len(GLOBAL_COLUMNS))
        return {name: data[:, i].copy() for i, name in enumerate(GLOBAL_COLUMNS)}

    def intersections(self) -> Table:
        """The per-intersection timeseries table (``intersection`` holds ids)."""
        width = len(INTERSECTION_COLUMNS)
        data = np.concatenate(self._ints) if self._ints else np.zeros((0, width))
        table = {name: data[:, i].copy() for i, name in enumerate(INTERSECTION_COLUMNS)}
        ids = np.array(self._engine.network.int_ids)
        table["intersection"] = ids[data[:, 2].astype(np.intp)] if ids.size else ids
        table["phase"] = table["phase"].astype(np.int64)
        return table

    def trips(self) -> Table:
        """One row per arrived vehicle, in arrival order."""
        if not self._trips:
            return {
                name: np.zeros(0, dtype=np.float64 if name not in _STR_COLS else "<U1")
                for name in TRIP_COLUMNS
            }
        return {name: np.concatenate([t[name] for t in self._trips]) for name in TRIP_COLUMNS}

    def tables(self) -> dict[str, Table]:
        """Every table by name (``timeseries``, ``intersections``, ``trips``)."""
        return {
            "timeseries": self.timeseries(),
            "intersections": self.intersections(),
            "trips": self.trips(),
        }

    def latest(self) -> dict[str, float]:
        """The last global sample (empty before the first one)."""
        if not self._global:
            return {}
        return dict(zip(GLOBAL_COLUMNS, self._global[-1], strict=True))

    def history(self, max_points: int = C.METRICS_HISTORY_POINTS) -> Table:
        """The global timeseries, thinned to at most ``max_points`` evenly spaced rows."""
        table = self.timeseries()
        n = table["time"].size
        if n <= max_points:
            return table
        keep = np.unique(np.linspace(0, n - 1, max_points).round().astype(np.intp))
        return {k: v[keep] for k, v in table.items()}

    # ------------------------------------------------------------------ facade helpers
    def arrived_uids(self) -> IntArray:
        """Uids of the arrived vehicles, in arrival order."""
        if not self._trips:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([t["uid"] for t in self._trips])

    def event_counts(self) -> dict[str, int]:
        """Events per type name, cumulative since reset."""
        return {t.value: int(n) for t, n in zip(EventType, self._counts.tolist(), strict=True)}

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict[str, float]:
        """Flattened J.6 summary (B.2 #13); trips departing before the warm-up and windows
        before it are excluded (J.1)."""
        eng = self._engine
        warmup = eng.config.metrics.warmup
        trips = self.trips()
        keep = trips["depart_time"] >= warmup
        travel, delay = trips["travel_time"][keep], trips["delay"][keep]
        total = trips["total_time"][keep]
        arrive = trips["arrive_time"]
        window = eng.time - warmup
        veh = eng.vehicles
        pending = np.isin(veh.status, (_WAITING, _RUNNING)) & (veh.depart_time >= warmup)
        censored = np.concatenate([total, eng.time - veh.depart_time[pending]])
        vkt = (self._distance() - self._s_d0) / C.METRES_PER_KM if self._s_d0 is not None else 0.0
        return {
            "vehicles.generated": eng.generated,
            "vehicles.inserted": int(self._counts[_INSERTED]),
            "vehicles.arrived": eng.arrived,
            "vehicles.en_route": int(np.count_nonzero(veh.active)),
            "vehicles.backlog": eng.backlog,
            "vehicles.removed": eng.removed,
            "vehicles.teleported": eng.teleported,
            "travel_time.mean": _mean(travel),
            "travel_time.median": _pct(travel, 50),
            "travel_time.p95": _pct(travel, C.SUMMARY_PERCENTILE),
            "travel_time.std": float(travel.std()) if travel.size else math.nan,
            "total_time_mean": _mean(total),
            "insertion_delay_mean": _mean(trips["insertion_delay"][keep]),
            "att_censored": _mean(censored),
            "delay.mean": _mean(delay),
            "delay.median": _pct(delay, 50),
            "delay.p95": _pct(delay, C.SUMMARY_PERCENTILE),
            "waiting_time_mean": _mean(trips["waiting_time"][keep]),
            "stops_mean": _mean(trips["stops"][keep].astype(np.float64)),
            "throughput_vph": (
                C.SECONDS_PER_HOUR * int(np.count_nonzero(arrive >= warmup)) / window
                if window > 0
                else math.nan
            ),
            "space_mean_speed": (
                vkt * C.METRES_PER_KM / (self._s_vht * C.SECONDS_PER_HOUR)
                if self._s_vht > 0
                else math.nan
            ),
            "vkt": vkt,
            "vht": self._s_vht,
            "queue.mean_total_veh": self._s_qsum / self._s_steps if self._s_steps else math.nan,
            "queue.max_lane_veh": float(self._s_lane_qmax),
        }


_STR_COLS: Final = frozenset({"vehicle_id", "type", "origin", "destination"})


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else math.nan


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else math.nan


def _boundary_code() -> int:
    from urbanflow.core.types import IntersectionKind

    return IntersectionKind.boundary.code
