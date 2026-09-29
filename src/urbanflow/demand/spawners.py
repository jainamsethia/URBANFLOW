"""Arrival processes: flows and trips become :class:`SpawnRequest` s (plan E.7 §1.3, I.3).

Per flow, headway ``h = 3600 / rate`` (or ``period``):

* uniform: ``t_k = begin + k h`` (computed directly, never accumulated);
* poisson: ``t_k = t_{k-1} + E_k``, ``E_k ~ Exp(rate / 3600)``, ``t_{-1} = begin``;
* binomial: one vehicle with probability ``p = rate dt / 3600`` in every step ``n`` whose
  time ``n dt`` lies in ``[begin, end)``;

while ``t_k < end`` and ``k < count``. A vehicle scheduled at ``t_k`` is emitted in step
``ceil(t_k / dt - 1e-9)``. Every random draw of a flow comes from its own stream
``flow:{id}`` (trips: ``trip:{id}``), in the fixed per-vehicle order type (``type_mix``)
-> route (``routes``) -> speed factor -> ``random`` depart lane, so adding or changing one
flow never changes another flow's vehicles (common random numbers, AG.1 R11). A poisson
flow draws its next inter-arrival gap after each vehicle's attributes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from urbanflow.core.constants import SECONDS_PER_HOUR, TIME_EPS
from urbanflow.core.errors import UrbanFlowError
from urbanflow.core.rng import RngStreams
from urbanflow.core.types import FloatArray
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.routing.base import Router, RouteTable
from urbanflow.routing.lanes import valid_mask
from urbanflow.scenario.schema import DemandSpec, FlowSpec, TripSpec
from urbanflow.vehicles.table import DEPART_LANE_BEST, DEPART_LANE_FIRST
from urbanflow.vehicles.types import VehicleTypes

__all__ = [
    "FlowSpawner",
    "SpawnRequest",
    "Spawner",
    "TripSchedule",
    "api_request",
    "build_spawners",
    "emission_step",
]


@dataclass(frozen=True, slots=True)
class SpawnRequest:
    """One vehicle to create in the current step (it then waits for insertion)."""

    vehicle_id: str
    """``"{flow}.{k}"`` or the trip id."""
    source_idx: int
    """Flow index, or ``len(flows) + trip index`` (-1 for API vehicles)."""
    type_idx: int
    route_id: int
    speed_factor: float
    depart_lane: int
    """``DEPART_LANE_BEST``, ``DEPART_LANE_FIRST`` or a lane index (``random`` is drawn)."""
    depart_speed: float
    """m/s; NaN = ``"max"``."""
    depart_time: float
    """Scheduled time, s (the vehicle is emitted in step ``emission_step(depart_time)``)."""


def emission_step(t: float, dt: float) -> int:
    """The step in which a vehicle scheduled at ``t`` is emitted: ``ceil(t/dt - 1e-9)``."""
    return math.ceil(t / dt - TIME_EPS)


class _Vehicles:
    """Route resolution and per-vehicle draws shared by flows and trips (F.6 order)."""

    def __init__(
        self,
        net: CompiledNetwork,
        routes: RouteTable,
        router: Router,
        types: VehicleTypes,
        item: FlowSpec | TripSpec,
    ) -> None:
        self.net, self.routes, self.router, self.types = net, routes, router, types
        self.lane_mode = item.depart_lane
        speed = item.depart_speed
        self.depart_speed = math.nan if speed == "max" else float(speed)
        self.od: tuple[int, int] | None = None
        if item.origin is not None and item.destination is not None:
            self.od = (self.road(item.origin), self.road(item.destination))
        self.via = tuple(self.road(v) for v in item.via)

    def road(self, road_id: str) -> int:
        return self.net.road_index[road_id]

    def intern(self, road_ids: Sequence[str]) -> int:
        return self.routes.intern([self.road(r) for r in road_ids])

    def resolve(self, route_id: int) -> int:
        """``route_id``, or for origin/destination items the router's path (at departure)."""
        if self.od is None:
            return route_id
        return self.routes.intern(self.router.route(*self.od, self.via))

    def request(
        self,
        rng: np.random.Generator,
        vehicle_id: str,
        source_idx: int,
        type_idx: int,
        route_id: int,
        t: float,
    ) -> SpawnRequest:
        """Draw the speed factor, then the ``random`` depart lane, and build the request."""
        sf = self.types.specs[type_idx].speed_factor
        factor = float(np.clip(rng.normal(sf.mean, sf.std), sf.min, sf.max))
        mode = self.lane_mode
        if mode == "random":  # uniform over the lanes that can reach the next road
            route = self.routes.get(route_id)
            first = int(route[0])
            mask = valid_mask(self.net, first, int(route[1]) if len(route) > 1 else -1)
            valid = [k for k in range(int(self.net.road_n_lanes[first])) if mask >> k & 1]
            lane = valid[int(rng.integers(len(valid)))]
        elif mode == "best":
            lane = DEPART_LANE_BEST
        elif mode == "first":
            lane = DEPART_LANE_FIRST
        else:
            lane = int(mode)
        return SpawnRequest(
            vehicle_id, source_idx, type_idx, route_id, factor, lane, self.depart_speed, t
        )


class FlowSpawner:
    """Arrivals of one flow (uniform, poisson or binomial); see the module docstring."""

    def __init__(
        self,
        flow: FlowSpec,
        source_idx: int,
        *,
        net: CompiledNetwork,
        routes: RouteTable,
        router: Router,
        types: VehicleTypes,
        rng: np.random.Generator,
        dt: float,
    ) -> None:
        self.flow = flow
        self.source_idx = source_idx
        self.rng = rng
        self.dt = dt
        self.count = 0
        """Vehicles emitted so far (``k`` of the next vehicle)."""
        self.exhausted = False
        """True once no further vehicle can be emitted (past ``end`` or ``count``)."""
        self._v = _Vehicles(net, routes, router, types, flow)
        self.headway = flow.period or SECONDS_PER_HOUR / float(flow.rate or 1)
        self._end = math.inf if flow.end is None else flow.end
        self._cap = math.inf if flow.count is None else flow.count
        mix = flow.type_mix or {flow.vehicle_type or "car": 1.0}
        self._types = [types.type_index(t) for t in mix]
        self._type_p = _weights(list(mix.values())) if flow.type_mix else None
        choices = flow.routes or ()
        self._routes = [self._v.intern(r.roads) for r in choices]
        self._route_p = _weights([r.weight for r in choices]) if choices else None
        if flow.route is not None:
            self._routes = [self._v.intern(flow.route)]
        self.next_time = math.nan
        """Scheduled time of the next vehicle, s (uniform and poisson)."""
        self._step, self._end_step = 0, math.inf  # binomial only
        if flow.arrival == "binomial":
            self._p = dt / self.headway  # rate * dt / 3600 (<= 1 by E506)
            self._step = emission_step(flow.begin, dt)
            self._end_step = math.inf if flow.end is None else emission_step(flow.end, dt)
        elif flow.arrival == "poisson":
            self.next_time = flow.begin + float(rng.exponential(self.headway))
        else:
            self.next_time = flow.begin
        self._update_exhausted()

    def _update_exhausted(self) -> None:
        if self.flow.arrival == "binomial":
            past_end = self._step >= self._end_step
        else:
            past_end = self.next_time >= self._end
        self.exhausted = past_end or self.count >= self._cap

    def due(self, step: int) -> list[SpawnRequest]:
        """Vehicles emitted in ``step`` (and in any earlier step not asked for yet)."""
        out: list[SpawnRequest] = []
        if self.flow.arrival == "binomial":
            while not self.exhausted and self._step <= step:
                n = self._step
                self._step += 1
                if self.rng.random() < self._p:
                    out.append(self._spawn(n * self.dt))
                self._update_exhausted()
            return out
        while not self.exhausted and emission_step(self.next_time, self.dt) <= step:
            out.append(self._spawn(self.next_time))
            if self.flow.arrival == "uniform":
                self.next_time = self.flow.begin + self.count * self.headway
            elif self.count < self._cap:
                self.next_time += float(self.rng.exponential(self.headway))
            self._update_exhausted()
        return out

    def _spawn(self, t: float) -> SpawnRequest:
        rng = self.rng
        type_idx = self._types[0]
        if self._type_p is not None:
            type_idx = self._types[int(rng.choice(len(self._types), p=self._type_p))]
        route_id = self._routes[0] if self._routes else -1
        if self._route_p is not None:
            route_id = self._routes[int(rng.choice(len(self._routes), p=self._route_p))]
        route_id = self._v.resolve(route_id)
        k = self.count
        self.count += 1
        return self._v.request(rng, f"{self.flow.id}.{k}", self.source_idx, type_idx, route_id, t)


class TripSchedule:
    """All trips, emitted in (emission step, file order); each draws from ``trip:{id}``."""

    def __init__(
        self,
        trips: Sequence[TripSpec],
        first_source_idx: int,
        *,
        net: CompiledNetwork,
        routes: RouteTable,
        router: Router,
        types: VehicleTypes,
        rng: RngStreams,
        dt: float,
    ) -> None:
        self.trips = tuple(trips)
        self._rng = rng
        self._first = first_source_idx
        steps = [emission_step(t.depart, dt) for t in self.trips]
        self._order = sorted(range(len(steps)), key=lambda i: (steps[i], i))
        self._steps = [steps[i] for i in self._order]
        self._next = 0
        self._v = [_Vehicles(net, routes, router, types, t) for t in self.trips]
        self._types = [types.type_index(t.vehicle_type) for t in self.trips]
        self._routes = [
            v.intern(t.route) if t.route else -1 for v, t in zip(self._v, self.trips, strict=True)
        ]

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self._order)

    def due(self, step: int) -> list[SpawnRequest]:
        """Trips emitted in ``step`` (and in any earlier step not asked for yet)."""
        out: list[SpawnRequest] = []
        while not self.exhausted and self._steps[self._next] <= step:
            i = self._order[self._next]
            self._next += 1
            trip, v = self.trips[i], self._v[i]
            rng = self._rng.stream(f"trip:{trip.id}")
            route_id = v.resolve(self._routes[i])
            source = self._first + i
            out.append(v.request(rng, trip.id, source, self._types[i], route_id, trip.depart))
        return out


Spawner = FlowSpawner | TripSchedule


def api_request(
    trip: TripSpec,
    *,
    net: CompiledNetwork,
    routes: RouteTable,
    router: Router,
    types: VehicleTypes,
    rng: np.random.Generator,
) -> SpawnRequest:
    """The :class:`SpawnRequest` of an API vehicle (``EngineCommands.add_vehicle``).

    ``trip`` must reference existing roads and a known type (the caller validates them);
    its ``depart`` is the current time. The speed factor and ``random`` lane are drawn from
    ``rng`` (the ``vehicle_params`` stream) in the F.6 order; ``source_idx`` is -1.
    """
    v = _Vehicles(net, routes, router, types, trip)
    route_id = v.resolve(v.intern(trip.route) if trip.route else -1)
    type_idx = types.type_index(trip.vehicle_type)
    return v.request(rng, trip.id, -1, type_idx, route_id, trip.depart)


def _weights(values: Sequence[float]) -> FloatArray:
    w = np.asarray(values, dtype=np.float64)
    return w / w.sum()


def build_spawners(
    demand: DemandSpec,
    net: CompiledNetwork,
    routes: RouteTable,
    router: Router,
    types: VehicleTypes,
    rng: RngStreams,
    dt: float,
) -> list[Spawner]:
    """Spawners of the *resolved* ``demand``: one per flow (file order), then the trips.

    Transit lines arrive in a later version and are rejected with a friendly error.
    """
    if demand.transit:
        ids = ", ".join(f'"{line.id}"' for line in demand.transit)
        raise UrbanFlowError(
            f"demand.transit: transit lines ({ids}) are not supported by this version of "
            "the engine yet; remove them from the scenario to run it"
        )
    spawners: list[Spawner] = [
        FlowSpawner(
            flow,
            i,
            net=net,
            routes=routes,
            router=router,
            types=types,
            rng=rng.stream(f"flow:{flow.id}"),
            dt=dt,
        )
        for i, flow in enumerate(demand.flows)
    ]
    if demand.trips:
        spawners.append(
            TripSchedule(
                demand.trips,
                len(demand.flows),
                net=net,
                routes=routes,
                router=router,
                types=types,
                rng=rng,
                dt=dt,
            )
        )
    return spawners
