"""Validated control commands applied between steps (plan F.4).

Commands check their arguments (``CommandError``, or ``NotFoundError`` with a did-you-mean
hint for unknown ids), mutate runtime state immediately, and become visible in the next
``step()``; their events are emitted into that step's buffer. Every command is appended to
:attr:`EngineCommands.command_log` as ``(step, name, args)`` for replay verification.
Signal commands arrive with signals; ``set_route`` and ``change_lane`` with lane changing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from urbanflow.core.errors import CommandError, NotFoundError, suggest
from urbanflow.core.events import EventBuffer, EventType
from urbanflow.core.types import VehicleStatus
from urbanflow.demand.insertion import enqueue
from urbanflow.demand.spawners import api_request
from urbanflow.routing.lanes import valid_mask
from urbanflow.scenario.schema import DepartLane, DepartSpeed, TripSpec
from urbanflow.scenario.validate import issues_from_pydantic

if TYPE_CHECKING:
    from urbanflow.engine.engine import Engine

__all__ = ["CommandLogEntry", "EngineCommands"]

CommandLogEntry = tuple[int, str, dict[str, Any]]
"""``(step_count when issued, command name, JSON-safe arguments)``."""

_WAITING = VehicleStatus.waiting_insert.code
_RUNNING = VehicleStatus.running.code
_REMOVED = VehicleStatus.removed.code


class EngineCommands:
    """Vehicle commands of one :class:`~urbanflow.engine.engine.Engine` (``engine.commands``)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        demand = engine.demand
        self._flow_ids = frozenset(f.id for f in demand.flows)
        self._trip_ids = frozenset(t.id for t in demand.trips)
        self.reset()

    def reset(self) -> None:
        """Forget the log, the API id counter and unsent events (engine reset)."""
        self.command_log: list[CommandLogEntry] = []
        self._api_count = 0
        self._pending: list[tuple[EventType, float, int, int, int]] = []

    def emit_pending(self, events: EventBuffer, step: int) -> None:
        """Append the events of commands issued since the last step (engine sub-step 0)."""
        for kind, time, handle, uid, link in self._pending:
            events.append(kind, step, time, handle=handle, uid=uid, link=link)
        self._pending.clear()

    # ------------------------------------------------------------------ vehicles
    def add_vehicle(
        self,
        *,
        route: Sequence[str] | None = None,
        origin: str | None = None,
        destination: str | None = None,
        via: Sequence[str] = (),
        vehicle_type: str = "car",
        id: str | None = None,
        depart_lane: DepartLane = "best",
        depart_speed: DepartSpeed = "max",
    ) -> str:
        """Spawn a vehicle now (``waiting_insert``); the next step tries to insert it.

        Give either ``route`` (road ids) or ``origin`` and ``destination`` (plus optional
        ``via``). ``id`` defaults to ``"api.{n}"``. Returns the vehicle id.
        """
        eng = self._engine
        net = eng.network
        vid = self._next_api_id() if id is None else id
        try:
            trip = TripSpec(
                id=vid,
                depart=eng.time,
                vehicle_type=vehicle_type,
                route=None if route is None else tuple(route),
                origin=origin,
                destination=destination,
                via=tuple(via),
                depart_lane=depart_lane,
                depart_speed=depart_speed,
            )
        except ValidationError as exc:
            details = "; ".join(
                i.message if i.path == "$" else f"{i.path}: {i.message}"
                for i in issues_from_pydantic(exc, root=TripSpec)
            )
            raise CommandError(f"add_vehicle: {details}") from None
        self._check_new_id(vid)
        od = () if trip.route else (trip.origin or "", trip.destination or "")
        for road in (*(trip.route or ()), *od, *trip.via):
            if road not in net.road_index:
                raise NotFoundError(f'unknown road "{road}"{suggest(road, net.road_index)}')
        eng.types.type_index(trip.vehicle_type)
        roads = [net.road_index[r] for r in trip.route or ()]
        for a, b in pairwise(roads):
            if (a, b) not in net.road_pair_conns:
                raise CommandError(
                    f'route is not connected: no movement from road "{net.road_ids[a]}" '
                    f'to road "{net.road_ids[b]}"'
                )
        first = roads[0] if roads else net.road_index[trip.origin or ""]
        lanes = int(net.road_n_lanes[first])
        lane = trip.depart_lane
        try:
            if isinstance(lane, int):
                if lane >= lanes:
                    raise CommandError(
                        f'depart_lane {lane} does not exist: road "{net.road_ids[first]}" has '
                        f"{lanes} lane(s) (0-{lanes - 1})"
                    )
                path = roads or eng.router.route(
                    first,
                    net.road_index[trip.destination or ""],
                    tuple(net.road_index[v] for v in trip.via),
                )
                nxt = path[1] if len(path) > 1 else -1
                if not valid_mask(net, first, nxt) >> lane & 1:
                    raise CommandError(
                        f'depart_lane {lane} of road "{net.road_ids[first]}" has no connection '
                        f'to the next road "{net.road_ids[nxt]}" of the route'
                    )
            req = api_request(
                trip,
                net=net,
                routes=eng.routes,
                router=eng.router,
                types=eng.types,
                rng=eng.rng.stream("vehicle_params"),
            )
        except NotFoundError as exc:  # unreachable destination
            raise CommandError(str(exc)) from None
        h = enqueue(eng.vehicles, eng.types, eng.routes, eng.queues, req)
        eng.generated += 1
        uid = int(eng.vehicles.uid[h])
        self._pending.append((EventType.vehicle_departed, eng.time, h, uid, -1))
        self._log(
            "add_vehicle",
            route=None if route is None else list(route),
            origin=origin,
            destination=destination,
            via=list(via),
            vehicle_type=vehicle_type,
            id=vid,
            depart_lane=depart_lane,
            depart_speed=depart_speed,
        )
        return vid

    def remove_vehicle(self, vehicle_id: str) -> None:
        """Take a waiting or running vehicle out now (``vehicle_removed`` in the next step)."""
        eng = self._engine
        veh = eng.vehicles
        h = self._handle(vehicle_id)
        link = -1
        if veh.status[h] == _WAITING:
            eng.queues[int(eng.routes.get(int(veh.route_id[h]))[0])].remove(h)
        elif veh.status[h] == _RUNNING:
            link = int(veh.link[h])
        veh.status[h] = _REMOVED
        uid = int(veh.uid[h])
        veh.free_deferred(h, between_steps=True)
        eng.removed += 1
        self._pending.append((EventType.vehicle_removed, eng.time, h, uid, link))
        self._log("remove_vehicle", vehicle_id=vehicle_id)

    def set_speed(
        self, vehicle_id: str, speed: float | None, duration: float | None = None
    ) -> None:
        """Replace the desired speed by ``speed`` m/s, for ``duration`` s (None: until cleared).

        Car-following, obstacles and the safe-speed cap still apply. ``speed=None`` clears
        the override.
        """
        eng = self._engine
        veh = eng.vehicles
        h = self._handle(vehicle_id)
        if speed is None:
            veh.speed_override[h] = math.nan
            veh.override_until[h] = math.inf
        else:
            if not _finite_at_least(speed, 0.0):
                raise CommandError(f"speed must be a finite number >= 0 m/s, got {speed!r}")
            if duration is not None and not (_finite_at_least(duration, 0.0) and duration > 0):
                raise CommandError(f"duration must be a finite number > 0 s, got {duration!r}")
            veh.speed_override[h] = float(speed)
            veh.override_until[h] = math.inf if duration is None else eng.time + float(duration)
        self._log("set_speed", vehicle_id=vehicle_id, speed=speed, duration=duration)

    # ------------------------------------------------------------------ helpers
    def _log(self, name: str, **args: Any) -> None:
        self.command_log.append((self._engine.step_count, name, args))

    def _handle(self, vehicle_id: str) -> int:
        veh = self._engine.vehicles
        if vehicle_id not in veh.id_to_handle and vehicle_id in veh.uid_to_id:
            raise NotFoundError(
                f'vehicle "{vehicle_id}" is no longer in the simulation (arrived or removed)'
            )
        return veh.handle_of(vehicle_id)

    def _check_new_id(self, vid: str) -> None:
        problem = self._id_problem(vid)
        if problem:
            raise CommandError(f'vehicle id "{vid}" {problem}')

    def _id_problem(self, vid: str) -> str:
        if vid in self._engine.vehicles.uid_to_id:
            return "is already used in this run"
        flow, _, k = vid.rpartition(".")
        if vid in self._trip_ids or (flow in self._flow_ids and k.isdigit()):
            return "is reserved for the scenario demand"
        return ""

    def _next_api_id(self) -> str:
        while True:
            vid = f"api.{self._api_count}"
            self._api_count += 1
            if not self._id_problem(vid):
                return vid


def _finite_at_least(value: object, low: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= low
    )
