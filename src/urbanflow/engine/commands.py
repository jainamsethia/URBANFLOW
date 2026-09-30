"""Validated control commands applied between steps (plan F.4).

Commands check their arguments (``CommandError``, or ``NotFoundError`` with a did-you-mean
hint for unknown ids), mutate runtime state immediately, and become visible in the next
``step()``; their events are emitted into that step's buffer. Every command is appended to
:attr:`EngineCommands.command_log` as ``(step, name, args)`` for replay verification.
``set_route`` and ``change_lane`` arrive with lane changing.

Signal commands take an intersection (id or index) and a phase (index or id); indices may
be Python or numpy integers:

* ``request_phase`` queues a phase at a controller that accepts requests (``external``);
  the runtime honours min-green, yellow and all-red;
* ``set_phase`` jumps to the phase now, without intergreen (tests and setup; the
  ``phase_changed`` event is marked forced);
* ``hold_phase`` parks the configured controller, installs a fresh ``external`` and
  requests the phase (any controller); ``release`` reinstates the parked controller, which
  continues from the current phase;
* ``set_controller`` installs a new controller (a registry name, ``{"type", "params"}``,
  an instance or a factory) that continues from the current phase; during a hold it
  replaces the parked controller. An instance already running another intersection is
  rejected (each intersection needs its own controller state).

Whenever the active controller changes (``set_controller`` outside a hold, ``release``)
the runtime's queued request is dropped: it belonged to the previous controller, and the
new one re-requests in its next ``decide`` if it wants to.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from urbanflow.core.errors import CommandError, NotFoundError, suggest
from urbanflow.core.events import EventBuffer, EventType
from urbanflow.core.types import VehicleStatus
from urbanflow.demand.insertion import enqueue
from urbanflow.demand.spawners import api_request
from urbanflow.routing.lanes import valid_mask
from urbanflow.scenario.schema import ControllerSpec, DepartLane, DepartSpeed, TripSpec
from urbanflow.scenario.validate import issues_from_pydantic
from urbanflow.signals import (
    ControllerRef,
    External,
    SignalController,
    SignalProgram,
    controller_name,
)

if TYPE_CHECKING:
    from urbanflow.engine.engine import Engine

__all__ = ["CommandLogEntry", "EngineCommands", "as_index"]

CommandLogEntry = tuple[int, str, dict[str, Any]]
"""``(step_count when issued, command name, JSON-safe arguments)``."""

_WAITING = VehicleStatus.waiting_insert.code
_RUNNING = VehicleStatus.running.code
_REMOVED = VehicleStatus.removed.code


class EngineCommands:
    """Vehicle and signal commands of one :class:`~urbanflow.engine.engine.Engine`."""

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

    # ------------------------------------------------------------------ signals
    def request_phase(self, intersection: int | str, phase: int | str) -> None:
        """Ask the controller of ``intersection`` for ``phase`` (honours min-green, yellow
        and all-red). Needs a controller that accepts requests (e.g. ``external``)."""
        j, prog = self._signal(intersection)
        q = self._phase(prog, phase)
        name = self._engine.network.int_ids[j]
        if self._engine.held[j]:
            raise CommandError(
                f'intersection "{name}" is under a manual hold; use hold_phase to change the '
                f'held phase or release("{name}") first'
            )
        controller = self._engine.controllers[j]
        if not _accepts(controller):
            raise CommandError(
                f'the controller "{controller_name(controller)}" of intersection "{name}" '
                "does not accept phase requests; install one that does with "
                f'set_controller("{name}", "external"), or use hold_phase("{name}", phase) '
                "for a temporary manual override"
            )
        controller.request_phase(q)  # type: ignore[attr-defined]
        self._log("request_phase", intersection=name, phase=q)

    def set_phase(self, intersection: int | str, phase: int | str) -> None:
        """Jump to ``phase`` now, without yellow or all-red (a forced change for tests and
        setup); the controller continues from it."""
        j, prog = self._signal(intersection)
        q = self._phase(prog, phase)
        self._engine.signals.force(j, q)
        self._log("set_phase", intersection=self._engine.network.int_ids[j], phase=q)

    def hold_phase(self, intersection: int | str, phase: int | str) -> None:
        """Manual override with any controller: park the configured controller, install a
        fresh ``external`` and request ``phase`` (min-green, yellow and all-red honoured).
        While held, :meth:`request_phase` raises; call :meth:`release` to end the hold."""
        eng = self._engine
        j, prog = self._signal(intersection)
        q = self._phase(prog, phase)
        if not eng.held[j]:
            eng.parked[j] = eng.controllers[j]
            eng.controllers[j] = eng.make_controller(j, External)
            eng.held[j] = True
        eng.controllers[j].request_phase(q)  # type: ignore[attr-defined]  # External
        self._log("hold_phase", intersection=eng.network.int_ids[j], phase=q)

    def release(self, intersection: int | str) -> None:
        """End a manual hold: the parked controller continues from the current phase."""
        eng = self._engine
        j, _ = self._signal(intersection)
        name = eng.network.int_ids[j]
        if not eng.held[j]:
            raise CommandError(f'intersection "{name}" has no manual hold to release')
        eng.controllers[j] = eng.parked.pop(j)
        eng.held[j] = False
        eng.signals.pending[j] = -1  # the hold's request is not the parked controller's
        self._log("release", intersection=name)

    def set_controller(self, intersection: int | str, ref: ControllerRef) -> None:
        """Install the controller ``ref`` at ``intersection``; it continues from the current
        phase. During a manual hold it replaces the parked controller. An instance already
        running another intersection raises ``CommandError`` (pass a factory)."""
        eng = self._engine
        j, _ = self._signal(intersection)
        if not isinstance(ref, type) and hasattr(ref, "decide"):
            running = [c for k, c in (*eng.controllers.items(), *eng.parked.items()) if k != j]
            if any(ref is c for c in running):
                raise CommandError(
                    "one controller instance cannot run several intersections; pass a "
                    "factory instead (e.g. its class), which is called per intersection"
                )
        controller = eng.make_controller(j, ref)
        if eng.held[j]:
            eng.parked[j] = controller
        else:
            eng.controllers[j] = controller
            eng.signals.pending[j] = -1  # the previous controller's request
        self._log("set_controller", intersection=eng.network.int_ids[j], ref=_ref_json(ref))

    def signal_index(self, intersection: object) -> int:
        """Global index of a signalised intersection given by id or index (a Python or numpy
        integer). Unknown ids and indices out of range raise ``NotFoundError`` (with a hint);
        other values and unsignalised intersections raise ``CommandError``."""
        net = self._engine.network
        k = as_index(intersection)
        if isinstance(intersection, str):
            if intersection not in net.int_index:
                hint = suggest(intersection, net.int_index)
                raise NotFoundError(f'unknown intersection "{intersection}"{hint}')
            j = net.int_index[intersection]
        elif k is None:
            raise CommandError(f"intersection must be an id or an index, got {intersection!r}")
        elif not 0 <= k < net.n_intersections:
            raise NotFoundError(
                f"intersection index {k} out of range ({net.n_intersections} intersections)"
            )
        else:
            j = k
        if j not in self._engine.controllers:
            signalised = [net.int_ids[i] for i in sorted(self._engine.controllers)]
            raise CommandError(
                f'intersection "{net.int_ids[j]}" has no traffic signal; signalized: '
                f"{', '.join(signalised) or 'none'}"
            )
        return j

    def _signal(self, intersection: object) -> tuple[int, SignalProgram]:
        """``(index, program)`` of a signalised intersection (:meth:`signal_index`)."""
        j = self.signal_index(intersection)
        return j, self._engine.signals.program(j)

    def _phase(self, prog: SignalProgram, phase: object) -> int:
        key = phase if isinstance(phase, str) else as_index(phase)
        if key is None:
            raise CommandError(f"phase must be an index or a phase id, got {phase!r}")
        name = self._engine.network.int_ids[prog.intersection]
        try:
            return prog.phase_index(key)
        except NotFoundError as exc:
            raise NotFoundError(f'intersection "{name}": {exc}') from None

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


def as_index(value: object) -> int | None:
    """``int(value)`` for Python and numpy integers (bools excluded), else None."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    return None


def _accepts(controller: SignalController) -> bool:
    return bool(getattr(controller, "accepts_requests", False)) and callable(
        getattr(controller, "request_phase", None)
    )


def _ref_json(ref: ControllerRef) -> Any:
    """A JSON-safe form of a controller reference for the command log."""
    if isinstance(ref, str):
        return ref
    if isinstance(ref, ControllerSpec):
        return ref.model_dump(mode="json")
    if isinstance(ref, Mapping):
        return {str(k): v for k, v in ref.items()}
    return {"type": controller_name(ref)}
