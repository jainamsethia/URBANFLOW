"""The signal controller protocol, its setup and per-step context, and the registry (H.3).

A controller only decides *which* phase it wants: :meth:`SignalController.decide` is called
in green and returns a phase index (a request) or None (keep). The
:class:`~urbanflow.signals.state.SignalRuntime` owns min-green, yellow and all-red, so every
controller transitions safely.

:class:`ControllerContext` is read-only and built per intersection per step. Timing fields
are plain values; the lane and movement fields are computed on first access from the
engine's per-step :class:`LaneData` (itself lazy, at most once per step), so a controller
that never reads them costs nothing. The signals layer never imports the engine: the engine
supplies :class:`LaneData`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from urbanflow.core.errors import ConfigError, NotFoundError, SimulationError
from urbanflow.core.registry import Registry
from urbanflow.core.types import BoolArray, FloatArray, IntArray, SignalState, Stage
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.scenario.schema import ControllerSpec
from urbanflow.signals.program import SignalProgram

__all__ = [
    "ControllerBase",
    "ControllerContext",
    "ControllerRef",
    "ControllerSetup",
    "EmptyParams",
    "LaneData",
    "SignalController",
    "controller_name",
    "controller_registry",
    "create_controller",
    "register_controller",
]


class SignalController(Protocol):
    """A signal controller (H.3); register with ``@register_controller("name")``.

    Controllers are constructed by the registry with no arguments; parameters arrive in
    :meth:`reset` as ``setup.params`` (a validated ``Params`` instance). Controllers with
    ``accepts_requests = True`` also implement ``request_phase(phase: int) -> None``
    (the engine's ``request_phase`` command calls it).
    """

    name: ClassVar[str]
    Params: ClassVar[type[BaseModel]]
    accepts_requests: ClassVar[bool]

    def reset(self, setup: ControllerSetup) -> None:
        """Bind to a program and parameters (every simulation reset, and when installed)."""
        ...

    def decide(self, ctx: ControllerContext) -> int | None:
        """Called only in green: None keeps the phase, an int requests that phase."""
        ...

    def state_dict(self) -> dict[str, JsonValue]:
        """JSON-safe internal state (snapshots and digests)."""
        ...

    def load_state_dict(self, d: Mapping[str, JsonValue]) -> None:
        """Restore :meth:`state_dict` output."""
        ...


class EmptyParams(BaseModel):
    """Parameters of a controller that takes none."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ControllerBase:
    """Optional base: empty ``Params``, no-op ``reset``, ``state_dict``, ``load_state_dict``.

    Subclasses implement :meth:`SignalController.decide`.
    """

    name: ClassVar[str] = ""
    Params: ClassVar[type[BaseModel]] = EmptyParams
    accepts_requests: ClassVar[bool] = False

    def reset(self, setup: ControllerSetup) -> None:
        """Nothing to prepare."""

    def state_dict(self) -> dict[str, JsonValue]:
        """No internal state."""
        return {}

    def load_state_dict(self, d: Mapping[str, JsonValue]) -> None:
        """No internal state."""


@dataclass(frozen=True, slots=True)
class ControllerSetup:
    """What :meth:`SignalController.reset` receives."""

    program: SignalProgram
    params: BaseModel
    """A validated instance of the controller's ``Params``."""
    rng: np.random.Generator
    """The ``controller:{intersection id}`` stream (B.2 #11)."""
    dt: float
    initial: bool
    """True at a simulation reset (the runtime is in ``GREEN(initial_phase)`` at t = 0 and
    :meth:`set_initial` may move it); False when installed mid-run (``set_controller``,
    manual holds): the controller then continues from the current state."""
    _place: Callable[..., None] = field(repr=False)

    def set_initial(
        self,
        phase: int,
        *,
        stage: Stage = Stage.green,
        target: int = -1,
        stage_elapsed: float = 0.0,
        green_elapsed: float | None = None,
    ) -> None:
        """Set the runtime state at t = 0, e.g. a fixed-time offset position.

        ``target`` is the next phase inside yellow or all-red; ``green_elapsed`` defaults to
        ``stage_elapsed``. Only allowed when :attr:`initial` is True.
        """
        if not self.initial:
            raise SimulationError(
                "set_initial is only allowed at a simulation reset, not for a controller "
                "installed mid-run"
            )
        self._place(
            phase,
            stage=stage,
            target=target,
            stage_elapsed=stage_elapsed,
            green_elapsed=green_elapsed,
        )


class LaneData(Protocol):
    """Per-step lane and vehicle aggregates the engine supplies to every context.

    Each property is computed on first access and at most once per step, from the state
    at the start of the step. Per-link arrays have one entry per link (lanes, then
    connectors); per-lane arrays one per lane.
    """

    @property
    def net(self) -> CompiledNetwork: ...

    @property
    def count(self) -> IntArray:
        """Per link: running vehicles whose front is on it."""
        ...

    @property
    def halting(self) -> IntArray:
        """Per link: halting vehicles (``speed < halting_speed``)."""
        ...

    @property
    def queue(self) -> IntArray:
        """Per lane: stop-line queue (J.3), vehicles."""
        ...

    @property
    def waiting(self) -> FloatArray:
        """Per lane: sum of ``link_waiting_time`` of its vehicles, s."""
        ...

    @property
    def detector_occupied(self) -> BoolArray:
        """Per lane: a vehicle body is inside ``[L - DETECTOR_LENGTH, L]``."""
        ...

    @property
    def detector_last_seen(self) -> FloatArray:
        """Per lane: seconds since the detector was last occupied (0 if now, inf if never)."""
        ...

    @property
    def approaching(self) -> IntArray:
        """Per lane: moving vehicles within ``APPROACH_DISTANCE_M`` of the stop line."""
        ...

    @property
    def planned(self) -> IntArray:
        """Per connector index: lane vehicles whose planned connector it is."""
        ...

    @property
    def planned_halting(self) -> IntArray:
        """Per connector index: halting lane vehicles planning it."""
        ...

    @property
    def emergency(self) -> tuple[IntArray, IntArray, FloatArray]:
        """Running emergency vehicles: ``(link, connector link, distance)``; the connector
        is the planned one on a lane (-1 if none) or the current one; the distance is to the
        stop line on a lane and ``-pos`` on a connector."""
        ...


class ControllerContext:
    """Read-only inputs of :meth:`SignalController.decide` for one intersection and step.

    Per-lane arrays are aligned with :attr:`in_lanes`, per-movement arrays with
    ``program.movements`` (local movement index ``k``), per-connector arrays with
    :attr:`connectors`.
    """

    def __init__(
        self,
        *,
        time: float,
        dt: float,
        program: SignalProgram,
        phase: int,
        stage: Stage,
        green_elapsed: float,
        stage_elapsed: float,
        lanes: LaneData,
        rng: np.random.Generator,
    ) -> None:
        self.time = time
        self.dt = dt
        self.program = program
        self.intersection = program.intersection
        """Global intersection index."""
        self.phase = phase
        self.stage = stage
        self.green_elapsed = green_elapsed
        self.stage_elapsed = stage_elapsed
        self.min_green = float(program.min_green[phase])
        """Of the current phase, s."""
        self.max_green = float(program.max_green[phase])
        self.n_phases = program.n_phases
        self.rng = rng
        """The ``controller:{intersection id}`` stream."""
        self._lanes = lanes

    # ------------------------------------------------------------------ program
    @cached_property
    def phase_movements(self) -> BoolArray:
        """``[P, M_j]``: the movement is green (G or g) in the phase."""
        return self.program.phase_state >= SignalState.g.code

    @cached_property
    def connectors(self) -> IntArray:
        """Connector link ids of the intersection, in movement order."""
        net = self._lanes.net
        ptr, movs = net.mov_conn_ptr, self.program.movements
        return np.concatenate([net.mov_conn[ptr[m] : ptr[m + 1]] for m in movs.tolist()])

    @cached_property
    def conn_movement(self) -> IntArray:
        """Local movement index of each connector."""
        net = self._lanes.net
        counts = np.diff(net.mov_conn_ptr)[self.program.movements]
        return np.repeat(np.arange(self.program.n_movements), counts)

    @cached_property
    def phase_conn(self) -> BoolArray:
        """``[P, C_j]``: the connector's movement is green in the phase."""
        return self.phase_movements[:, self.conn_movement]

    # ------------------------------------------------------------------ incoming lanes
    @cached_property
    def in_lanes(self) -> IntArray:
        """Incoming lane link ids (clockwise by approach bearing, then lane index)."""
        net, j = self._lanes.net, self.intersection
        return net.int_in_lanes[net.int_in_ptr[j] : net.int_in_ptr[j + 1]]

    @cached_property
    def lane_count(self) -> IntArray:
        return self._lanes.count[self.in_lanes]

    @cached_property
    def lane_halting(self) -> IntArray:
        return self._lanes.halting[self.in_lanes]

    @cached_property
    def lane_queue(self) -> IntArray:
        """Stop-line queue (J.3), vehicles."""
        return self._lanes.queue[self.in_lanes]

    @cached_property
    def lane_waiting(self) -> FloatArray:
        """Sum of the vehicles' waiting time on the lane, s."""
        return self._lanes.waiting[self.in_lanes]

    @cached_property
    def detector_occupied(self) -> BoolArray:
        return self._lanes.detector_occupied[self.in_lanes]

    @cached_property
    def detector_last_seen(self) -> FloatArray:
        """Seconds since the lane's detector was last occupied (inf if never)."""
        return self._lanes.detector_last_seen[self.in_lanes]

    @cached_property
    def approaching(self) -> IntArray:
        """Moving vehicles within ``APPROACH_DISTANCE_M`` of the stop line."""
        return self._lanes.approaching[self.in_lanes]

    # ------------------------------------------------------------------ movements
    def _per_movement(self, per_conn: IntArray) -> IntArray:
        out = np.zeros(self.program.n_movements, dtype=np.int64)
        np.add.at(out, self.conn_movement, per_conn)
        return out

    @cached_property
    def movement_count(self) -> IntArray:
        """Lane vehicles planning one of the movement's connectors."""
        n_lanes = self._lanes.net.n_lanes
        return self._per_movement(self._lanes.planned[self.connectors - n_lanes])

    @cached_property
    def movement_halting(self) -> IntArray:
        """Halting lane vehicles planning one of the movement's connectors."""
        n_lanes = self._lanes.net.n_lanes
        return self._per_movement(self._lanes.planned_halting[self.connectors - n_lanes])

    @cached_property
    def movement_out_count(self) -> IntArray:
        """Vehicles on the movement's target lanes (each lane counted once)."""
        net = self._lanes.net
        to = net.conn_to_lane[self.connectors - net.n_lanes]
        pairs = np.unique(np.stack([self.conn_movement, to]), axis=1)
        out = np.zeros(self.program.n_movements, dtype=np.int64)
        np.add.at(out, pairs[0], self._lanes.count[pairs[1]])
        return out

    @cached_property
    def conn_from_lane(self) -> IntArray:
        """Lane link id each connector leaves from."""
        net = self._lanes.net
        return net.conn_from_lane[self.connectors - net.n_lanes]

    @cached_property
    def conn_in_count(self) -> IntArray:
        """Vehicles on each connector's from-lane (Varaiya pressure input)."""
        return self._lanes.count[self.conn_from_lane]

    @cached_property
    def conn_out_count(self) -> IntArray:
        """Vehicles on each connector's to-lane."""
        net = self._lanes.net
        return self._lanes.count[net.conn_to_lane[self.connectors - net.n_lanes]]

    # ------------------------------------------------------------------ emergency
    @cached_property
    def emergency_approach(self) -> list[tuple[int, int, float]]:
        """``(lane link id, local movement index or -1, distance)`` of emergency vehicles on
        the incoming lanes (distance to the stop line) or on the intersection's connectors
        (lane = the connector's from-lane, distance ``-pos``), nearest first."""
        net = self._lanes.net
        link, conn, dist = self._lanes.emergency
        local = np.full(net.n_links, -1, dtype=np.intp)
        local[self.connectors] = self.conn_movement
        on_lane = np.isin(link, self.in_lanes)
        on_conn = local[link] >= 0
        lane = np.where(on_conn, net.conn_from_lane[np.maximum(link - net.n_lanes, 0)], link)
        movement = np.where(conn >= 0, local[conn], -1)
        keep = np.flatnonzero(on_lane | on_conn)
        keep = keep[np.lexsort((lane[keep], dist[keep]))]
        return [(int(lane[i]), int(movement[i]), float(dist[i])) for i in keep.tolist()]


# --------------------------------------------------------------------------- registry
controller_registry: Registry[type[SignalController]] = Registry("controllers")


def register_controller[T: type[SignalController]](name: str) -> Callable[[T], T]:
    """Class decorator registering a :class:`SignalController` as ``name`` (sets
    ``cls.name`` when the class does not define it)."""

    def decorator(cls: T) -> T:
        if not vars(cls).get("name"):
            cls.name = name
        controller_registry.register(name, cls)
        return cls

    return decorator


ControllerRef = str | Mapping[str, Any] | ControllerSpec | SignalController | Callable[[], Any]
"""A registry name, ``{"type", "params"}``, a ``ControllerSpec``, an instance, or a
zero-argument factory (e.g. a controller class)."""


def create_controller(ref: ControllerRef, where: str = "") -> tuple[SignalController, BaseModel]:
    """A controller instance and its validated ``Params`` from ``ref``.

    Names and specs go through :data:`controller_registry` (unknown names raise
    ``NotFoundError`` with a hint); bad parameters raise ``ConfigError``. Instances and
    factories get default ``Params``. ``where`` prefixes error messages (e.g. the
    intersection).
    """
    prefix = f"{where}: " if where else ""
    if isinstance(ref, str | ControllerSpec | Mapping):
        try:
            spec = ref if isinstance(ref, ControllerSpec) else _spec(ref)
        except ValidationError as exc:
            raise ConfigError(f"{prefix}invalid controller {ref!r}: {exc}") from None
        try:
            cls = controller_registry.get(spec.type)
        except NotFoundError as exc:
            raise NotFoundError(f"{prefix}{exc}") from None
        instance: Any = cls()
        params_data: Mapping[str, Any] = spec.params
    else:
        instance = ref() if isinstance(ref, type) or not hasattr(ref, "decide") else ref
        params_data = {}
    if not callable(getattr(instance, "decide", None)):
        raise ConfigError(f"{prefix}{instance!r} is not a signal controller (no decide())")
    try:
        params = type(instance).Params.model_validate(dict(params_data))
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(x) for x in e['loc']) or 'params'}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(
            f'{prefix}invalid parameters for controller "{controller_name(instance)}": {details}'
        ) from None
    return instance, params


def controller_name(controller: object) -> str:
    """The registry name of a controller (its class name if it has none)."""
    return str(getattr(controller, "name", "") or type(controller).__name__)


def _spec(ref: str | Mapping[str, Any]) -> ControllerSpec:
    if isinstance(ref, str):
        return ControllerSpec(type=ref)
    return ControllerSpec.model_validate(dict(ref))
