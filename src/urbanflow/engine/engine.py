"""The step orchestrator (plan F.1, AA 5.6).

``Engine.step()`` advances from ``t_n = n dt`` to ``t_{n+1}`` synchronously: decisions in
sub-steps 4-7 read the state committed at ``t_n`` plus earlier sub-steps' results. The
sub-steps run in plan order: 0 begin, 1 signals, 2 spawn, 3 insert, 4 leaders,
6 intersections, 7 longitudinal, 8 advance, 9 bookkeeping (with the ``deadlock_timeout``
watchdog), 10 finish. Sub-step 5 (lane changes) arrives with lane changing.

Sub-step 1: every signalised intersection in green asks its controller (``decide``, with
a lazily filled :class:`~urbanflow.signals.controllers.base.ControllerContext`), then
:meth:`SignalRuntime.advance` moves the stage timers and ``movement_state``, which the
admission of sub-step 6 and the lookahead obstacles read.

Events of a step carry ``step = step_count`` after the step (the frame they belong to);
spawn, insertion and command events are stamped with the step's start time, link
crossings and arrivals with their interpolated time, and halting transitions with the
step's end time. Time is always ``step_count * dt``, never accumulated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
from pydantic import JsonValue

from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import ConfigError, SimulationError
from urbanflow.core.events import EventBuffer, EventType
from urbanflow.core.rng import RngStreams
from urbanflow.core.types import IntersectionKind, Stage
from urbanflow.demand.insertion import InsertionQueues, enqueue, insert_step, lane_tails
from urbanflow.demand.spawners import Spawner, build_spawners
from urbanflow.engine.advance import advance_links
from urbanflow.engine.bookkeeping import bookkeeping_step
from urbanflow.engine.commands import EngineCommands
from urbanflow.engine.intersections import (
    JunctionIndex,
    admit,
    reservations,
    signal_lookahead,
)
from urbanflow.engine.invariants import check_always, check_debug
from urbanflow.engine.leaders import SiblingGroups, compute_leaders, remaining_roads
from urbanflow.engine.longitudinal import expire_overrides, longitudinal_step
from urbanflow.engine.signalling import LaneStats, detector_occupancy
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.network.graph import build_road_graph
from urbanflow.routing.base import Router, RouteTable
from urbanflow.scenario.schema import DemandSpec
from urbanflow.signals import (
    ControllerContext,
    ControllerRef,
    ControllerSetup,
    SignalController,
    SignalProgram,
    SignalRuntime,
    controller_name,
    create_controller,
)
from urbanflow.vehicles.car_following import CarFollowingModel
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = ["Engine"]


def _check_supported(config: SimulationConfig, demand: DemandSpec) -> None:
    """Fail early and by name on features scheduled for later versions of the engine."""
    if demand.transit:
        ids = ", ".join(f'"{line.id}"' for line in demand.transit)
        raise SimulationError(
            f"transit lines ({ids}) are not supported by this version of the engine yet; "
            "remove them from the scenario's demand to run it"
        )
    if config.accel != "numpy":
        raise SimulationError(
            f'accel="{config.accel}" is not available in this version; use accel="numpy"'
        )


def _check_signals(
    net: CompiledNetwork,
    programs: Sequence[SignalProgram],
    controllers: Mapping[int, ControllerRef],
) -> None:
    """Every signalised intersection has exactly one program; overrides name signals."""
    expected = np.flatnonzero(net.int_kind == IntersectionKind.signalized.code).tolist()
    given = sorted(p.intersection for p in programs)
    if given != expected:
        missing = ", ".join(f'"{net.int_ids[j]}"' for j in expected if j not in given)
        raise SimulationError(
            "every signalized intersection needs one signal program (signals=); "
            f"missing: {missing or 'none'}, given for intersections {given}"
        )
    for j in controllers:
        if j not in expected:
            name = f'"{net.int_ids[j]}"' if 0 <= j < net.n_intersections else str(j)
            raise ConfigError(f"controllers: intersection {name} is not signalized")


class Engine:
    """Runtime state and the step pipeline of one simulation (the only mutator, AD 0.8).

    ``network`` is shared read-only; ``types`` must be built for ``car_following`` (its
    ``Params``); ``demand`` is the *resolved* scenario demand. ``signals`` are the programs
    of every signalised intersection (:func:`~urbanflow.signals.program.build_programs`);
    ``controllers`` optionally replaces the configured controller of some of them (by
    intersection index: a registry name, ``{"type", "params"}``, a ``ControllerSpec``, an
    instance or a factory). The constructor rejects transit lines (a later version) with a
    :class:`SimulationError` naming them.
    """

    def __init__(
        self,
        network: CompiledNetwork,
        config: SimulationConfig,
        *,
        rng: RngStreams,
        router: Router,
        car_following: CarFollowingModel,
        types: VehicleTypes,
        demand: DemandSpec,
        signals: Sequence[SignalProgram] = (),
        controllers: Mapping[int, ControllerRef] | None = None,
    ) -> None:
        _check_supported(config, demand)
        overrides = dict(controllers or {})
        _check_signals(network, signals, overrides)
        self.network = network
        self.config = config
        self.router = router
        self.car_following = car_following
        self.types = types
        self.demand = demand
        self.graph: nx.DiGraph = build_road_graph(network)
        self.sibling_groups = SiblingGroups.build(network)
        """Static merge/diverge groups of the leader search."""
        self.junctions = JunctionIndex.build(network)
        """Static per-connector data of the admission loop."""
        self.commands = EngineCommands(self)
        """Validated vehicle commands (F.4)."""
        self.queues = InsertionQueues(network)
        """FIFO insertion queues; ``len(queues)`` is the backlog."""
        self.signals = SignalRuntime(
            signals, n_intersections=network.n_intersections, n_movements=network.n_movements
        )
        """Signal stage timers and ``movement_state`` (H.2)."""
        self._controller_refs = overrides
        self._reset(rng)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None) -> None:
        """Fresh runtime from the cached network; ``None`` reuses the current seed."""
        self._reset(RngStreams(self.rng.seed if seed is None else seed))

    def _reset(self, rng: RngStreams) -> None:
        self.rng = rng
        self.vehicles = VehicleTable()
        self.routes = RouteTable()
        self.router.reset(self.network, self.graph, self.config)
        self.queues.clear()
        self.spawners: list[Spawner] = build_spawners(
            self.demand, self.network, self.routes, self.router, self.types, rng, self.config.dt
        )
        self.events = EventBuffer()
        """Events of the current step (cleared at step start)."""
        self._model_rng = rng.stream(f"model:{self.config.car_following}")
        self._step_count = 0
        self._next_seq = 0
        self.generated = 0
        """Vehicles spawned (flows, trips, API)."""
        self.arrived = 0
        self.removed = 0
        self.teleported = 0
        self.forced_commits = 0
        self.red_runs = 0
        self.zone_conflicts = 0
        self.safety_cap_violations = 0
        self.commands.reset()
        self.signals.reset()
        self.controllers: dict[int, SignalController] = {}
        """The active controller of each signalised intersection."""
        self.parked: dict[int, SignalController] = {}
        """Configured controllers parked by a manual hold (``hold_phase``)."""
        self.held = np.zeros(self.network.n_intersections, dtype=bool)
        """Intersections under a manual hold."""
        self.configured: dict[int, dict[str, JsonValue]] = {}
        """The configured controller of each signalised intersection as ``{"type",
        "params"}`` (its name and validated parameters; results provenance)."""
        for prog in self.signals.programs:
            j = prog.intersection
            ref = self._controller_refs.get(j, prog.controller)
            self.controllers[j] = self.make_controller(j, ref, initial=True)
        self.signals.end_step()
        self._detector_seen = np.full(self.network.n_lanes, -np.inf)

    # ------------------------------------------------------------------ state
    @property
    def step_count(self) -> int:
        """Steps completed since the last reset."""
        return self._step_count

    @property
    def time(self) -> float:
        """Simulated time, s: ``step_count * dt``."""
        return self._step_count * self.config.dt

    @property
    def backlog(self) -> int:
        """Vehicles waiting to be inserted."""
        return len(self.queues)

    @property
    def demand_exhausted(self) -> bool:
        """Every spawner is past its end or count and nothing waits for insertion."""
        return all(s.exhausted for s in self.spawners) and not len(self.queues)

    # ------------------------------------------------------------------ signals
    def make_controller(
        self, j: int, ref: ControllerRef, *, initial: bool = False
    ) -> SignalController:
        """A controller for signalised intersection ``j`` from ``ref``, already reset.

        ``initial`` (simulation reset) lets it place the runtime state; mid-run installs
        continue from the current state.
        """
        prog = self.signals.program(j)
        where = f'intersection "{self.network.int_ids[j]}"'
        controller, params = create_controller(ref, where)
        setup = ControllerSetup(
            program=prog,
            params=params,
            rng=self._controller_rng(j),
            dt=self.config.dt,
            initial=initial,
            _place=lambda *a, **k: self.signals.place(j, *a, **k),
        )
        if initial:
            self.configured[j] = {
                "type": controller_name(controller),
                "params": params.model_dump(mode="json"),
            }
        try:
            controller.reset(setup)
        except Exception as exc:
            raise SimulationError(
                f'signal controller "{controller_name(controller)}" of {where} failed in '
                f"reset: {type(exc).__name__}: {exc}"
            ) from exc
        return controller

    def _controller_rng(self, j: int) -> np.random.Generator:
        return self.rng.stream(f"controller:{self.network.int_ids[j]}")

    def _signal_step(self, run: np.ndarray, t: float, tag: int) -> None:
        """F.1 sub-step 1: controllers decide (in green), then the runtime advances."""
        net, sig, dt = self.network, self.signals, self.config.dt
        occupied = detector_occupancy(net, self.vehicles, run)
        self._detector_seen[occupied] = t
        lanes = LaneStats(net, self.vehicles, self.types, run, occupied, t - self._detector_seen)
        for prog in sig.programs:
            j = prog.intersection
            if sig.stage[j] != Stage.green.code:
                continue  # decide is only called in green (H.3)
            controller = self.controllers[j]
            ctx = ControllerContext(
                time=t,
                dt=dt,
                program=prog,
                phase=int(sig.phase[j]),
                stage=Stage.green,
                green_elapsed=float(sig.green_elapsed[j]),
                stage_elapsed=float(sig.stage_elapsed[j]),
                lanes=lanes,
                rng=self._controller_rng(j),
            )
            name, where = controller_name(controller), net.int_ids[j]
            who = f'signal controller "{name}" of intersection "{where}"'
            try:
                q = controller.decide(ctx)
            except Exception as exc:
                raise SimulationError(
                    f"{who} failed at t={t:g} s: {type(exc).__name__}: {exc}"
                ) from exc
            if q is None:
                continue
            if isinstance(q, bool) or not isinstance(q, int | np.integer):
                raise SimulationError(
                    f"{who} returned {q!r}; decide() must return a phase index or None"
                )
            if not 0 <= q < prog.n_phases:
                raise SimulationError(
                    f"{who} requested phase {q}, but the intersection has {prog.n_phases} "
                    f"phases (0-{prog.n_phases - 1})"
                )
            sig.request(j, int(q))
        sig.advance(dt, self.events, step=tag, time=t)

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        """Advance one step (F.1)."""
        net, veh, types, cfg = self.network, self.vehicles, self.types, self.config
        n, dt = self._step_count, cfg.dt
        t = n * dt
        tag = n + 1  # the events' step: the frame they belong to
        events = self.events

        # 0 begin
        events.clear()
        veh.recycle()
        self.commands.emit_pending(events, tag)

        # 1 signals
        if self.signals.programs:
            self._signal_step(veh.running(), t, tag)

        # 2 spawn
        for spawner in self.spawners:
            for req in spawner.due(n):
                h = enqueue(veh, types, self.routes, self.queues, req)
                self.generated += 1
                events.append(
                    EventType.vehicle_departed, tag, req.depart_time, handle=h, uid=int(veh.uid[h])
                )

        # 3 insert
        run = veh.running()
        reserved = reservations(net, veh, types, run)
        limit = math.inf if cfg.max_vehicles is None else max(0, cfg.max_vehicles - run.size)
        tails = lane_tails(net, veh, types)
        new = insert_step(
            net, veh, types, self.routes, self.queues, tails, reserved, dt=dt, time=t, limit=limit
        )
        veh.halting[new] = veh.speed[new] < cfg.halting_speed
        first = new[veh.teleports[new] == 0]  # vehicle_inserted once: not again after a teleport
        for kind, hs in (
            (EventType.vehicle_inserted, first),
            (EventType.vehicle_entered_link, new),
        ):
            events.append_many(kind, tag, t, handles=hs, uids=veh.uid[hs], links=veh.link[hs])

        # 4 leaders
        run = veh.running()
        remaining = remaining_roads(veh, self.routes, run)
        leaders = compute_leaders(net, veh, run, types, remaining, self.sibling_groups)

        # 6 intersections
        expire_overrides(veh, run, t)
        admission = admit(
            net,
            veh,
            types,
            run,
            leaders,
            remaining,
            reserved,
            self.junctions,
            dt=dt,
            next_seq=self._next_seq,
            movement_state=self.signals.movement_state if self.signals.programs else None,
        )
        self._next_seq = admission.next_seq
        self.forced_commits += admission.forced
        self.red_runs += admission.red_runs
        if admission.commits:  # the newly committed now look past their stop line
            leaders = compute_leaders(net, veh, run, types, remaining, self.sibling_groups)
        obstacle = admission.obstacle_gap
        if self.signals.programs:  # stop lines ending the lookahead at r / y (F.1 step 4)
            ahead = signal_lookahead(
                net,
                veh,
                types,
                run,
                leaders.stop_gap,
                self.routes,
                self.junctions,
                self.signals.movement_state,
            )
            obstacle = np.minimum(obstacle, ahead)

        # 7 longitudinal
        v0_start = veh.v0[run].copy()
        cap = np.minimum(admission.cap_gap, leaders.stop_gap)
        lon = longitudinal_step(
            net,
            veh,
            types,
            self.car_following,
            run,
            leaders,
            obstacle,
            cap,
            dt=dt,
            rng=self._model_rng,
        )

        # 8 advance
        adv = advance_links(
            net,
            veh,
            types,
            self.routes,
            self.router,
            run,
            lon.dx,
            lon.v_start,
            events,
            step=tag,
            time=t,
            dt=dt,
        )
        self.arrived += int(adv.arrived.size)
        self.safety_cap_violations += lon.violations + adv.violations

        # 9 bookkeeping
        book = bookkeeping_step(
            net,
            veh,
            self.routes,
            self.queues,
            run,
            adv.dx,
            v0_start,
            adv.arrived,
            events,
            dt=dt,
            halting_speed=cfg.halting_speed,
            deadlock_timeout=cfg.deadlock_timeout,
            step=tag,
            time=t + dt,
        )
        self.teleported += int(book.teleported.size)
        self.arrived += int(book.arrived.size)

        # 10 finish
        self._step_count = tag
        check_always(veh, veh.running(), tag)
        if cfg.debug_checks:
            check_debug(self, adv)
        self.signals.end_step()
