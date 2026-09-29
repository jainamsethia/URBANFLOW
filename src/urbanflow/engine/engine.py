"""The step orchestrator (plan F.1, AA 5.6).

``Engine.step()`` advances from ``t_n = n dt`` to ``t_{n+1}`` synchronously: decisions in
sub-steps 4-7 read the state committed at ``t_n`` plus earlier sub-steps' results. The
sub-steps run in plan order: 0 begin, 2 spawn, 3 insert, 4 leaders, 6 intersections,
7 longitudinal, 8 advance, 9 bookkeeping (with the ``deadlock_timeout`` watchdog),
10 finish. Sub-step 1 (signals) arrives with signalised intersections and sub-step 5
(lane changes) with lane changing.

Events of a step carry ``step = step_count`` after the step (the frame they belong to);
spawn, insertion and command events are stamped with the step's start time, link
crossings and arrivals with their interpolated time, and halting transitions with the
step's end time. Time is always ``step_count * dt``, never accumulated.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import SimulationError
from urbanflow.core.events import EventBuffer, EventType
from urbanflow.core.rng import RngStreams
from urbanflow.core.types import IntersectionKind
from urbanflow.demand.insertion import InsertionQueues, enqueue, insert_step, lane_tails
from urbanflow.demand.spawners import Spawner, build_spawners
from urbanflow.engine.advance import advance_links
from urbanflow.engine.bookkeeping import bookkeeping_step
from urbanflow.engine.commands import EngineCommands
from urbanflow.engine.intersections import JunctionIndex, admit, reservations
from urbanflow.engine.invariants import check_always, check_debug
from urbanflow.engine.leaders import SiblingGroups, compute_leaders, remaining_roads
from urbanflow.engine.longitudinal import expire_overrides, longitudinal_step
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.network.graph import build_road_graph
from urbanflow.routing.base import Router, RouteTable
from urbanflow.scenario.schema import DemandSpec
from urbanflow.vehicles.car_following import CarFollowingModel
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = ["Engine"]


def _check_supported(net: CompiledNetwork, config: SimulationConfig, demand: DemandSpec) -> None:
    """Fail early and by name on features scheduled for later versions of the engine."""
    signalized = np.flatnonzero(net.int_kind == IntersectionKind.signalized.code)
    if signalized.size:
        ids = ", ".join(f'"{net.int_ids[j]}"' for j in signalized.tolist())
        raise SimulationError(
            f"signalized intersections ({ids}) are not supported by this version of the "
            'engine yet; use kind "priority" or "uncontrolled" for them'
        )
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


class Engine:
    """Runtime state and the step pipeline of one simulation (the only mutator, AD 0.8).

    ``network`` is shared read-only; ``types`` must be built for ``car_following`` (its
    ``Params``); ``demand`` is the *resolved* scenario demand. The constructor rejects
    signalized intersections and transit lines (later versions) with a
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
    ) -> None:
        _check_supported(network, config, demand)
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
        )
        self._next_seq = admission.next_seq
        self.forced_commits += admission.forced
        if admission.commits:  # the newly committed now look past their stop line
            leaders = compute_leaders(net, veh, run, types, remaining, self.sibling_groups)

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
            admission.obstacle_gap,
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
