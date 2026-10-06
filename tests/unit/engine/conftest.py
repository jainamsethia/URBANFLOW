"""Hand-built engine states on compiled networks (synthetic vehicles, no demand)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pytest

from urbanflow import ScenarioBuilder, generate
from urbanflow.core.events import EventBuffer
from urbanflow.core.types import IntArray, SignalState, VehicleStatus
from urbanflow.engine.intersections import (
    Admission,
    Grants,
    JunctionIndex,
    admit,
    grant_zones,
    reservations,
)
from urbanflow.engine.leaders import Leaders, SiblingGroups, compute_leaders, remaining_roads
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.routing import RouteTable, plan_connector, valid_mask
from urbanflow.vehicles import IDM, VehicleTable, VehicleTypes, desired_speed


@dataclass
class World:
    """A network plus a vehicle table filled by :meth:`place`."""

    net: CompiledNetwork
    veh: VehicleTable = field(default_factory=VehicleTable)
    routes: RouteTable = field(default_factory=RouteTable)
    events: EventBuffer = field(default_factory=EventBuffer)

    def __post_init__(self) -> None:
        self.types = VehicleTypes.from_specs(self.net.vehicle_types, IDM.Params)
        self.groups = SiblingGroups.build(self.net)
        self.junctions = JunctionIndex.build(self.net)

    def link(self, link_id: str) -> int:
        return self.net.link_index[link_id]

    def place(
        self,
        link: str,
        pos: float,
        speed: float = 0.0,
        *,
        route: Sequence[str],
        committed: bool = False,
        vtype: str = "car",
        factor: float = 1.0,
    ) -> int:
        """A running vehicle on ``link`` following ``route`` (road ids), engine semantics.

        Lane vehicles get their I.2 plan; connector vehicles plan (and are committed to)
        their own connector.
        """
        net, veh = self.net, self.veh
        h = veh.alloc(f"v{veh.next_uid}")
        lk = self.link(link)
        roads = [net.road_index[r] for r in route]
        rid = self.routes.intern(roads)
        on_lane = lk < net.n_lanes
        road = int(net.link_road[lk]) if on_lane else int(net.mov_from_road[net.link_movement[lk]])
        cursor = roads.index(road)
        t = self.types.index[vtype]
        veh.status[h] = VehicleStatus.running.code
        veh.active[h] = True
        veh.type_idx[h] = t
        veh.length[h] = self.types.length[t]
        veh.width[h] = self.types.width[t]
        veh.speed_factor[h] = factor
        veh.link[h] = lk
        veh.pos[h] = pos
        veh.speed[h] = speed
        veh.route_id[h] = rid
        veh.route_cursor[h] = cursor
        nxt = roads[cursor + 1] if cursor + 1 < len(roads) else -1
        veh.valid_mask[h] = valid_mask(net, road, nxt)
        veh.next_conn[h] = plan_connector(net, lk, roads, cursor) if on_lane else lk
        veh.committed[h] = committed or not on_lane
        veh.v0[h] = desired_speed(net, veh, self.types, np.array([h]))[0]
        return h

    def run(self) -> IntArray:
        return self.veh.running()

    def remaining(self) -> IntArray:
        return remaining_roads(self.veh, self.routes, self.run())

    def leaders(self) -> Leaders:
        run = self.run()
        return compute_leaders(self.net, self.veh, run, self.types, self.remaining(), self.groups)

    def admit(
        self,
        *,
        dt: float = 1.0,
        next_seq: int = 0,
        signals: dict[str, str] | None = None,
        permissive: Sequence[str] = (),
        sneakers: np.ndarray | None = None,
    ) -> tuple[Admission, np.ndarray]:
        """F.3 admission; ``signals`` maps movement ids to states ("r", "y", "g", "G") at
        a signalised junction (unlisted movements are red); ``permissive`` lists the
        movements whose yellow ends a g, with ``sneakers`` the per-lane uids eligible for
        end-of-green clearing (a fresh array of -1 if omitted)."""
        run = self.run()
        state = None
        if signals is not None:
            state = np.full(self.net.n_movements, SignalState.r.code, dtype=np.uint8)
            for mid, s in signals.items():
                state[self.net.mov_index[mid]] = SignalState(s).code
        marked = np.zeros(self.net.n_movements, dtype=bool)
        marked[[self.net.mov_index[m] for m in permissive]] = True
        if sneakers is None:
            sneakers = np.full(self.net.n_lanes, -1, dtype=np.int64)
        reserved = reservations(self.net, self.veh, self.types, run)
        adm = admit(
            self.net,
            self.veh,
            self.types,
            run,
            self.leaders(),
            self.remaining(),
            reserved,
            self.junctions,
            dt=dt,
            next_seq=next_seq,
            movement_state=state,
            permissive=marked,
            sneakers=sneakers,
        )
        return adm, reserved

    def grant(self, *, dt: float = 1.0) -> Grants:
        """F.3 zone locks of the committed vehicles on the current state."""
        return grant_zones(
            self.net, self.veh, self.types, self.run(), self.leaders(), self.junctions, dt=dt
        )

    def at(self, arr: np.ndarray, h: int) -> float:
        """Value of a run-aligned array for handle ``h``."""
        return float(arr[int(np.searchsorted(self.run(), h))])


@pytest.fixture(scope="session")
def junction_net() -> CompiledNetwork:
    """1-lane 4-arm uncontrolled junction (links ``{E,N,W,S}_{in,out}_0``)."""
    return compile_network(generate("single_intersection", kind="uncontrolled", lanes=1))


@pytest.fixture(scope="session")
def priority_net() -> CompiledNetwork:
    """The same junction with priority control; E-W is the major axis."""
    return compile_network(generate("single_intersection", kind="priority", lanes=1))


@pytest.fixture(scope="session")
def signal_net() -> CompiledNetwork:
    """The 1-lane junction, signalized, plus a type "hard" that brakes at 6 m/s^2."""
    b = ScenarioBuilder.from_scenario(generate("single_intersection", kind="signalized", lanes=1))
    b.vehicle_type("hard", decel=6.0, emergency_decel=6.0)
    return compile_network(b.build())


@pytest.fixture
def signalised(signal_net: CompiledNetwork) -> World:
    return World(signal_net)


@pytest.fixture
def junction(junction_net: CompiledNetwork) -> World:
    return World(junction_net)


@pytest.fixture
def priority(priority_net: CompiledNetwork) -> World:
    return World(priority_net)


@pytest.fixture
def corridor_world(corridor_net: CompiledNetwork) -> World:
    return World(corridor_net)


@pytest.fixture(scope="session")
def two_junctions_net() -> CompiledNetwork:
    """W -> J1 -> J2 -> {E, S}, 1 lane, derived movements: ``J1_J2_0`` is 14.6 m long and
    the right turn ``J1_J2_0->J2_S_0`` 5.65 m (shorter than a bus)."""
    b = ScenarioBuilder("two_junctions")
    b.boundary("W", (-200.0, 0.0))
    b.intersection("J1", (0.0, 0.0), kind="uncontrolled")
    b.intersection("J2", (25.0, 0.0), kind="uncontrolled")
    b.boundary("E", (225.0, 0.0))
    b.boundary("S", (25.0, -200.0))
    b.road("W_J1", "W", "J1")
    b.road("J1_J2", "J1", "J2")
    b.road("J2_E", "J2", "E")
    b.road("J2_S", "J2", "S")
    return compile_network(b.build())


@pytest.fixture
def two_junctions(two_junctions_net: CompiledNetwork) -> World:
    return World(two_junctions_net)
