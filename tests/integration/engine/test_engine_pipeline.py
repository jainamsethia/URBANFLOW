"""The step pipeline end to end (plan F.1, F.6, AA 5.6; AT-05, AT-06 for uncontrolled)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from urbanflow import generate
from urbanflow.core.errors import ConfigError, SimulationError
from urbanflow.core.events import EventType
from urbanflow.engine import Engine
from urbanflow.scenario import Scenario, ScenarioBuilder
from urbanflow.vehicles.table import COLUMNS

pytestmark = pytest.mark.integration

MakeEngine = Callable[..., Engine]
Builder = Callable[..., ScenarioBuilder]
E = EventType


def _busy(junction_builder: Builder, **simulation: object) -> Scenario:
    b = junction_builder(lanes=2, **simulation)
    for src, dst in (("W_in", "E_out"), ("N_in", "S_out"), ("E_in", "S_out"), ("S_in", "W_out")):
        b.flow(f"{src}-{dst}", route=[src, dst], rate=500.0, arrival="poisson")
    return b.build()


def test_empty_network_steps(make_engine: MakeEngine, junction_builder: Builder) -> None:
    e = make_engine(junction_builder().build())
    assert e.demand_exhausted
    for _ in range(10):
        e.step()
        assert len(e.events) == 0
    assert (e.step_count, e.time, e.generated, e.backlog) == (10, 10.0, 0, 0)


def test_time_is_step_count_times_dt(make_engine: MakeEngine, junction_builder: Builder) -> None:
    e = make_engine(junction_builder().build(), dt=0.1)
    for n in range(1, 31):
        e.step()
        assert e.time == n * 0.1  # never accumulated (0.1 * 30 != sum of 30 * 0.1)


def test_sub_step_order_is_observable_in_events(
    make_engine: MakeEngine, junction_builder: Builder
) -> None:
    e = make_engine(_busy(junction_builder), seed=3)
    phase = {
        E.vehicle_departed: 2,
        E.vehicle_inserted: 3,
        E.vehicle_changed_lane: 5,
        E.vehicle_exited_link: 8,
    }
    phase |= {E.vehicle_arrived: 8, E.vehicle_stopped: 9, E.vehicle_resumed: 9}
    seen: set[int] = set()
    for _ in range(300):
        e.step()
        ev = e.events
        inserted = set(ev.uid[ev.type == E.vehicle_inserted.code].tolist())
        entered_at_insertion: set[int] = set()
        order = []
        for code, uid in zip(ev.type.tolist(), ev.uid.tolist(), strict=True):
            kind = EventType.from_code(code)
            if kind is E.vehicle_entered_link:
                first = uid in inserted and uid not in entered_at_insertion
                entered_at_insertion.add(uid)
                order.append(3 if first else 8)
            else:
                order.append(phase[kind])
        assert order == sorted(order), order
        assert set(ev.step.tolist()) <= {e.step_count}
        seen |= set(order)
    assert seen == {2, 3, 5, 8, 9}  # 5: lane changes


def test_signal_programs_must_cover_every_signalized_intersection(
    make_engine: MakeEngine,
) -> None:
    from urbanflow.core.config import SimulationConfig
    from urbanflow.core.rng import RngStreams
    from urbanflow.network import compile_network
    from urbanflow.routing import ShortestPathRouter
    from urbanflow.vehicles import IDM, VehicleTypes

    scenario = generate("single_intersection", kind="signalized")
    e = make_engine(scenario)  # the fixture passes the programs
    assert [p.intersection for p in e.signals.programs] == [0] and e.signals.state_string(0)
    net = compile_network(scenario)
    with pytest.raises(SimulationError, match=r'missing: "J"'):
        Engine(
            net,
            SimulationConfig(),
            rng=RngStreams(0),
            router=ShortestPathRouter(),
            car_following=IDM(),
            types=VehicleTypes.from_specs(net.vehicle_types, IDM.Params),
            demand=scenario.resolved.demand,
        )
    with pytest.raises(ConfigError, match='intersection "N" is not signalized'):
        make_engine(scenario, controllers={net.int_index["N"]: "external"})


def test_rejects_transit_lines(make_engine: MakeEngine, junction_builder: Builder) -> None:
    b = junction_builder()
    b.transit_line(
        "L1", route=["W_in", "E_out"], stops=[{"road": "W_in", "position": 50.0}], headway=300.0
    )
    with pytest.raises(SimulationError, match='transit lines \\("L1"\\)'):
        make_engine(b.build())


def test_rejects_the_numba_backend(make_engine: MakeEngine, junction_builder: Builder) -> None:
    with pytest.raises(SimulationError, match="numba"):
        make_engine(junction_builder().build(), accel="numba")


def test_max_vehicles_caps_running_vehicles(
    make_engine: MakeEngine, junction_builder: Builder
) -> None:
    e = make_engine(_busy(junction_builder), max_vehicles=3)
    for _ in range(120):
        e.step()
        assert len(e.vehicles.running()) <= 3
    assert e.backlog > 0 and e.arrived > 0 and not e.demand_exhausted


def _table(e: Engine) -> dict[str, np.ndarray]:
    veh = e.vehicles
    return {name: getattr(veh, name)[: veh.top].copy() for name in COLUMNS}


def _same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return a.keys() == b.keys() and all(
        np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a
    )


def test_determinism(make_engine: MakeEngine, junction_builder: Builder) -> None:
    scenario = _busy(junction_builder)
    a, b, c = (make_engine(scenario, seed=s) for s in (5, 5, 6))
    diverged = False
    for _ in range(300):
        for e in (a, b, c):
            e.step()
        assert _same(_table(a), _table(b))
        assert a.events.view().keys() == b.events.view().keys()
        assert all(
            np.array_equal(x, y)
            for x, y in zip(a.events.view().values(), b.events.view().values(), strict=True)
        )
        diverged |= not _same(_table(a), _table(c))
    assert diverged


def test_reset_rebuilds_the_runtime(make_engine: MakeEngine, junction_builder: Builder) -> None:
    e = make_engine(_busy(junction_builder), seed=5)
    fresh = make_engine(_busy(junction_builder), seed=5)
    for _ in range(80):
        e.step()
    e.commands.add_vehicle(route=["W_in", "E_out"])
    e.reset()
    assert (e.step_count, e.generated, e.arrived, len(e.vehicles), e.backlog) == (0, 0, 0, 0, 0)
    assert not e.commands.command_log
    for _ in range(80):
        e.step()
        fresh.step()
        assert _same(_table(e), _table(fresh))
    e.reset(seed=6)
    assert e.rng.seed == 6
    for _ in range(80):
        e.step()
    assert not _same(_table(e), _table(fresh))


def test_at05_at06_uncontrolled_run(make_engine: MakeEngine) -> None:
    """AT-05/AT-06: 600 s at an uncontrolled junction with debug checks on."""
    e = make_engine(generate("single_intersection", kind="uncontrolled"), seed=11, duration=600)
    veh = e.vehicles
    travel, free_flow = [], []
    for _ in range(600):
        run = veh.running()
        before = {
            int(veh.uid[h]): (int(veh.link[h]), float(veh.pos[h]), float(veh.v0[h])) for h in run
        }
        e.step()
        run = veh.running()
        for h in run.tolist():
            uid = int(veh.uid[h])
            if uid in before:
                link, pos, v0 = before[uid]
                if link == veh.link[h]:
                    assert veh.pos[h] >= pos - 1e-9  # pos non-decreasing within a link
                assert veh.speed[h] <= max(v0, float(veh.v0[h])) + 1e-6
        # AT-06 conservation (also I6 inside the step)
        assert e.generated == e.backlog + run.size + e.arrived + e.removed
        ev = e.events
        for h, t in zip(
            ev.handle[ev.type == E.vehicle_arrived.code].tolist(),
            ev.time[ev.type == E.vehicle_arrived.code].tolist(),
            strict=True,
        ):
            travel.append(t - float(veh.insert_time[h]))
            free_flow.append(float(veh.ff_time[h]))
    assert e.arrived > 0 and len(travel) == e.arrived
    assert np.mean(travel) >= np.mean(free_flow)
    assert e.safety_cap_violations == 0
