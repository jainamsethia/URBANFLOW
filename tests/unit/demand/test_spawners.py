"""Arrival processes and per-vehicle draws (plan E.7 §1.3, I.3, F.6, AG.1 R11)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import UrbanFlowError
from urbanflow.core.rng import RngStreams
from urbanflow.demand import FlowSpawner, SpawnRequest, TripSchedule, build_spawners
from urbanflow.demand.spawners import Spawner, emission_step
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.network.graph import build_road_graph
from urbanflow.routing import RouteTable, ShortestPathRouter
from urbanflow.scenario import Scenario
from urbanflow.vehicles import IDM, VehicleTypes
from urbanflow.vehicles.table import DEPART_LANE_BEST, DEPART_LANE_FIRST

ROUTE_E = ["W_J1", "J1_J2", "J2_E"]
ROUTE_N = ["W_J1", "J1_J2", "J2_N"]


@dataclass
class Demand:
    spawners: list[Spawner]
    net: CompiledNetwork
    routes: RouteTable
    types: VehicleTypes

    def run(self, steps: int, which: int = 0) -> list[SpawnRequest]:
        sp = self.spawners[which]
        return [r for n in range(steps) for r in sp.due(n)]

    def roads(self, route_id: int) -> list[str]:
        return [self.net.road_ids[r] for r in self.routes.get(route_id)]


Build = Callable[..., Demand]


@pytest.fixture
def build(corridor_doc: Callable[..., dict[str, Any]]) -> Build:
    def make(*, seed: int = 0, dt: float = 1.0, **demand: Any) -> Demand:
        scenario = Scenario.from_dict(corridor_doc(**demand))
        net = compile_network(scenario)
        routes = RouteTable()
        router = ShortestPathRouter()
        router.reset(net, build_road_graph(net), SimulationConfig())
        types = VehicleTypes.from_specs(scenario.resolved.vehicle_types, IDM.Params)
        spawners = build_spawners(
            scenario.resolved.demand, net, routes, router, types, RngStreams(seed), dt
        )
        return Demand(spawners, net, routes, types)

    return make


def _flow(**fields: Any) -> dict[str, Any]:
    return {"id": "f", "route": ROUTE_E, **fields}


def test_emission_step() -> None:
    assert emission_step(0.0, 1.0) == 0
    assert emission_step(0.5, 1.0) == 1
    assert emission_step(1.0, 1.0) == 1
    assert emission_step(0.3, 0.1) == 3  # 0.3/0.1 = 2.9999999999999996
    assert emission_step(3 * 0.1, 0.1) == 3  # 0.30000000000000004
    assert emission_step(1.0 + 1e-12, 1.0) == 1  # within the 1e-9 tolerance


def test_uniform_times_are_exact_without_drift(build: Build) -> None:
    demand = build(dt=0.1, flows=[_flow(rate=1000.0, begin=0.5, count=2000)])
    got = demand.run(80_000)
    h = 3600 / 1000.0
    assert [r.depart_time for r in got] == [0.5 + k * h for k in range(2000)]  # computed, exact
    accumulated = 0.5
    for _ in range(1999):
        accumulated += h
    assert accumulated != got[-1].depart_time  # summing would have drifted
    sp = build(dt=0.1, flows=[_flow(period=0.25, count=8)]).spawners[0]
    per_step = [len(sp.due(n)) for n in range(20)]
    # t_k = 0, .25, .5, .75, 1, 1.25, ... emitted at ceil(t/dt - 1e-9)
    assert [n for n, c in enumerate(per_step) for _ in range(c)] == [
        emission_step(0.25 * k, 0.1) for k in range(8)
    ]


def test_ids_sources_and_exhaustion(build: Build) -> None:
    demand = build(
        flows=[_flow(period=10.0, end=30.0), _flow(id="g", period=4.0, count=3, begin=5.0)]
    )
    f, g = demand.spawners
    got_f = demand.run(100, 0)
    assert [r.depart_time for r in got_f] == [0.0, 10.0, 20.0]  # end is exclusive
    assert [r.vehicle_id for r in got_f] == ["f.0", "f.1", "f.2"]
    assert {r.source_idx for r in got_f} == {0}
    assert f.exhausted
    got_g = demand.run(100, 1)
    assert [r.depart_time for r in got_g] == [5.0, 9.0, 13.0]  # count caps
    assert {r.source_idx for r in got_g} == {1}
    assert g.exhausted
    assert isinstance(f, FlowSpawner)
    assert not build(flows=[_flow(period=10.0)]).spawners[0].exhausted  # runs to the end


def test_poisson_mean_and_variance(build: Build) -> None:
    n = 100_000
    demand = build(flows=[_flow(rate=3600.0, arrival="poisson", count=n, begin=2.0)])
    got = demand.spawners[0].due(10**9)  # catch up: everything at once
    assert len(got) == n
    times = np.array([r.depart_time for r in got])
    gaps = np.diff(np.r_[2.0, times])  # t_{-1} = begin
    # Exp(lambda = 1/s): mean 1, variance 1; 3 sigma of the estimators over 1e5 draws
    assert abs(gaps.mean() - 1.0) < 3 / math.sqrt(n)
    assert abs(gaps.var() - 1.0) < 3 * math.sqrt(8 / n)
    assert gaps.min() > 0


def test_poisson_respects_end(build: Build) -> None:
    got = build(flows=[_flow(rate=720.0, arrival="poisson", end=600.0)]).run(1000)
    assert got
    assert max(r.depart_time for r in got) < 600.0
    assert len(got) == pytest.approx(120, abs=4 * math.sqrt(120))


def test_binomial_probability(build: Build) -> None:
    steps = 20_000
    demand = build(flows=[_flow(rate=1800.0, arrival="binomial", end=float(steps))])
    got = demand.run(steps + 50)
    p = 1800 * 1.0 / 3600
    assert abs(len(got) / steps - p) < 3 * math.sqrt(p * (1 - p) / steps)
    times = [r.depart_time for r in got]
    assert len(set(times)) == len(times)  # at most one per step
    assert all(t == int(t) for t in times)  # departs at n dt
    assert max(times) < steps
    assert demand.spawners[0].exhausted


def test_binomial_window_and_count(build: Build) -> None:
    got = build(flows=[_flow(rate=3600.0, arrival="binomial", begin=10.2, end=20.0)]).run(100)
    # p = 1: every step n with n dt in [10.2, 20), i.e. steps 11..19
    assert [r.depart_time for r in got] == [float(n) for n in range(11, 20)]
    capped = build(flows=[_flow(rate=3600.0, arrival="binomial", count=4)])
    assert len(capped.run(100)) == 4
    assert capped.spawners[0].exhausted


def _signature(demand: Demand, reqs: list[SpawnRequest]) -> list[tuple[Any, ...]]:
    return [
        (
            r.vehicle_id,
            r.depart_time,
            r.type_idx,
            tuple(demand.roads(r.route_id)),  # route ids are table-local; compare roads
            r.speed_factor,
            r.depart_lane,
        )
        for r in reqs
    ]


def test_common_random_numbers(build: Build) -> None:
    rich = _flow(
        route=None,
        routes=[{"roads": ROUTE_E, "weight": 1.0}, {"roads": ROUTE_N, "weight": 2.0}],
        type_mix={"car": 3.0, "truck": 1.0},
        rate=900.0,
        arrival="poisson",
        depart_lane="random",
    )
    one = build(seed=11, flows=[rich])
    alone = _signature(one, one.run(900))
    other = {"id": "other", "route": ROUTE_N, "rate": 1200.0, "arrival": "poisson"}
    two = build(seed=11, flows=[other, rich])
    assert alone == _signature(two, two.run(900, which=1))
    reseeded = build(seed=12, flows=[rich])
    assert alone != _signature(reseeded, reseeded.run(900))


def test_flows_without_choices_draw_nothing_for_them(build: Build) -> None:
    # no routes/type_mix: the schedule and speed factors do not depend on which fixed route
    # or OD resolution the flow uses
    fixed = build(seed=3, flows=[_flow(rate=600.0, arrival="poisson")]).run(1200)
    od = build(
        seed=3,
        flows=[
            {
                "id": "f",
                "origin": "W_J1",
                "destination": "J2_E",
                "rate": 600.0,
                "arrival": "poisson",
            }
        ],
    )
    got = od.run(1200)
    key = [(r.vehicle_id, r.depart_time, r.speed_factor) for r in fixed]
    assert key == [(r.vehicle_id, r.depart_time, r.speed_factor) for r in got]
    assert {tuple(od.roads(r.route_id)) for r in got} == {tuple(ROUTE_E)}
    assert len({r.route_id for r in got}) == 1  # OD route interned once


def test_route_and_type_shares(build: Build) -> None:
    n = 10_000
    demand = build(
        flows=[
            _flow(
                route=None,
                routes=[{"roads": ROUTE_E, "weight": 1.0}, {"roads": ROUTE_N, "weight": 3.0}],
                type_mix={"car": 1.0, "bus": 1.0},
                period=1.0,
                count=n,
            )
        ]
    )
    got = demand.spawners[0].due(10**9)
    share_n = np.mean([demand.roads(r.route_id)[-1] == "J2_N" for r in got])
    assert abs(share_n - 0.75) < 3 * math.sqrt(0.75 * 0.25 / n)
    share_bus = np.mean([r.type_idx == demand.types.index["bus"] for r in got])
    assert abs(share_bus - 0.5) < 3 * math.sqrt(0.25 / n)
    assert len(demand.routes) == 2  # each distribution entry interned once, at build


def test_speed_factor_is_a_clipped_normal(build: Build) -> None:
    got = build(flows=[_flow(period=1.0, count=3000)]).run(4000)
    f = np.array([r.speed_factor for r in got])
    assert f.min() >= 0.8
    assert f.max() <= 1.2
    assert abs(f.mean() - 1.0) < 0.01
    assert np.any(f == 0.8)  # clipped tails exist at 2 sigma
    emergency = build(flows=[_flow(period=1.0, count=5, vehicle_type="emergency")]).run(10)
    assert {r.speed_factor for r in emergency} == {1.3}  # std 0


def test_depart_lane_and_speed_modes(build: Build) -> None:
    flows = [
        _flow(id="best", period=5.0, count=1),
        _flow(id="first", period=5.0, count=1, depart_lane="first", depart_speed=4.5),
        _flow(id="exact", period=5.0, count=1, depart_lane=1),
        _flow(id="rnd", period=1.0, count=200, depart_lane="random"),
        {
            "id": "rnd_n",
            "route": ["J1_J2", "J2_N"],
            "period": 1.0,
            "count": 50,
            "depart_lane": "random",
        },
    ]
    demand = build(flows=flows)
    by_flow = [demand.run(300, i) for i in range(len(flows))]
    assert by_flow[0][0].depart_lane == DEPART_LANE_BEST
    assert math.isnan(by_flow[0][0].depart_speed)  # "max"
    assert (by_flow[1][0].depart_lane, by_flow[1][0].depart_speed) == (DEPART_LANE_FIRST, 4.5)
    assert by_flow[2][0].depart_lane == 1
    assert {r.depart_lane for r in by_flow[3]} == {0, 1}  # both W_J1 lanes reach J1_J2
    assert {r.depart_lane for r in by_flow[4]} == {0}  # only lane 0 of J1_J2 reaches J2_N


def test_trips(build: Build) -> None:
    trips = [
        {"id": "late", "depart": 7.5, "route": ROUTE_E},
        {"id": "od", "depart": 3.0, "origin": "W_J1", "destination": "J2_S"},
        {"id": "early", "depart": 3.0, "route": ROUTE_N, "vehicle_type": "bus"},
    ]
    demand = build(flows=[_flow(period=100.0, count=1)], trips=trips)
    schedule = demand.spawners[1]
    assert isinstance(schedule, TripSchedule)
    assert [r.vehicle_id for r in schedule.due(2)] == []
    first = schedule.due(3)
    assert [r.vehicle_id for r in first] == ["od", "early"]  # same step: file order
    assert [r.source_idx for r in first] == [2, 3]  # after the one flow
    assert demand.roads(first[0].route_id) == ["W_J1", "J1_J2", "J2_S"]
    assert first[1].type_idx == demand.types.index["bus"]
    assert [r.vehicle_id for r in schedule.due(8)] == ["late"]
    assert schedule.exhausted
    # each trip draws from its own trip:{id} stream
    sf = RngStreams(0).stream("trip:late").normal(1.0, 0.1)
    late = build(trips=trips).spawners[0].due(10)[-1]
    assert late.speed_factor == pytest.approx(float(np.clip(sf, 0.8, 1.2)))


def test_transit_is_rejected_by_name(build: Build) -> None:
    line = {
        "id": "bus1",
        "route": ROUTE_E,
        "headway": 600.0,
        "stops": [{"road": "W_J1", "position": 100.0}],
    }
    with pytest.raises(UrbanFlowError, match=r'transit lines \("bus1"\) are not supported'):
        build(transit=[line])
