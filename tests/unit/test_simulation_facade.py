"""The Simulation facade (plan AA 5.3, AD.1 8.9): construction, controllers=, lifecycle, run,
summary."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from urbanflow import Scenario, ScenarioBuilder, Simulation, SimulationResult, generate
from urbanflow.core import constants as C
from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import (
    ConfigError,
    NotFoundError,
    ScenarioValidationError,
    SimulationError,
)
from urbanflow.core.events import Event, EventType, Subscription
from urbanflow.core.types import VehicleStatus
from urbanflow.engine import Engine
from urbanflow.routing import ShortestPathRouter
from urbanflow.signals import ControllerBase, ControllerContext, External, FixedTime
from urbanflow.simulation import ProgressInfo

DET = {"speed_factor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}}
SUMMARY_KEYS = {
    "vehicles.generated",
    "vehicles.inserted",
    "vehicles.arrived",
    "vehicles.en_route",
    "vehicles.backlog",
    "vehicles.removed",
    "vehicles.teleported",
    "travel_time.mean",
    "travel_time.median",
    "travel_time.p95",
    "travel_time.std",
    "throughput_vph",
    "total_time_mean",
    "insertion_delay_mean",
    "att_censored",
    "delay.mean",
    "delay.median",
    "delay.p95",
    "waiting_time_mean",
    "stops_mean",
    "space_mean_speed",
    "vkt",
    "vht",
    "queue.mean_total_veh",
    "queue.max_lane_veh",
}


@pytest.fixture(scope="module")
def junction() -> Scenario:
    return generate("single_intersection", kind="uncontrolled", duration=300)


def _straight(*departs: float, length: float = 1000.0, **simulation: Any) -> Scenario:
    """One road A->B of ``length`` m; one deterministic car trip per ``departs`` entry."""
    b = ScenarioBuilder("straight", **simulation)
    b.vehicle_type("det", **DET)
    b.boundary("A", (0.0, 0.0))
    b.boundary("B", (length, 0.0))
    b.road("AB", "A", "B")
    for i, t in enumerate(departs):
        b.trip(f"v{i}", t, route=["AB"], vehicle_type="det")
    return b.build()


# ------------------------------------------------------------------------------- construction
def test_user_examples_a_and_b(junction: Scenario, tmp_path: Path) -> None:
    path = junction.save(tmp_path / "grid.json")
    sim = Simulation.from_scenario(path)  # example A
    sim.reset()
    while not sim.done:
        sim.step()
    metrics = sim.metrics.summary()
    assert set(metrics) == SUMMARY_KEYS and metrics["vehicles.arrived"] > 0
    scenario = Scenario.load(path)  # example B
    sim_b = Simulation(scenario)
    sim_b.run()
    result = sim_b.get_results()
    assert isinstance(result, SimulationResult) and result.sim_time == 300
    assert result.summary == metrics  # same scenario, same seed


def test_every_scenario_form_is_accepted(junction: Scenario, tmp_path: Path) -> None:
    path = junction.save(tmp_path / "s.json")
    forms = [junction, junction.spec, junction.to_dict(), path, str(path)]
    hashes = {Simulation(form).scenario.content_hash for form in forms}
    assert hashes == {junction.content_hash}


def test_config_precedence() -> None:
    scenario = _straight(0.0, dt=0.5, seed=3, duration=100)
    sim = Simulation(scenario, SimulationConfig(dt=0.25, halting_speed=0.2), seed=9)
    cfg = sim.config
    assert (cfg.dt, cfg.seed, cfg.duration, cfg.halting_speed) == (0.25, 9, 100, 0.2)
    assert (sim.dt, sim.seed) == (0.25, 9)
    assert Simulation(scenario).config.dt == 0.5  # the scenario block beats the defaults


def test_router_by_name_or_instance(junction: Scenario) -> None:
    assert Simulation(junction, router="shortest").config.router == "shortest"
    sim = Simulation(junction, router=ShortestPathRouter())
    assert sim.config.router == "shortest"
    with pytest.raises(ScenarioValidationError, match='unknown router "fastest"'):
        Simulation(junction, router="fastest")


def test_deep_check_errors_raise(junction: Scenario) -> None:
    with pytest.raises(ScenarioValidationError) as info:
        Simulation(junction, car_following="gipps")
    assert [(i.code, i.path) for i in info.value.errors] == [("E904", "simulation.car_following")]


def test_unsupported_features_fail_by_name(junction: Scenario) -> None:
    with pytest.raises(ConfigError, match=r"record.enabled"):
        Simulation(junction, record={"enabled": True})
    with pytest.raises(ConfigError, match="custom metric collectors"):
        Simulation(junction, metrics={"collectors": ["default", "speeding"]})
    with pytest.raises(ConfigError, match="unknown field"):
        Simulation(junction, **{"dtt": 1.0})  # type: ignore[arg-type]


def test_config_overrides_are_rechecked_against_the_demand(
    corridor_doc: Callable[..., dict[str, Any]],
) -> None:
    """E506/E507 use the run's final dt and duration, not only the scenario's own block."""
    route = ["W_J1", "J1_J2", "J2_E"]
    binomial = {"id": "b", "route": route, "rate": 3000, "arrival": "binomial", "end": 600}
    scenario = Scenario.from_dict(corridor_doc(flows=[binomial]))  # dt = 1: p = 0.83
    Simulation(scenario)
    with pytest.raises(ScenarioValidationError) as info:
        Simulation(scenario, dt=2.0, duration=600)  # p = 1.67
    assert [(i.code, i.path) for i in info.value.errors] == [("E506", "demand.flows[0].rate")]
    endless = Scenario.from_dict(corridor_doc(flows=[{"id": "e", "route": route, "rate": 300}]))
    Simulation(endless)  # the scenario's duration (600 s) ends it
    with pytest.raises(ScenarioValidationError) as info:
        Simulation(endless, duration=None)
    assert [(i.code, i.path) for i in info.value.errors] == [("E507", "demand.flows[0]")]


def test_endless_run_drains_through_the_watchdog(
    corridor_doc: Callable[..., dict[str, Any]],
) -> None:
    """Review repro: W_J1 lane 1 only reaches J1_J2 lanes 0 and 2, and neither connects to
    J2_E. Without lane changes those vehicles wait at the lane end until the watchdog
    teleports them, so a duration=None run still ends."""
    flow = {"id": "f", "route": ["W_J1", "J1_J2", "J2_E"], "rate": 600, "end": 120}
    doc = corridor_doc(flows=[flow])
    doc["simulation"]["duration"] = None
    sim = Simulation(Scenario.from_dict(doc), deadlock_timeout=60.0)
    s = sim.run(until=3000.0).summary  # bounded, in case it never drains
    assert sim.done and sim.time < 3000.0
    assert s["vehicles.teleported"] > 0
    assert s["vehicles.arrived"] == s["vehicles.generated"] == s["vehicles.inserted"] == 20
    stuck = Simulation(Scenario.from_dict(doc), deadlock_timeout=0.0)  # 0: watchdog off
    s = stuck.run(until=3000.0).summary
    assert not stuck.done and s["vehicles.teleported"] == 0
    assert s["vehicles.arrived"] < s["vehicles.generated"]


# ------------------------------------------------------------------------------- time and done
def test_time_and_end_time() -> None:
    sim = Simulation(_straight(0.0, duration=10.2, dt=0.5))
    assert (sim.time, sim.step_count, sim.end_time) == (0.0, 0, 10.0)
    sim.step(3)
    assert (sim.step_count, sim.time) == (3, 1.5)
    sim.step(100)  # stops at the limit
    assert sim.step_count == 20 and sim.done and sim.is_done()
    with pytest.raises(SimulationError, match="done"):
        sim.step()
    with pytest.raises(SimulationError, match="n must be >= 1"):
        Simulation(_straight(0.0)).step(0)


def test_run_until_and_duration() -> None:
    sim = Simulation(_straight(0.0, duration=20))
    sim.run(until=3.2)
    assert sim.time == 4.0  # the first step boundary at or after `until`
    sim.run(duration=2)
    assert sim.time == 6.0
    sim.run(until=1.0)  # in the past: nothing to do
    assert sim.time == 6.0
    result = sim.run(until=1e6)  # capped by the time limit
    assert sim.time == 20.0 and result.steps == 20
    assert sim.run().steps == 20  # already done: returns the results
    with pytest.raises(ConfigError, match="not both"):
        Simulation(_straight(0.0)).run(until=5, duration=5)
    with pytest.raises(ConfigError, match="finite"):
        Simulation(_straight(0.0)).run(until=math.inf)


def test_duration_none_runs_until_drained() -> None:
    sim = Simulation(_straight(0.0, 30.0, length=300.0, duration=None))
    assert sim.end_time is None and not sim.done and not sim.is_drained()
    result = sim.run()
    assert sim.is_drained() and sim.done
    assert result.summary["vehicles.arrived"] == 2
    arrival = 30.0 + (300.0 - C.VEHICLE_LENGTH) / C.SPEED_LIMIT
    assert sim.time == math.ceil(arrival)  # the step in which the last car arrived


def test_is_drained_ignores_the_time_limit() -> None:
    sim = Simulation(_straight(0.0, length=100.0, duration=3600))
    sim.run(until=30)
    assert sim.is_drained() and not sim.done


# ------------------------------------------------------------------------------- lifecycle
def test_reset_reproduces_and_reseeds(junction: Scenario) -> None:
    sim = Simulation(junction, seed=4)
    first = sim.run(until=120).summary
    sim.reset()
    assert (sim.step_count, sim.time, sim.seed) == (0, 0.0, 4)
    assert sim.metrics.summary()["vehicles.generated"] == 0
    assert sim.run(until=120).summary == first
    sim.reset(seed=5)
    assert sim.seed == 5 and sim.run(until=120).summary != first
    with pytest.raises(ConfigError, match="seed"):
        sim.reset(seed=-1)


def test_corrupted_after_an_interrupted_step(
    junction: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = Simulation(junction)
    sim.step(5)

    def interrupt(self: Engine) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(Engine, "step", interrupt)
        with pytest.raises(KeyboardInterrupt):
            sim.step()
    with pytest.raises(SimulationError, match="corrupted"):
        sim.step()
    with pytest.raises(SimulationError, match="corrupted"):
        sim.run()
    result = sim.get_results()
    assert result.interrupted and result.steps == 5
    sim.reset()
    sim.step()
    assert not sim.get_results().interrupted


def test_close_and_context_manager(junction: Scenario) -> None:
    with Simulation(junction) as sim:
        sim.step()
    for call in (sim.step, sim.run, sim.reset):
        with pytest.raises(SimulationError, match="closed"):
            call()
    sim.close()  # idempotent
    assert sim.get_results().steps == 1
    assert "closed" in repr(sim)


def test_progress_callback_and_bar(junction: Scenario, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[ProgressInfo] = []
    Simulation(junction).run(until=60, progress=seen.append)
    last = seen[-1]
    assert (last.time, last.end_time, last.step_count) == (60.0, 60.0, 60)
    assert last.running >= 0 and last.arrived >= 0 and last.steps_per_s > 0
    Simulation(junction).run(until=5, progress=True)
    assert "Running" in capsys.readouterr().err


def test_run_is_logged(junction: Scenario, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="urbanflow.simulation"):
        Simulation(junction).run(until=10)
    (record,) = [r for r in caplog.records if r.message == "run finished"]
    assert record.__dict__["sim_time"] == 10.0 and "steps_per_s" in record.__dict__


# ------------------------------------------------------------------------------- summary
def test_summary_of_a_single_trip() -> None:
    sim = Simulation(_straight(0.0, duration=120))
    result = sim.run()
    s = result.summary
    travel = (1000.0 - C.VEHICLE_LENGTH) / C.SPEED_LIMIT  # inserted cruising at v0
    assert s["travel_time.mean"] == pytest.approx(travel, abs=1e-9)
    assert s["travel_time.median"] == s["travel_time.p95"] == s["travel_time.mean"]
    assert s["travel_time.std"] == 0.0
    assert s["throughput_vph"] == pytest.approx(3600.0 / 120.0)
    counts = {k: s[k] for k in SUMMARY_KEYS if k.startswith("vehicles.")}
    assert counts == {
        "vehicles.generated": 1,
        "vehicles.inserted": 1,
        "vehicles.arrived": 1,
        "vehicles.en_route": 0,
        "vehicles.backlog": 0,
        "vehicles.removed": 0,
        "vehicles.teleported": 0,
    }
    assert result.event_counts["vehicle_arrived"] == 1
    assert result.event_counts["vehicle_departed"] == 1
    assert set(result.event_counts) >= {"phase_changed", "simulation_started"}


def test_summary_before_any_arrival_is_nan() -> None:
    s = Simulation(_straight(0.0)).metrics.summary()
    assert math.isnan(s["travel_time.mean"]) and math.isnan(s["throughput_vph"])
    assert s["vehicles.generated"] == 0


def test_warmup_excludes_early_trips_and_arrivals() -> None:
    sim = Simulation(_straight(0.0, 100.0, length=300.0, duration=200), metrics={"warmup": 50})
    s = sim.run().summary
    assert s["vehicles.arrived"] == 2  # counts are totals
    travel = (300.0 - C.VEHICLE_LENGTH) / C.SPEED_LIMIT
    assert s["travel_time.mean"] == pytest.approx(travel)  # only the trip departing at 100
    assert s["throughput_vph"] == pytest.approx(3600.0 * 1 / (200 - 50))


def test_conservation_every_step(junction: Scenario) -> None:
    sim = Simulation(junction)
    while not sim.done:
        sim.step()
        s = sim.metrics.summary()
        assert s["vehicles.generated"] == (
            s["vehicles.backlog"]
            + s["vehicles.en_route"]
            + s["vehicles.arrived"]
            + s["vehicles.removed"]
        )
        assert s["vehicles.en_route"] == len(sim.vehicles)
    assert sim.get_results().summary["vehicles.arrived"] > 0


def test_results_identity(junction: Scenario) -> None:
    sim = Simulation(junction, seed=2)
    sim.run(until=30)
    result = sim.get_results()
    assert result.scenario_name == junction.name
    assert result.scenario_hash == junction.content_hash
    assert (result.seed, result.steps, result.sim_time) == (2, 30, 30.0)
    assert result.config == sim.config and not result.interrupted
    assert result.wall_time > 0 and result.run_id is None
    assert (
        np.isfinite(result.summary["travel_time.mean"]) or result.summary["vehicles.arrived"] == 0
    )


# ------------------------------------------------------------------------------- controllers=
def _two_signals() -> Scenario:
    """Signalised junctions A and B on a W-E arterial, each with a N-S cross street."""
    b = ScenarioBuilder("two_signals", duration=300)
    b.boundary("W", (-200.0, 0.0))
    b.boundary("E", (500.0, 0.0))
    for j, x in (("A", 0.0), ("B", 300.0)):
        b.intersection(j, (x, 0.0), kind="signalized")
        for name, dy in (("N", 200.0), ("S", -200.0)):
            b.boundary(f"{j}{name}", (x, dy))
            b.road(f"{j}{name}_in", f"{j}{name}", j)
            b.road(f"{j}{name}_out", j, f"{j}{name}")
    for a, c in (("W", "A"), ("A", "B"), ("B", "E")):
        b.road(f"{a}_{c}", a, c)
        b.road(f"{c}_{a}", c, a)
    b.flow("we", route=["W_A", "A_B", "B_E"], rate=600.0)
    return b.build()


class RandomPhase(ControllerBase):
    """Requests a random phase from ``ctx.rng`` once min green is over."""

    made: ClassVar[list[RandomPhase]] = []

    def __init__(self) -> None:
        RandomPhase.made.append(self)

    def decide(self, ctx: ControllerContext) -> int | None:
        if ctx.green_elapsed < ctx.min_green:
            return None
        return int(ctx.rng.integers(ctx.n_phases))


def _names(sim: Simulation) -> dict[str, str]:
    return {j: sim.signals[j].controller for j in sim.signals.ids}


def test_controllers_star_applies_everywhere_and_ids_win() -> None:
    scenario = _two_signals()
    assert _names(Simulation(scenario)) == {"A": "fixed_time", "B": "fixed_time"}
    sim = Simulation(scenario, controllers={"*": "external"})
    assert _names(sim) == {"A": "external", "B": "external"}
    sim = Simulation(scenario, controllers={"B": "external", "*": "fixed_time"})
    assert _names(sim) == {"A": "fixed_time", "B": "external"}
    sim = Simulation(scenario, controllers={"A": "external"})
    assert _names(sim) == {"A": "external", "B": "fixed_time"}


def test_controller_refs_by_name_mapping_and_factory() -> None:
    scenario = _two_signals()
    offset = {"type": "fixed_time", "params": {"offset": 20.0}}
    sim = Simulation(scenario, controllers={"A": offset, "B": External})
    assert isinstance(sim.signals.controller("A"), FixedTime)
    assert isinstance(sim.signals.controller("B"), External)
    sim.step()  # A starts 48 s into its 68 s cycle (u = -20 mod 68): 14 s into p1's green
    assert (sim.signals["A"].phase_index, sim.signals["A"].stage_elapsed) == (1, 15.0)
    assert (sim.signals["B"].phase_index, sim.signals["B"].stage_elapsed) == (0, 1.0)


def test_reset_rebuilds_the_controllers_deterministically() -> None:
    RandomPhase.made.clear()
    sim = Simulation(_two_signals(), controllers={"*": RandomPhase}, duration=200)
    assert len(RandomPhase.made) == 2  # the factory is called per intersection

    def phases() -> list[list[int]]:
        out = []
        while not sim.done:
            sim.step()
            out.append(sim.signals.phase_indices().tolist())
        return out

    first = phases()
    assert len({tuple(p) for p in first}) > 1  # it does switch
    sim.reset()
    sim.signals.set_controller("A", "external")  # a runtime change that reset() undoes
    sim.reset()
    assert len(RandomPhase.made) == 6 and _names(sim) == {"A": "RandomPhase", "B": "RandomPhase"}
    assert phases() == first  # fresh instances, re-seeded controller:{id} streams
    sim.reset(seed=5)
    assert phases() != first


def test_results_record_the_configured_controllers(tmp_path: Path) -> None:
    """Review finding: results of runs that differ only in ``controllers=`` must differ."""
    scenario = _two_signals()
    plain = Simulation(scenario).get_results()
    assert plain.controllers == {
        "A": {"type": "fixed_time", "params": {"offset": 0.0}},
        "B": {"type": "fixed_time", "params": {"offset": 0.0}},
    }
    offset = {"type": "fixed_time", "params": {"offset": 20}}
    sim = Simulation(scenario, controllers={"*": "external", "A": offset, "B": RandomPhase})
    sim.signals.set_controller("A", "external")  # runtime changes are the command log's
    result = sim.run(until=5)
    assert result.controllers == {
        "A": {"type": "fixed_time", "params": {"offset": 20.0}},
        "B": {"type": "RandomPhase", "params": {}},
    }
    assert result.to_dict()["controllers"] == result.controllers
    assert SimulationResult.load(result.save(tmp_path)).controllers == result.controllers
    assert (
        Simulation(generate("single_intersection", kind="priority")).get_results().controllers == {}
    )


def test_controller_override_errors() -> None:
    scenario = _two_signals()
    with pytest.raises(NotFoundError, match=r'controllers: unknown intersection "AA" \(did you'):
        Simulation(scenario, controllers={"AA": "external"})
    with pytest.raises(ConfigError, match='controllers: intersection "W" is not signalized'):
        Simulation(scenario, controllers={"W": "external"})
    with pytest.raises(ScenarioValidationError) as info:
        Simulation(scenario, controllers={"*": "extrenal", "B": {"type": "fixed_time", "x": 1}})
    assert [(i.code, i.path) for i in info.value.errors] == [
        ("E901", 'controllers["*"]'),
        ("E002", "controllers.B.x"),  # unknown field
    ]
    with pytest.raises(ConfigError, match="one controller instance cannot run several"):
        Simulation(scenario, controllers={"*": External()})
    assert Simulation(scenario, controllers={"A": External(), "B": External()})
    unsignalised = generate("single_intersection", kind="uncontrolled")
    assert Simulation(unsignalised, controllers={"*": "external"}).signals.ids == ()


# ------------------------------------------------------------------------------- sim.events
def test_events_api_delivers_frozen_events_with_ids(junction: Scenario) -> None:
    sim = Simulation(junction, duration=120)
    seen: list[Event] = []
    sub = sim.events.subscribe([EventType.vehicle_inserted, "vehicle_arrived"], seen.append)
    assert isinstance(sub, Subscription) and sub.active
    sim.run()
    counts = sim.events.counts()
    assert set(counts) == set(EventType)
    by_type = {t: [e for e in seen if e.type is t] for t in EventType}
    assert len(by_type[EventType.vehicle_inserted]) == counts[EventType.vehicle_inserted] > 0
    assert len(by_type[EventType.vehicle_arrived]) == counts[EventType.vehicle_arrived] > 0
    assert counts[EventType.vehicle_arrived] == sim.metrics.summary()["vehicles.arrived"]
    assert {e.type for e in seen} == {EventType.vehicle_inserted, EventType.vehicle_arrived}
    first = by_type[EventType.vehicle_inserted][0]
    assert first.vehicle in sim.vehicles.ids(VehicleStatus.arrived) + sim.vehicles.ids()
    assert first.link in sim.network.lane_ids and first.intersection is None
    assert first.data == {} and first.time == (first.step - 1) * sim.dt
    arrival = by_type[EventType.vehicle_arrived][0]
    assert (first.step - 1) * sim.dt <= arrival.time <= arrival.step * sim.dt
    assert counts == {EventType(k): v for k, v in sim.get_results().event_counts.items()}


def test_callbacks_run_after_the_step_and_control_takes_effect_next_step() -> None:
    scenario = _straight(0.0, 5.0)
    sim = Simulation(scenario, duration=200)
    log: list[tuple[int, int, str | None]] = []

    def on_insert(event: Event) -> None:
        # the step is complete: the views already show the new vehicle
        assert sim.step_count == event.step and event.vehicle in sim.vehicles
        log.append((event.step, sim.step_count, event.vehicle))
        sim.vehicles.set_speed(str(event.vehicle), 1.0)

    sim.events.subscribe(EventType.vehicle_inserted, on_insert)
    sim.run(until=20)
    assert [v for *_, v in log] == ["v0", "v1"] and all(a == b for a, b, _ in log)
    assert sim.vehicles["v0"].speed <= 1.0 + 1e-9 and sim.vehicles["v0"].speed_override == 1.0


def test_callback_exceptions_propagate_with_a_consistent_state() -> None:
    sim = Simulation(_straight(0.0, 5.0), duration=200)

    def boom(event: Event) -> None:
        raise RuntimeError(f"bad callback for {event.vehicle}")

    sub = sim.events.subscribe("vehicle_inserted", boom)
    with pytest.raises(RuntimeError, match="bad callback for v0"):
        sim.run()
    assert sim.step_count == 1 and "v0" in sim.vehicles  # the step completed
    assert sim.metrics.summary()["vehicles.inserted"] == 1
    sub.unsubscribe()
    sim.step()  # not corrupted
    assert sim.step_count == 2


def test_no_dispatch_without_subscribers(
    junction: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = Simulation(junction, duration=60)
    calls = []
    monkeypatch.setattr(sim._bus, "dispatch", lambda *_, **__: calls.append(1))
    sim.run(until=30)
    assert calls == []
    sub = sim.events.subscribe(None, lambda _: None)
    sim.step(3)
    assert len(calls) == 3
    sub.unsubscribe()
    sim.step()
    assert len(calls) == 3


def test_subscriptions_survive_reset_and_counts_restart(junction: Scenario) -> None:
    sim = Simulation(junction, duration=60)
    seen: list[Event] = []
    sim.events.subscribe(EventType.vehicle_departed, seen.append)
    sim.run()
    first = len(seen)
    assert first == sim.events.counts()[EventType.vehicle_departed] > 0
    sim.reset()
    assert sim.events.counts()[EventType.vehicle_departed] == 0
    sim.run()
    assert len(seen) == 2 * first  # same seed, same departures


def test_phase_changed_events_decode_the_phase() -> None:
    sim = Simulation(_two_signals(), duration=200)
    seen: list[Event] = []
    sim.events.subscribe(EventType.phase_changed, seen.append)
    sim.run(until=70)
    sim.signals.set_phase("A", 0)
    sim.step()
    assert seen and all(e.intersection in {"A", "B"} and e.link is None for e in seen)
    assert all(e.vehicle is None and set(e.data) == {"phase", "forced"} for e in seen)
    assert seen[-1].data == {"phase": 0, "forced": True} and seen[-1].intersection == "A"
    assert not any(e.data["forced"] for e in seen[:-1])
