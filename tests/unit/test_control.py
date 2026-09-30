"""``sim.signals``: reads, commands and their errors (plan AA 5.4, 5.5, H.2, F.4, AD.1 8.10)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from urbanflow import Scenario, Simulation, bundled, generate
from urbanflow.core.errors import CommandError, ConfigError, NotFoundError
from urbanflow.core.types import SignalState, Stage
from urbanflow.signals import External, FixedTime

GREEN, YELLOW, ALL_RED = Stage.green, Stage.yellow, Stage.all_red


@pytest.fixture(scope="module")
def signalised() -> Scenario:
    """The bundled junction "J": fixed-time p0 / p1 (30 s each), yellow 3, all-red 1,
    min green 5, max green 60, dt 1."""
    return Scenario.load(bundled("single_intersection"))


@pytest.fixture
def sim(signalised: Scenario) -> Simulation:
    return Simulation(signalised, duration=600, debug_checks=True)


def _timeline(sim: Simulation, steps: int) -> list[tuple[Stage, int]]:
    out = []
    for _ in range(steps):
        sim.step()
        view = sim.signals["J"]
        out.append((view.stage, view.phase_index))
    return out


# ------------------------------------------------------------------------------- reading
def test_ids_lookup_and_iteration(sim: Simulation) -> None:
    signals = sim.signals
    assert signals.ids == ("J",) and len(signals) == 1 and "J" in signals and "E" not in signals
    index = sim.network.index("intersection", "J")
    assert signals[index] == signals["J"] == signals[np.int64(index)]
    # membership uses the same resolver as lookups
    assert index in signals and np.int32(index) in signals
    other = sim.network.index("intersection", "E")
    assert other not in signals and 99 not in signals and True not in signals
    assert 1.5 not in signals and None not in signals
    assert [v.phase_id for v in signals] == ["p0"]
    none = Simulation(generate("single_intersection", kind="uncontrolled"))
    assert none.signals.ids == () and list(none.signals) == []
    assert none.signals.phase_indices().size == 0


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("JJ", 'unknown intersection "JJ" (did you mean "J"?)'),
        ("E", 'intersection "E" has no traffic signal; signalized: J'),
        (99, "intersection index 99 out of range"),
        (True, "intersection must be an id or an index, got True"),
    ],
)
def test_unknown_or_unsignalised_intersections(
    sim: Simulation, key: str | int, message: str
) -> None:
    for call in (
        lambda: sim.signals[key],
        lambda: sim.signals.phases(key),
        lambda: sim.signals.controller(key),
        lambda: sim.signals.can_switch(key),
    ):
        with pytest.raises(NotFoundError, match=re.escape(message)):
            call()


def test_phases(sim: Simulation) -> None:
    p0, p1 = sim.signals.phases("J")
    assert (p0.index, p0.id, p1.index, p1.id) == (0, "p0", 1, "p1")
    assert (p0.duration, p0.min_green, p0.max_green) == (30.0, 5.0, 60.0)
    assert p0.green == {
        "E_in->N_out": "G",
        "E_in->S_out": "g",
        "E_in->W_out": "G",
        "W_in->E_out": "G",
        "W_in->N_out": "g",
        "W_in->S_out": "G",
    }
    assert set(p1.green) == {m for m in sim.signals["J"].movement_states if m[0] in "NS"}


def test_vectors_are_read_only_and_cached_per_step(sim: Simulation) -> None:
    signals = sim.signals
    phases, states = signals.phase_indices(), signals.movement_states()
    assert phases.tolist() == [0] and not phases.flags.writeable
    assert not states.flags.writeable and states.size == len(sim.network.movement_ids)
    order = list(sim.signals["J"].movement_states)
    assert order == list(sim.network.movement_ids)  # the only junction
    expected = [SignalState(s).code for s in sim.signals["J"].movement_states.values()]
    assert states.tolist() == expected
    assert signals.phase_indices() is phases  # cached until the next step
    for _ in range(31):  # 30 s of p0 green, then yellow
        sim.step()
    assert signals.phase_indices() is not phases
    assert set(signals.movement_states().tolist()) == {SignalState.y.code, SignalState.r.code}


def test_can_switch_follows_min_green_and_stage(sim: Simulation) -> None:
    seen = []
    for _ in range(34):
        seen.append(sim.signals.can_switch("J"))
        sim.step()
    # t = 0..4: min green (5 s) not reached; 5..30: green; 31..33: yellow / all-red
    assert seen == [False] * 5 + [True] * 26 + [False] * 3


# ------------------------------------------------------------------------------- control
def test_request_phase_needs_a_controller_that_accepts_requests(sim: Simulation) -> None:
    with pytest.raises(CommandError, match=r'set_controller\("J", "external"\)'):
        sim.signals.request_phase("J", 1)
    with pytest.raises(CommandError, match=r'hold_phase\("J", phase\)'):
        sim.signals.request_phase("J", "p1")


def test_request_phase_honours_min_green_yellow_and_all_red(signalised: Scenario) -> None:
    sim = Simulation(signalised, controllers={"J": "external"}, debug_checks=True)
    sim.signals.request_phase("J", "p1")  # at t = 0: min green (5 s) first
    timeline = _timeline(sim, 12)
    assert timeline == [(GREEN, 0)] * 5 + [(YELLOW, 0)] * 3 + [(ALL_RED, 0)] + [(GREEN, 1)] * 3
    sim.signals.request_phase("J", 0)  # the last request wins
    sim.signals.request_phase("J", 1)  # already green: nothing happens
    assert _timeline(sim, 10) == [(GREEN, 1)] * 10


def test_bad_phases(sim: Simulation) -> None:
    hint = r'intersection "J": unknown phase "p00" \(did you mean "p0"\?\)'
    with pytest.raises(NotFoundError, match=hint):
        sim.signals.set_phase("J", "p00")
    with pytest.raises(NotFoundError, match=r"phase 5 does not exist \(2 phases, 0-1\)"):
        sim.signals.hold_phase("J", 5)
    with pytest.raises(CommandError, match="phase must be an index or a phase id"):
        sim.signals.set_phase("J", 1.0)  # type: ignore[arg-type]
    with pytest.raises(NotFoundError, match='unknown intersection "Q"'):
        sim.signals.request_phase("Q", 0)
    with pytest.raises(CommandError, match='"E" has no traffic signal'):
        sim.signals.set_phase("E", 0)


def test_set_phase_jumps_without_intergreen(sim: Simulation) -> None:
    sim.run(until=10)
    sim.signals.set_phase("J", "p1")
    view = sim.signals["J"]  # effective at once (the cache is cleared)
    assert (view.stage, view.phase_index, view.stage_elapsed) == (GREEN, 1, 0.0)
    assert sim.signals.phase_indices().tolist() == [1]
    assert view.state_string == "rrrgGGGGgrrr"
    sim.step()
    view = sim.signals["J"]
    assert (view.phase_index, view.stage_elapsed, view.remaining) == (1, 1.0, 29.0)


def test_hold_phase_parks_the_controller_and_release_resumes_it(sim: Simulation) -> None:
    configured = sim.signals.controller("J")
    assert isinstance(configured, FixedTime)
    sim.run(until=10)
    sim.signals.hold_phase("J", "p1")
    view = sim.signals["J"]
    assert view.held and view.controller == "external" and view.remaining is None
    assert isinstance(sim.signals.controller("J"), External)
    with pytest.raises(CommandError, match="under a manual hold"):
        sim.signals.request_phase("J", 0)
    # min green is long over: yellow 3 s, all-red 1 s, then p1 for as long as it is held
    assert _timeline(sim, 4) == [(YELLOW, 0)] * 3 + [(ALL_RED, 0)]
    assert _timeline(sim, 100) == [(GREEN, 1)] * 100  # fixed_time would switch after 30 s
    sim.signals.hold_phase("J", 0)  # a second hold changes the held phase
    assert _timeline(sim, 5)[-1] == (GREEN, 0)
    sim.signals.release("J")
    view = sim.signals["J"]
    assert not view.held and view.controller == "fixed_time"
    assert sim.signals.controller("J") is configured
    assert view.remaining == 30.0 - view.stage_elapsed  # resumes from the current phase
    with pytest.raises(CommandError, match='"J" has no manual hold to release'):
        sim.signals.release("J")
    log = [(name, args) for _, name, args in sim._engine.commands.command_log]
    assert log == [
        ("hold_phase", {"intersection": "J", "phase": 1}),
        ("hold_phase", {"intersection": "J", "phase": 0}),
        ("release", {"intersection": "J"}),
    ]


def test_set_controller(sim: Simulation) -> None:
    sim.run(until=12)
    sim.signals.set_controller("J", "external")
    assert isinstance(sim.signals.controller("J"), External)
    assert _timeline(sim, 60)[-1] == (GREEN, 0)  # nobody asks: p0 stays green
    sim.signals.set_controller("J", {"type": "fixed_time", "params": {"offset": 5}})
    assert sim.signals["J"].controller == "fixed_time"
    sim.step()  # 73 s of green: the next step switches (offsets apply only at reset)
    assert sim.signals["J"].stage is YELLOW
    sim.signals.set_controller("J", External)  # a factory
    assert isinstance(sim.signals.controller("J"), External)
    with pytest.raises(NotFoundError, match='unknown controllers "fixd_time"'):
        sim.signals.set_controller("J", "fixd_time")
    with pytest.raises(ConfigError, match='invalid parameters for controller "fixed_time"'):
        sim.signals.set_controller("J", {"type": "fixed_time", "params": {"ofset": 1}})
    sim.reset()  # the configured controller again
    assert isinstance(sim.signals.controller("J"), FixedTime)


def test_numpy_integers_are_accepted(signalised: Scenario) -> None:
    """Review repro: RL action arrays and ``phase_indices()`` round-trip into commands."""
    sim = Simulation(signalised, controllers={"J": "external"}, debug_checks=True)
    j = np.int64(sim.network.index("intersection", "J"))
    sim.signals.set_phase("J", np.int64(1))
    assert sim.signals[j].phase_index == 1
    sim.signals.set_phase(j, sim.signals.phase_indices()[0] - 1)  # np.int64(0)
    assert sim.signals["J"].phase_index == 0
    sim.signals.request_phase(np.int32(j), np.uint8(1))
    sim.signals.hold_phase(j, np.int16(0))
    assert sim.signals.phases(j) == sim.signals.phases("J")
    assert sim.signals.can_switch(j) is False and sim.signals.controller(j).name == "external"
    sim.signals.release(j)
    log = [args for _, _, args in sim._engine.commands.command_log]
    assert all(type(a.get("phase", 0)) is int for a in log)  # JSON-safe
    with pytest.raises(CommandError, match="phase must be an index or a phase id"):
        sim.signals.set_phase("J", np.float64(1.0))  # type: ignore[arg-type]
    with pytest.raises(CommandError, match="phase must be an index or a phase id"):
        sim.signals.set_phase("J", np.bool_(True))  # type: ignore[arg-type]


def test_release_drops_the_holds_queued_request(sim: Simulation) -> None:
    """Review repro 2: a hold requested during min green must not act after its release;
    fixed_time keeps p0 for its 30 s."""
    sim.signals.hold_phase("J", "p1")  # at t = 0: min green (5 s) first
    sim.step(2)
    sim.signals.release("J")
    assert _timeline(sim, 12) == [(GREEN, 0)] * 12  # t = 14: still p0
    assert _timeline(sim, 17)[-1] == (YELLOW, 0)  # p0 ends after 30 s


class _AsksForP1:
    """A custom controller that wants p1 at once (queued during min green)."""

    name = "asks_p1"
    Params = External.Params
    accepts_requests = False

    def reset(self, setup: object) -> None:
        pass

    def decide(self, ctx: object) -> int | None:  # noqa: ARG002
        return 1

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, d: object) -> None:
        pass


def test_set_controller_drops_the_previous_controllers_request(signalised: Scenario) -> None:
    """Review repro 1: the new controller (external) was never asked for p1."""
    sim = Simulation(signalised, controllers={"J": _AsksForP1}, debug_checks=True)
    sim.step(2)  # p1 is queued: min green (5 s) not reached
    sim.signals.set_controller("J", "external")
    assert _timeline(sim, 12) == [(GREEN, 0)] * 12


def test_one_instance_cannot_run_two_intersections() -> None:
    from urbanflow.scenario import ScenarioBuilder

    b = ScenarioBuilder("two", dt=1.0)
    b.boundary("W", (-200.0, 0.0))
    b.intersection("J1", (0.0, 0.0), kind="signalized")
    b.intersection("J2", (200.0, 0.0), kind="signalized")
    b.boundary("E", (400.0, 0.0))
    for a, z in (("W", "J1"), ("J1", "J2"), ("J2", "E")):
        b.road(f"{a}_{z}", a, z)
        b.road(f"{z}_{a}", z, a)
    sim = Simulation(b.build())
    shared = sim.signals.controller("J1")
    with pytest.raises(CommandError, match="one controller instance cannot run several"):
        sim.signals.set_controller("J2", shared)
    sim.signals.hold_phase("J2", 0)
    with pytest.raises(CommandError, match="one controller instance cannot run several"):
        sim.signals.set_controller("J1", sim.signals.controller("J2"))  # J2's hold
    parked = sim._engine.parked[sim.network.index("intersection", "J2")]
    with pytest.raises(CommandError, match="one controller instance cannot run several"):
        sim.signals.set_controller("J1", parked)  # parked at J2
    assert sim.signals.controller("J1") is shared  # nothing was reset or replaced
    sim.signals.set_controller("J1", shared)  # re-installing at its own intersection is fine
    sim.signals.set_controller("J2", External)  # a factory: one instance per call
    sim.step()


def test_views_are_snapshots(sim: Simulation) -> None:
    before = sim.signals["J"]
    states = dict(before.movement_states)
    for _ in range(31):
        sim.step()
    assert before.stage is GREEN and dict(before.movement_states) == states
    assert sim.signals["J"].stage is YELLOW
    assert np.array_equal(sim.signals.phase_indices(), [0])
