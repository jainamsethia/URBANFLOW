"""Signal commands (plan F.4, H.2): validation, effect after the next step, holds, the log."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from urbanflow import generate
from urbanflow.core.errors import CommandError, ConfigError, NotFoundError, SimulationError
from urbanflow.core.events import EventType, decode_phase_aux
from urbanflow.core.types import Stage
from urbanflow.engine import Engine
from urbanflow.signals import ControllerBase, ControllerContext, External, FixedTime

MakeEngine = Callable[..., Engine]
J = 0  # the junction's intersection index
GREEN, YELLOW, ALL_RED = Stage.green, Stage.yellow, Stage.all_red


@pytest.fixture
def make(make_engine: MakeEngine) -> Callable[..., Engine]:
    """The 1-lane signalised junction (p0 E-W, p1 N-S; 30 s, yellow 3, all-red 1, min 5)."""
    scenario = generate("single_intersection", lanes=1, demand_rate=300)

    def build(**kwargs: Any) -> Engine:
        e = make_engine(scenario, **kwargs)
        assert e.network.int_ids[J] == "J"
        return e

    return build


def _state(e: Engine) -> tuple[Stage, int]:
    return Stage.from_code(int(e.signals.stage[J])), int(e.signals.phase[J])


def _phase_events(e: Engine) -> list[tuple[int, int, bool]]:
    """``(intersection, phase, forced)`` of this step's ``phase_changed`` events (link -1)."""
    ev = e.events
    sel = ev.type == EventType.phase_changed.code
    assert (ev.link[sel] == -1).all()
    return [
        (j, *decode_phase_aux(aux))
        for j, aux in zip(ev.intersection[sel].tolist(), ev.aux[sel].tolist(), strict=True)
    ]


def test_request_phase_needs_a_controller_that_accepts_requests(
    make: Callable[..., Engine],
) -> None:
    e = make()
    with pytest.raises(CommandError, match="does not accept phase requests") as info:
        e.commands.request_phase("J", 1)
    assert 'set_controller("J", "external")' in str(info.value)
    assert 'hold_phase("J", phase)' in str(info.value)
    assert e.commands.command_log == []


def test_request_phase_honours_min_green_yellow_and_all_red(make: Callable[..., Engine]) -> None:
    e = make(controllers={J: "external"})
    e.step()
    e.commands.request_phase("J", "p1")
    assert e.commands.command_log == [(1, "request_phase", {"intersection": "J", "phase": 1})]
    states, events = [], []
    for _ in range(10):
        e.step()
        states.append(_state(e))
        events += [(e.step_count, *x) for x in _phase_events(e)]
    # green 0 was applied in step 1; min green 5 -> steps 2-5 green, then 3 + 1 intergreen
    assert states == [(GREEN, 0)] * 4 + [(YELLOW, 0)] * 3 + [(ALL_RED, 0)] + [(GREEN, 1)] * 2
    assert events == [(10, J, 1, False)]


@pytest.mark.parametrize(
    ("intersection", "phase", "error", "match"),
    [
        ("JJ", 0, NotFoundError, 'unknown intersection "JJ" \\(did you mean "J"\\?\\)'),
        ("N", 0, CommandError, 'intersection "N" has no traffic signal; signalized: J'),
        (7, 0, NotFoundError, "intersection index 7 out of range"),
        (1.5, 0, CommandError, "intersection must be an id or an index"),
        ("J", "p11", NotFoundError, 'intersection "J": unknown phase "p11" \\(did you mean "p1"'),
        ("J", 2, NotFoundError, 'intersection "J": phase 2 does not exist \\(2 phases, 0-1\\)'),
        ("J", True, CommandError, "phase must be an index or a phase id"),
    ],
)
def test_validation(
    make: Callable[..., Engine], intersection: Any, phase: Any, error: type[Exception], match: str
) -> None:
    e = make(controllers={J: "external"})
    for command in (e.commands.request_phase, e.commands.set_phase, e.commands.hold_phase):
        with pytest.raises(error, match=match):
            command(intersection, phase)
    if phase == 0:
        with pytest.raises(error, match=match):
            e.commands.release(intersection)
        with pytest.raises(error, match=match):
            e.commands.set_controller(intersection, "external")
    assert e.commands.command_log == []


def test_set_phase_is_a_forced_jump(make: Callable[..., Engine]) -> None:
    e = make()
    for _ in range(12):
        e.step()
    e.commands.set_phase(J, "p1")  # debug checks: I10 exempts the forced change
    assert _state(e) == (GREEN, 1) and e.signals.green_elapsed[J] == 0.0
    e.step()
    assert _phase_events(e) == [(J, 1, True)]
    assert e.commands.command_log[-1] == (12, "set_phase", {"intersection": "J", "phase": 1})
    for _ in range(29):  # fixed_time continues from the new phase: 30 s of green 1
        e.step()
        assert _state(e) == (GREEN, 1)
    e.step()
    assert _state(e) == (YELLOW, 1)


def test_hold_parks_fixed_time_and_release_resumes_it(make: Callable[..., Engine]) -> None:
    e = make()
    configured = e.controllers[J]
    assert isinstance(configured, FixedTime)
    for _ in range(3):
        e.step()
    e.commands.hold_phase("J", 1)
    assert e.held[J] and e.parked[J] is configured and isinstance(e.controllers[J], External)
    with pytest.raises(CommandError, match='intersection "J" is under a manual hold'):
        e.commands.request_phase("J", 0)
    states = []
    for _ in range(80):
        e.step()
        states.append(_state(e))
    # min green (5 s) served, then yellow and all-red, then held in green 1 for good
    assert states[:6] == [(GREEN, 0)] * 2 + [(YELLOW, 0)] * 3 + [(ALL_RED, 0)]
    assert states[6:] == [(GREEN, 1)] * 74  # fixed_time would have left after 30 s
    e.commands.hold_phase("J", 1)  # holding again keeps the configured controller parked
    assert e.parked[J] is configured
    e.commands.release("J")
    assert not e.held[J] and e.controllers[J] is configured and J not in e.parked
    e.step()  # green 1 already lasted 74 s >= 30 s: fixed_time switches at once
    assert _state(e) == (YELLOW, 1)
    with pytest.raises(CommandError, match='intersection "J" has no manual hold to release'):
        e.commands.release("J")
    names = [name for _, name, _ in e.commands.command_log]
    assert names == ["hold_phase", "hold_phase", "release"]


def test_set_controller(make: Callable[..., Engine]) -> None:
    e = make()
    e.step()
    e.commands.set_controller("J", "external")
    assert isinstance(e.controllers[J], External)
    e.commands.request_phase("J", 1)
    with pytest.raises(
        NotFoundError,
        match='intersection "J": unknown controllers "fixed_tme" \\(did you mean "fixed_time"',
    ):
        e.commands.set_controller("J", "fixed_tme")
    with pytest.raises(
        ConfigError, match='intersection "J": invalid parameters for controller "fixed_time"'
    ):
        e.commands.set_controller("J", {"type": "fixed_time", "params": {"offset": "late"}})
    # during a hold the new controller replaces the parked one and takes over on release
    e.commands.hold_phase("J", 0)
    e.commands.set_controller("J", {"type": "fixed_time", "params": {"offset": 5}})
    assert isinstance(e.controllers[J], External) and isinstance(e.parked[J], FixedTime)
    e.commands.release("J")
    assert isinstance(e.controllers[J], FixedTime)
    e.commands.set_controller("J", External())
    json.dumps(e.commands.command_log)  # JSON-safe
    refs = [args["ref"] for _, name, args in e.commands.command_log if name == "set_controller"]
    assert refs == [
        "external",
        {"type": "fixed_time", "params": {"offset": 5}},
        {"type": "external"},
    ]


def test_reset_restores_the_configured_controllers(make: Callable[..., Engine]) -> None:
    e = make()
    e.step()
    e.commands.hold_phase("J", 1)
    e.reset()
    assert isinstance(e.controllers[J], FixedTime) and not e.held.any() and not e.parked
    assert _state(e) == (GREEN, 0) and e.commands.command_log == []


class Broken(ControllerBase):
    name = "broken"

    def decide(self, ctx: ControllerContext) -> int | None:  # noqa: ARG002
        raise ZeroDivisionError("oops")


class Wrong(ControllerBase):
    name = "wrong"

    def decide(self, ctx: ControllerContext) -> int | None:
        return 9 if ctx.green_elapsed > 1 else "p1"  # type: ignore[return-value]


def test_controller_errors_name_the_intersection(make: Callable[..., Engine]) -> None:
    e = make(controllers={J: Broken()})
    with pytest.raises(
        SimulationError,
        match='controller "broken" of intersection "J" failed at t=0 s: ZeroDivisionError: oops',
    ):
        e.step()
    e = make(controllers={J: Wrong})
    with pytest.raises(
        SimulationError, match="returned 'p1'; decide\\(\\) must return a phase index"
    ):
        e.step()
    e = make(controllers={J: Wrong})
    e.signals.green_elapsed[J] = 2.0
    with pytest.raises(
        SimulationError, match="requested phase 9, but the intersection has 2 phases"
    ):
        e.step()
