"""The external controller: queued requests, min-green, last wins (plan H.4)."""

from __future__ import annotations

from collections.abc import Callable

from urbanflow.core.types import Stage
from urbanflow.signals import External, controller_registry

from .conftest import EW, EW_LEFT, NS, Driver, program_of

GREEN, YELLOW, ALL_RED = Stage.green, Stage.yellow, Stage.all_red


def _external(driver: Driver) -> External:
    assert isinstance(driver.controller, External)
    return driver.controller


def test_registered_and_accepts_requests() -> None:
    assert controller_registry.get("external") is External
    assert External.accepts_requests and External.name == "external"


def test_request_waits_for_min_green_then_switches(make_driver: Callable[..., Driver]) -> None:
    net, prog = program_of(min_green=5)
    driver = make_driver(net, prog, "external")
    ext = _external(driver)
    for _ in range(8):  # no request: rests in phase 0
        driver.step()
    assert driver.state() == (GREEN, 0, -1)
    driver.runtime.place(prog.intersection, 0)  # restart the green: 0 s applied
    ext.request_phase(1)
    states = []
    for _ in range(6):
        driver.step()
        states.append(driver.state())
        if driver.state()[0] is GREEN:
            assert ext.pending == 1  # kept (and re-requested) until the switch starts
    assert states == [(GREEN, 0, -1)] * 5 + [(YELLOW, 0, 1)]
    assert ext.pending is None  # cleared when the switch started
    for _ in range(4):
        driver.step()
    assert driver.state() == (GREEN, 1, -1)
    for _ in range(10):  # no new request: stays
        driver.step()
    assert driver.state() == (GREEN, 1, -1)


def test_last_request_wins_and_the_current_phase_is_a_no_op(
    make_driver: Callable[..., Driver],
) -> None:
    net, prog = program_of([{"green": EW}, {"green": NS}, {"green": EW_LEFT}], min_green=0)
    driver = make_driver(net, prog, "external")
    ext = _external(driver)
    ext.request_phase(1)
    ext.request_phase(2)
    driver.step()
    assert driver.state() == (YELLOW, 0, 2) and ext.pending is None
    ext.request_phase(0)
    driver.step()
    assert driver.state() == (YELLOW, 0, 2) and ext.pending == 0
    seen = []
    for _ in range(4):
        driver.step()
        seen.append(driver.state()[:2])
    # green 2 is applied once, then the queued 0 (min green 0); EW_LEFT -> EW loses nothing,
    # so that switch is immediate
    assert seen == [(YELLOW, 0), (ALL_RED, 0), (GREEN, 2), (GREEN, 0)] and ext.pending is None
    ext2 = _external(make_driver(net, prog, "external"))
    ext2.request_phase(0)
    assert ext2.pending == 0


def test_request_during_a_transition_applies_after_the_new_min_green(
    make_driver: Callable[..., Driver],
) -> None:
    net, prog = program_of([{"green": EW}, {"green": NS}, {"green": EW_LEFT}], min_green=3)
    driver = make_driver(net, prog, "external")
    ext = _external(driver)
    for _ in range(3):
        driver.step()
    ext.request_phase(1)
    driver.step()
    assert driver.state() == (YELLOW, 0, 1)
    ext.request_phase(2)  # during the transition: decide is not called until green 1
    seen = []
    for _ in range(12):
        driver.step()
        seen.append(driver.state()[:2])
    assert seen[:6] == [(YELLOW, 0)] * 2 + [(ALL_RED, 0)] + [(GREEN, 1)] * 3
    assert seen[6] == (YELLOW, 1)  # 3 s of green 1 served first


def test_state_dict_round_trip() -> None:
    ext = External()
    assert ext.state_dict() == {"pending": None}
    ext.request_phase(3)
    other = External()
    other.load_state_dict(ext.state_dict())
    assert other.pending == 3 and other.state_dict() == {"pending": 3}
    other.load_state_dict({"pending": None})
    assert other.pending is None
