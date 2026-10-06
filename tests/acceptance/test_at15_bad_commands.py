"""AT-15: bad commands fail with clear errors and the simulation carries on (plan V, F.4).

Unknown vehicles and intersections, a phase out of range and a disconnected ``set_route``
raise ``NotFoundError`` or ``CommandError`` with hints (no CityFlow-style segfault or
abort); nothing is logged or changed, and the run continues to the end with every
invariant checked.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from urbanflow import CommandError, NotFoundError, Scenario, Simulation, bundled

pytestmark = pytest.mark.acceptance


def test_at15_bad_commands() -> None:
    sim = Simulation(Scenario.load(bundled("single_intersection")), duration=300, debug_checks=True)
    sim.run(until=60)
    vid = sim.vehicles.ids()[0]
    route = sim.vehicles[vid].route
    before = sim.state_digest()
    bad: list[tuple[Callable[[], object], type[Exception], str]] = [
        (lambda: sim.vehicles.set_speed(vid + "x", 3.0), NotFoundError, f'did you mean "{vid}"'),
        (lambda: sim.vehicles.remove("nobody"), NotFoundError, 'unknown vehicle "nobody"'),
        (lambda: sim.signals.request_phase("K", 0), NotFoundError, 'unknown intersection "K"'),
        (lambda: sim.signals["JJ"], NotFoundError, 'did you mean "J"'),
        (lambda: sim.signals.set_phase("J", 7), NotFoundError, "phase 7 does not exist"),
        (lambda: sim.signals.hold_phase("J", "p9"), NotFoundError, 'unknown phase "p9"'),
        (lambda: sim.signals.request_phase("J", 1), CommandError, "does not accept phase requests"),
        (lambda: sim.signals.release("J"), CommandError, "has no manual hold to release"),
        (
            lambda: sim.vehicles.set_route(vid, [route[0], route[0]]),
            CommandError,
            "route is not connected: no movement from road",
        ),
        (
            lambda: sim.vehicles.set_route(vid, [route[0], "Z_out"]),
            NotFoundError,
            'unknown road "Z_out"',
        ),
        (lambda: sim.vehicles.set_route(vid, ["N_out"]), CommandError, "must start at"),
        (lambda: sim.vehicles.add(route=["W_in", "W_out"]), CommandError, "not connected"),
    ]
    for call, error, hint in bad:
        with pytest.raises(error) as info:
            call()
        assert hint in str(info.value), (hint, str(info.value))
    assert sim.state_digest() == before and sim._engine.commands.command_log == []
    result = sim.run()
    assert result.sim_time == 300.0 and result.summary["vehicles.arrived"] > 0
