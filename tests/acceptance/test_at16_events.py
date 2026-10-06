"""AT-16: event subscriptions over a 600 s run (plan V, F.8, AA 5.4).

On the bundled signalised junction (fixed time, every invariant checked) a subscriber to
``vehicle_arrived`` and ``phase_changed`` sees exactly the arrived count of the summary, and
the phase changes it sees are exactly the green starts of the signal timeline read step by
step through ``sim.signals``.
"""

from __future__ import annotations

import pytest

from urbanflow import EventType, Scenario, Simulation, bundled
from urbanflow.core.events import Event
from urbanflow.core.types import Stage

pytestmark = pytest.mark.acceptance


def test_at16_events_match_the_metrics_and_the_signal_timeline() -> None:
    sim = Simulation(Scenario.load(bundled("single_intersection")), duration=600, debug_checks=True)
    events: list[Event] = []
    sim.events.subscribe([EventType.vehicle_arrived, EventType.phase_changed], events.append)
    timeline: list[tuple[int, int]] = []  # green starts: (step, phase)
    view = sim.signals["J"]
    previous = (view.stage, view.phase_index)
    while not sim.done:
        sim.step()
        view = sim.signals["J"]
        now = (view.stage, view.phase_index)
        if now[0] is Stage.green and now != previous:
            timeline.append((sim.step_count, view.phase_index))
        previous = now

    arrived = [e for e in events if e.type is EventType.vehicle_arrived]
    assert len(arrived) == sim.metrics.summary()["vehicles.arrived"] > 0
    assert len(arrived) == sim.events.counts()[EventType.vehicle_arrived]
    assert len({e.vehicle for e in arrived}) == len(arrived)
    assert sorted(e.vehicle for e in arrived) == sorted(sim.vehicles.ids("arrived"))
    assert all((e.step - 1) * sim.dt <= e.time <= e.step * sim.dt for e in arrived)

    phases = [e for e in events if e.type is EventType.phase_changed]
    assert [(e.step, e.data["phase"]) for e in phases] == timeline
    assert len(timeline) == 17  # 600 s of a 68 s two-phase cycle: a green start every 34 s
    assert all(e.intersection == "J" and not e.data["forced"] for e in phases)
    assert len(phases) == sim.events.counts()[EventType.phase_changed]
