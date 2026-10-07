from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from urbanflow import Simulation, generate
from urbanflow.scenario import ScenarioBuilder
from urbanflow.signals.controllers import PreemptionParams

PREEMPT = {"type": "preemption", "params": {"inner": "fixed_time"}}


def _junction_with_ambulance() -> object:
    # p0 serves E/W for 30 s from t = 0; the ambulance comes from the north at t = 1
    base = generate("single_intersection", demand_rate=100, duration=200)
    b = ScenarioBuilder.from_scenario(base)
    b.trip("ems", 1.0, route=["N_in", "S_out"], vehicle_type="emergency")
    return b.build()


def _ambulance(controller: object) -> tuple[float, int, list[bool]]:
    sim = Simulation(_junction_with_ambulance(), controllers={"J": controller}, seed=0)
    flags = []
    while not sim.done:
        sim.step()
        flags.append(sim.signals["J"].preempting)
    trips = sim.get_results().tables["trips"]
    i = int(np.flatnonzero(trips["vehicle_id"] == "ems")[0])
    return float(trips["travel_time"][i]), int(trips["stops"][i]), flags


def test_ambulance_gets_green() -> None:
    fixed, fixed_stops, _ = _ambulance("fixed_time")
    served, stops, flags = _ambulance(PREEMPT)
    assert fixed_stops >= 1 and stops == 0
    assert served < fixed - 10
    assert any(flags) and not flags[-1]  # preempted while it approached, released after


def test_params() -> None:
    assert PreemptionParams(inner="max_pressure").inner.type == "max_pressure"
    assert PreemptionParams().inner.type == "fixed_time"
    for bad in ({"inner": "preemption"}, {"inner": "nope"}, {"detection_distance": 0}):
        with pytest.raises(ValidationError):
            PreemptionParams.model_validate(bad)
    with pytest.raises(ValidationError, match="offset"):
        PreemptionParams(inner={"type": "fixed_time", "params": {"offset": "x"}})


def test_snapshot_round_trip() -> None:
    sim = Simulation(_junction_with_ambulance(), controllers={"J": PREEMPT}, seed=0)
    sim.step(8)
    snap = sim.snapshot()
    sim.step(20)
    digest = sim.state_digest()
    sim.restore(snap)
    sim.step(20)
    assert sim.state_digest() == digest


def test_emergency_generator() -> None:
    s = generate("emergency", times=[300.0, 900.0], duration=1500)
    trips = s.resolved.demand.trips
    assert [t.id for t in trips] == ["ems.0", "ems.1"] and trips[0].vehicle_type == "emergency"
    signals = [ix.signal for ix in s.resolved.network.intersections if ix.signal is not None]
    assert signals and all(sig.controller.type == "preemption" for sig in signals)
    plain = generate("emergency", preemption=False)
    assert all(
        ix.signal.controller.type == "fixed_time"
        for ix in plain.resolved.network.intersections
        if ix.signal is not None
    )
    with pytest.raises(Exception, match="times"):
        generate("emergency", times=[4000.0])
