"""The AT-09 / AT-10 run: a platoon held at a red light, then released (public API only)."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest

from urbanflow import ScenarioBuilder, Simulation

ARMS = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}
MOVEMENT, LANE = "W_in->E_out", "W_in_0"  # approach A: W -> E, one lane
PLATOON = 50
HOLD_AT = 50.0  # s; the platoon arrives from ~29 s at 2 s headways
RED_S = 120.0


@dataclass
class QueueRun:
    """What AT-09 and AT-10 check."""

    sim: Simulation
    dt: float
    crossed_in_yellow: set[int] = field(default_factory=set)
    """Uids that left approach A during a yellow or all-red step."""
    committed_at_red: set[int] = field(default_factory=set)
    """Uids on approach A committed to cross when its light turned red."""
    crossed_in_red: set[int] = field(default_factory=set)
    """Uids that left approach A during a red step."""
    queue: int = 0
    """Vehicles on approach A at the end of the red."""
    front_position: float = 0.0
    front_speed: float = 0.0
    lane_length: float = 0.0
    green_time: float = 0.0
    """Start of the first green step after the hold."""
    first_move_time: float = 0.0
    """First time the queue front was seen moving (> 0.1 m/s) after the green started."""
    crossings: list[float] = field(default_factory=list)
    """End times of the steps in which each vehicle left approach A, in order."""

    def discharge_vph(self) -> float:
        """Saturation flow from the headways of the 5th to the 15th queued vehicle."""
        headways = np.diff(np.asarray(self.crossings))[4:15]
        return float(3600.0 / headways.mean())


def _on_lane(sim: Simulation, lane: int, *, committed: bool = False) -> set[int]:
    raw = sim.state.raw  # live, read-only slot views
    on = raw["active"] & (raw["link"] == lane)
    if committed:
        on &= raw["committed"]
    return set(raw["uid"][on].tolist())


def _red_then_green(dt: float) -> QueueRun:
    """A 1-lane 4-arm junction (arms 400 m) under ``external`` control, W -> E green; 50
    vehicles W -> E at 1800 veh/h. At 50 s, while the platoon streams through, a manual
    hold turns W -> E red (yellow, all-red) for 120 s; a second hold then gives it green
    until the queue has cleared."""
    b = ScenarioBuilder("red_hold", dt=dt, duration=1200.0)
    b.intersection("J", (0.0, 0.0), kind="signalized")
    for name, (dx, dy) in ARMS.items():
        b.boundary(name, (dx * 400.0, dy * 400.0))
        b.road(f"{name}_in", name, "J")
        b.road(f"{name}_out", "J", name)
    b.flow("we", route=["W_in", "E_out"], rate=1800.0, count=PLATOON)
    sim = Simulation(b.build(), controllers={"J": "external"}, debug_checks=True)
    run = QueueRun(sim, dt)
    phases = sim.signals.phases("J")
    red = next(p.index for p in phases if MOVEMENT not in p.green)
    green = next(p.index for p in phases if MOVEMENT in p.green)
    lane = sim.network.index("lane", LANE)
    run.lane_length = sim.lanes[LANE].length
    assert sim.signals["J"].movement_states[MOVEMENT] == "G"
    sim.run(until=HOLD_AT)

    sim.signals.hold_phase("J", red)  # yellow and all-red first
    red_steps = 0
    while red_steps * dt < RED_S - 1e-9:
        before = sim.signals["J"].movement_states[MOVEMENT]
        on_lane, committed = _on_lane(sim, lane), _on_lane(sim, lane, committed=True)
        sim.step()
        crossed = on_lane - _on_lane(sim, lane)
        if sim.signals["J"].movement_states[MOVEMENT] == "r":  # applied during this step
            if before != "r":
                run.committed_at_red = committed
            red_steps += 1
            run.crossed_in_red |= crossed
        else:
            run.crossed_in_yellow |= crossed
    queue = sim.lanes[LANE]
    run.queue = queue.vehicle_count
    front = sim.vehicles[queue.vehicle_ids[0]]
    run.front_position, run.front_speed = front.position or 0.0, front.speed

    sim.signals.hold_phase("J", green)
    while len(run.crossings) < run.queue:
        on_lane = _on_lane(sim, lane)
        moving = sim.vehicles[front.id].speed > 0.1 if front.id in sim.vehicles else True
        if run.green_time and not run.first_move_time and moving:
            run.first_move_time = sim.time
        sim.step()
        if not run.green_time and sim.signals["J"].movement_states[MOVEMENT] == "G":
            run.green_time = sim.time - dt  # the green's first step started here
        run.crossings += [sim.time] * len(on_lane - _on_lane(sim, lane))
        assert sim.time < 1000.0, "the queue did not clear"
    return run


@pytest.fixture(scope="session")
def red_then_green() -> Callable[[float], QueueRun]:
    """``red_then_green(dt)``: the AT-09/AT-10 run, computed once per ``dt``."""
    return functools.cache(_red_then_green)
