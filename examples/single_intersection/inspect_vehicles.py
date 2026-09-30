"""Generate a signalised junction, run it for a while and inspect vehicles, lanes, the
intersection and the vector state.

    uv run python examples/single_intersection/inspect_vehicles.py

Set ``URBANFLOW_EXAMPLE_QUICK=1`` for a short run.
"""

from __future__ import annotations

import os

import numpy as np

from urbanflow import Simulation, generate


def main() -> None:
    quick = os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1"
    scenario = generate("single_intersection", kind="signalized", arms=4, lanes=2)
    sim = Simulation(scenario, seed=3, duration=600)
    sim.run(until=60 if quick else 300)
    print(f"t={sim.time:g} s: {len(sim.vehicles)} running, {sim.vehicles.count('arrived')} arrived")

    # the intersection and its light
    junction = sim.intersections["J"]
    light = junction.signal
    assert light is not None
    print(
        f"\n{junction.id} ({junction.kind.value}): phase {light.phase_id} {light.stage.value} "
        f"for {light.stage_elapsed:g} s, movement states {light.state_string}"
    )
    for mov in junction.movements:
        print(f"  {mov.id:<14} {mov.turn.value:<8} state {mov.state}  rank {mov.rank}")

    # lanes: the longest stop-line queue (vector forms are aligned with sim.lanes.ids)
    queues = sim.lanes.queue_lengths()
    lane = sim.lanes[int(np.argmax(queues))]
    print(f"\nlongest queue: {lane.queue_length} vehicles on {lane.id} (road {lane.road})")
    for vid in lane.vehicle_ids[:5]:  # front to back
        car = sim.vehicles[vid]
        gap_to_line = lane.length - (car.position or 0.0)
        print(
            f"  {car.id:<10} {gap_to_line:6.1f} m to the line  {car.speed:5.2f} m/s  "
            f"waited {car.waiting_time:4.0f} s  next road {car.next_road}"
        )

    # the vector state: one read-only row per running vehicle, keyed by uid
    state = sim.state
    halting = state.speeds < sim.config.halting_speed
    (x0, y0), (x1, y1) = state.xy.min(axis=0), state.xy.max(axis=0)
    print(
        f"\n{state.uids.size} vehicles: mean speed {state.speeds.mean():.2f} m/s, "
        f"{int(halting.sum())} halting, within x {x0:.0f}..{x1:.0f} m, y {y0:.0f}..{y1:.0f} m"
    )
    columns = sim.vehicles.to_columns()  # pandas.DataFrame(columns) works too
    print("columns:", ", ".join(columns))


if __name__ == "__main__":
    main()
