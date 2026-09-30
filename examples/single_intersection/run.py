"""Run the bundled signalised junction, watch its traffic light and print the results.

    uv run python examples/single_intersection/run.py

Set ``URBANFLOW_EXAMPLE_QUICK=1`` for a short run.
"""

from __future__ import annotations

import os

from urbanflow import Simulation, bundled


def main() -> None:
    quick = os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1"
    duration, every = (120.0, 10) if quick else (1800.0, 60)  # s, steps between reports
    with Simulation.from_scenario(bundled("single_intersection"), seed=7, duration=duration) as sim:
        print(f"{sim.scenario.name}: controller {sim.signals['J'].controller}")
        for phase in sim.signals.phases("J"):
            green = ", ".join(f"{m} ({s})" for m, s in phase.green.items())
            print(f"  phase {phase.id}: {phase.duration:g} s green for {green}")
        while not sim.done:
            sim.step(every)
            light = sim.signals["J"]
            print(
                f"t={sim.time:6.0f} s  running={len(sim.vehicles):3d}  "
                f"J: {light.phase_id} {light.stage.value:<7} {light.state_string}  "
                f"{light.remaining:4.0f} s left of the stage (cycle {light.cycle:g} s)"
            )
        result = sim.get_results()
    print(result)
    print(f"mean travel time {result.summary['travel_time.mean']:.1f} s")


if __name__ == "__main__":
    main()
