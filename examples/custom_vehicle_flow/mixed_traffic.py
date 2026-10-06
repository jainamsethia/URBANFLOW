"""Mixed traffic: a custom vehicle type, type mixes, Poisson / binomial / uniform arrivals.

uv run python examples/custom_vehicle_flow/mixed_traffic.py
"""

from __future__ import annotations

import os
from collections import Counter

from urbanflow import ScenarioBuilder, Simulation


def build(duration: float):
    b = ScenarioBuilder("mixed-corridor", duration=duration)
    b.boundary("W", (0.0, 0.0)).intersection("J", (400.0, 0.0)).boundary("E", (800.0, 0.0))
    b.boundary("N", (400.0, 300.0)).boundary("S", (400.0, -300.0))
    b.two_way("W", "J", lanes=2).two_way("J", "E", lanes=2)
    b.two_way("N", "J", lanes=1).two_way("J", "S", lanes=1)
    b.vehicle_type("delivery_van", vclass="truck", length=7.0, width=2.2, max_speed=22.0, accel=1.2)
    b.flow("main", route=["W_J", "J_E"], rate=900, arrival="poisson",
           type_mix={"car": 0.8, "delivery_van": 0.15, "bus": 0.05})  # fmt: skip
    b.flow("side", route=["N_J", "J_S"], rate=300, arrival="binomial")
    b.flow("trucks", route=["E_J", "J_W"], period=20.0, vehicle_type="truck")
    b.signal_all(template="two_phase", green=30)
    return b.build()


def main() -> None:
    duration = 300 if os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1" else 1800
    sim = Simulation(build(duration), seed=8)
    result = sim.run()
    types = Counter(result.tables["trips"]["type"].tolist())
    print(result)
    print("arrived by vehicle type:", dict(types))


if __name__ == "__main__":
    main()
