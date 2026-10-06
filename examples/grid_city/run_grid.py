"""Generate a 4x4 grid with non-uniform block lengths, run it, export the metric tables.

uv run python examples/grid_city/run_grid.py
"""

from __future__ import annotations

import os
import tempfile

from urbanflow import Simulation, generate


def main() -> None:
    quick = os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1"
    scenario = generate(
        "grid",
        rows=4,
        cols=4,
        col_spacing=[250.0, 400.0, 250.0],
        entry_rate=350,
        duration=300 if quick else 3600,
    )
    with Simulation(scenario, controllers={"*": "max_pressure"}, seed=5) as sim:
        result = sim.run()
    print(result)
    out = tempfile.mkdtemp(prefix="urbanflow-grid-")
    for path in result.export(out, format="csv"):
        print(f"wrote {path}")
    busiest = result.tables["intersections"]
    ids, queues = busiest["intersection"], busiest["queue_mean"]
    worst = sorted({str(i) for i in ids}, key=lambda j: -queues[ids == j].mean())[:3]
    print("intersections with the longest mean queues:", ", ".join(worst))


if __name__ == "__main__":
    main()
