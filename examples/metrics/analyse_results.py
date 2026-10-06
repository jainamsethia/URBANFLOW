"""Work with the metric tables: timeseries, per-intersection and per-trip records.

    uv run python examples/metrics/analyse_results.py

Tables are plain ``{column: numpy array}`` dicts; ``result.to_*``-style conversions are
not needed for numpy work, and ``result.export(dir, format="parquet")`` writes Parquet
when polars is installed.
"""

from __future__ import annotations

import os

import numpy as np

from urbanflow import Simulation, bundled


def main() -> None:
    duration = 300 if os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1" else 3600
    sim = Simulation.from_scenario(bundled("grid_3x3"), seed=4, duration=duration)
    result = sim.run()
    ts, trips = result.tables["timeseries"], result.tables["trips"]

    peak = int(np.argmax(ts["active"]))
    print(f"peak load: {ts['active'][peak]:.0f} vehicles at t={ts['time'][peak]:.0f} s")
    print(f"mean network queue: {np.nanmean(ts['queue_mean']):.1f} veh")

    delay = trips["delay"]
    print(
        f"trips: {delay.size}, delay p50 {np.percentile(delay, 50):.1f} s, "
        f"p95 {np.percentile(delay, 95):.1f} s"
    )
    by_origin: dict[str, list[float]] = {}
    for origin, d in zip(trips["origin"].tolist(), delay.tolist(), strict=True):
        by_origin.setdefault(origin, []).append(d)
    worst = max(by_origin, key=lambda k: np.mean(by_origin[k]))
    print(f"entry road with the highest mean delay: {worst} ({np.mean(by_origin[worst]):.1f} s)")
    stopped = trips["stops"] > 0
    print(f"{100 * stopped.mean():.0f}% of trips stopped at least once")


if __name__ == "__main__":
    main()
