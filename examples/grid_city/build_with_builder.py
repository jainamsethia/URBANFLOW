"""Build a 2x2 signalised grid by hand with ScenarioBuilder, validate it, run it.

    uv run python examples/grid_city/build_with_builder.py

Movements, lane mappings and signal programs are derived automatically; validation
errors would name the exact JSON path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from urbanflow import ScenarioBuilder, Simulation

S = 300.0  # block length, m


def build() -> ScenarioBuilder:
    b = ScenarioBuilder("grid-2x2", duration=3600)
    for r in range(2):
        for c in range(2):
            b.intersection(f"J{r}{c}", (c * S, r * S))
        b.boundary(f"W{r}", (-S / 2, r * S)).boundary(f"E{r}", (1.5 * S, r * S))
    for c in range(2):
        b.boundary(f"S{c}", (c * S, -S / 2)).boundary(f"N{c}", (c * S, 1.5 * S))
    for r in range(2):
        b.two_way(f"W{r}", f"J{r}0", lanes=2).two_way(f"J{r}0", f"J{r}1", lanes=2)
        b.two_way(f"J{r}1", f"E{r}", lanes=2)
    for c in range(2):
        b.two_way(f"S{c}", f"J0{c}", lanes=2).two_way(f"J0{c}", f"J1{c}", lanes=2)
        b.two_way(f"J1{c}", f"N{c}", lanes=2)
    for r in range(2):
        b.flow(
            f"we{r}", origin=f"W{r}_J{r}0", destination=f"J{r}1_E{r}", rate=500, arrival="poisson"
        )
        b.flow(
            f"ew{r}", origin=f"E{r}_J{r}1", destination=f"J{r}0_W{r}", rate=500, arrival="poisson"
        )
    for c in range(2):
        b.flow(
            f"sn{c}", origin=f"S{c}_J0{c}", destination=f"J1{c}_N{c}", rate=400, arrival="poisson"
        )
    b.signal_all(template="two_phase", green=30)
    return b


def main() -> None:
    scenario = build().build()
    print(scenario.name, scenario.summary())
    out = Path(tempfile.mkdtemp()) / "grid2x2.json"
    scenario.save(out)
    print(f"saved {out}")
    duration = 300 if os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1" else 1800
    result = Simulation(scenario, duration=duration, seed=1).run()
    print(result)


if __name__ == "__main__":
    main()
