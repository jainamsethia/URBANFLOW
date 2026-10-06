"""Record a run to a .ufr replay, then read frames back at arbitrary steps.

    uv run python examples/replay/record_and_read.py

Open the file in the workbench with ``uv run urbanflow replay <file> --view``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from urbanflow import Simulation, bundled
from urbanflow.replay import ReplayReader


def main() -> None:
    duration = 120 if os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1" else 900
    path = Path(tempfile.mkdtemp()) / "grid.ufr"
    sim = Simulation.from_scenario(bundled("grid_3x3"), seed=2, duration=duration, record=path)
    sim.run()
    path = sim.stop_recording()
    print(f"recorded {path} ({path.stat().st_size / 1e6:.2f} MB)")

    replay = ReplayReader(path)
    print(f"{replay.n_frames} frames, {replay.manifest['n_vehicles']} vehicles")
    vehicles = replay.vehicles()
    for step in (duration // 4, duration // 2, duration):
        frame = replay.frame(step)
        stopped = int((frame.speed < 0.1).sum())
        some = ", ".join(vehicles[int(u)][0] for u in frame.uid[:3])
        print(f"step {frame.step:4d}: {frame.n:3d} vehicles ({stopped} stopped), e.g. {some}")
    summary = replay.summary() or {}
    print("mean travel time:", round(summary.get("summary", {}).get("travel_time.mean", 0), 1), "s")


if __name__ == "__main__":
    main()
