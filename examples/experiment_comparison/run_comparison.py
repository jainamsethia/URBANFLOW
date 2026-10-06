"""A controller experiment on the 4x4 grid: 3 controllers x 5 seeds, saved as JSON.

    uv run python examples/experiment_comparison/run_comparison.py

Same thing from the CLI:
    uv run urbanflow compare grid_4x4 -c fixed_time -c actuated -c max_pressure --seeds 5
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from urbanflow import Scenario, bundled
from urbanflow.experiments.compare import compare_controllers


def main() -> None:
    quick = os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1"
    result = compare_controllers(
        Scenario.load(bundled("grid_4x4")),
        ["fixed_time", "actuated", "max_pressure"],
        seeds=1 if quick else 5,
        duration=240 if quick else 3600,
        max_workers=1 if quick else 6,
    )
    out = Path(tempfile.mkdtemp()) / "comparison.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    delay = next(m for m in result["metrics"] if m["metric"] == "delay.mean")
    for c, stats in delay["by_controller"].items():
        lo, hi = stats["ci95"]
        ci = "" if lo is None else f" [{lo:.1f}, {hi:.1f}]"
        print(f"{c:13s} mean delay {stats['mean']:.1f} s{ci}")
    print(f"full result: {out}")


if __name__ == "__main__":
    main()
