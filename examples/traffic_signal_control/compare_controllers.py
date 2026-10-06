"""Compare the built-in signal controllers on the bundled 3x3 grid (paired over seeds).

    uv run python examples/traffic_signal_control/compare_controllers.py

Every controller runs on the same seeds, so each seed has identical demand and the
differences are paired. Set ``URBANFLOW_EXAMPLE_QUICK=1`` for a short run.
"""

from __future__ import annotations

import os

from urbanflow import Scenario, bundled
from urbanflow.experiments.compare import compare_controllers


def main() -> None:
    quick = os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1"
    result = compare_controllers(
        Scenario.load(bundled("grid_3x3")),
        ["fixed_time", "actuated", "max_pressure"],
        seeds=1 if quick else 3,
        duration=300 if quick else 1800,
        max_workers=1 if quick else 4,
    )
    print(f"{result['scenario']}: {len(result['seeds'])} seed(s), baseline {result['baseline']}")
    for row in result["metrics"]:
        cells = []
        for c in result["controllers"]:
            mean = row["by_controller"][c]["mean"]
            delta = row["vs_baseline"].get(c)
            pct = (
                "" if not delta or delta["delta_pct"] is None else f" ({delta['delta_pct']:+.0f}%)"
            )
            cells.append(f"{c}={mean:.1f}{pct}" if mean is not None else f"{c}=n/a")
        print(f"  {row['metric']:22s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
