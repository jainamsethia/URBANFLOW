"""Run ``controllers.json`` with the experiment runner: every variant x seed in parallel
worker processes, stored in a database, then compared (paired over the seeds).

    uv run python examples/experiment_comparison/run_experiment.py

Same thing from the CLI (results go to the workspace's runs/ and .urbanflow/urbanflow.db):
    uv run urbanflow experiment run examples/experiment_comparison/controllers.json
    uv run urbanflow experiment compare grid-controllers
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from urbanflow.experiments.compare import compare_experiment
from urbanflow.experiments.runner import ExperimentRunner
from urbanflow.experiments.spec import SeedRange, load_experiment
from urbanflow.experiments.store import ExperimentStore

HERE = Path(__file__).resolve().parent


def main() -> None:
    spec = load_experiment(HERE / "controllers.json")
    if os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1":  # CI: 2 variants, 1 seed, 2 min
        spec = spec.model_copy(
            update={
                "variants": spec.variants[:2],
                "seeds": SeedRange(count=1),
                "base_config": {"duration": 120},
            }
        )
    workspace = Path(tempfile.mkdtemp(prefix="urbanflow-experiment-"))
    store = ExperimentStore.open(f"sqlite:///{(workspace / 'urbanflow.db').as_posix()}")
    runner = ExperimentRunner(workspace, store, max_workers=os.cpu_count())
    experiment_id = runner.submit(spec)
    status = runner.run(experiment_id)
    print(f"experiment {experiment_id}: {status.value}; run directories in {workspace / 'runs'}")
    report = compare_experiment(store, experiment_id)
    base = report["baseline"]
    for row in report["metrics"]:
        if row["metric"] not in ("travel_time.mean", "delay.mean", "throughput_vph"):
            continue
        print(f"\n{row['metric']} (baseline {base}):")
        for name, stats in row["by_controller"].items():
            d = row["vs_baseline"].get(name)
            vs = "" if d is None or d["delta"] is None else f"  {d['delta']:+8.2f} vs {base}"
            print(f"  {name:20s} {stats['mean']:9.2f}{vs}")
    store.close()


if __name__ == "__main__":
    main()
