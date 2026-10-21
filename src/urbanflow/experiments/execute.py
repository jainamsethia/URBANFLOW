"""One experiment run, executed in a worker process (plan Q.2).

The runner's pool starts every worker with :func:`init_worker`, which hands it a progress
queue and a cancel event (spawn-context objects: a ``threading.Event`` cannot cross into a
spawned process). :func:`execute_run` reports ``("started", run_id, pid)`` first, then
``("progress", run_id, fraction)`` about once a second, checks the cancel event and the
run's wall-time limit between chunks of about 1 % of the run, and writes the run directory:
``spec.json``, ``scenario.json``, ``env.json``, the metric tables, an optional
``replay.ufr``, ``result.json`` and ``summary.json`` **last** (its presence marks a finished
run). A failure writes ``error.txt`` with the traceback and is returned, never raised, so
one bad run cannot stop the others.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

__all__ = ["execute_run", "init_worker"]

_PROGRESS: Any = None
_CANCEL: Any = None
PROGRESS_INTERVAL_S = 1.0


def init_worker(progress: Any, cancel: Any) -> None:
    """Pool initializer: the progress queue and the cancel event of this experiment."""
    global _PROGRESS, _CANCEL
    _PROGRESS, _CANCEL = progress, cancel


def _send(message: tuple[Any, ...]) -> None:
    if _PROGRESS is not None:
        with contextlib.suppress(Exception):  # best effort: the result still returns
            _PROGRESS.put_nowait(message)


def _finite(summary: dict[str, float]) -> dict[str, float | None]:
    return {k: (v if isinstance(v, int) or math.isfinite(v) else None) for k, v in summary.items()}


def execute_run(run_json: str) -> dict[str, Any]:
    """Run one :class:`~urbanflow.experiments.spec.RunSpec` (JSON); returns its outcome
    ``{"run_id", "status": completed|failed|cancelled, "wall_time_s", "summary",
    "state_digest", "error"}``."""
    from urbanflow.core.config import SimulationConfig
    from urbanflow.experiments.artifacts import RunArtifacts
    from urbanflow.experiments.spec import RunSpec
    from urbanflow.scenario.scenario import Scenario
    from urbanflow.simulation import Simulation

    run = RunSpec.model_validate_json(run_json)
    out = Path(run.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _send(("started", run.run_id, os.getpid()))
    start = time.perf_counter()
    outcome: dict[str, Any] = {"run_id": run.run_id, "summary": None, "state_digest": None}
    try:
        scenario = Scenario.from_json(run.scenario_json)
        config = SimulationConfig.model_validate(run.config)
        record = str(out / "replay.ufr") if run.record_replay else False
        art = RunArtifacts(run.run_id, out)
        art.write_spec(run.model_dump(mode="json", exclude={"scenario_json", "output_dir"}))
        with Simulation(
            scenario, config, controllers=run.policy.controllers(), record=record
        ) as sim:
            art.write_scenario(scenario)
            art.write_env(sim.config.accel)
            total = None if sim.end_time is None else round(sim.end_time / sim.dt)
            chunk = max(1, (total or 1000) // 100)
            last = -math.inf
            status = "completed"
            while not sim.done:
                sim.step(chunk if total is None else min(chunk, total - sim.step_count))
                if _CANCEL is not None and _CANCEL.is_set():
                    status = "cancelled"
                    break
                now = time.perf_counter()
                if run.timeout_s is not None and now - start > run.timeout_s:
                    raise TimeoutError(f"the run took longer than {run.timeout_s:g} s")
                if now - last >= PROGRESS_INTERVAL_S:
                    fraction = sim.step_count / total if total else math.nan
                    _send(("progress", run.run_id, fraction))
                    last = now
            result = sim.get_results()
        outcome.update(
            status=status,
            summary=_finite(dict(result.summary)),
            state_digest=result.state_digest,
            steps=result.steps,
        )
        if status == "completed":
            result.export(out / "tables", format=run.export_format)
            art.write_result(result)  # result.json, then summary.json last
    except Exception as exc:  # one failing run must not stop the experiment
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        outcome.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    outcome["wall_time_s"] = time.perf_counter() - start
    return outcome
