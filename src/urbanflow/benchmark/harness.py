"""Benchmark measurement (plan S.2).

Every repeat of a case runs in a **fresh spawned process**, one after the other (never
concurrently), so repeats neither share caches nor compete for cores. In the child: build
the scenario and the simulation (timed as ``compile_s``), simulate the warm-up untimed, then
time the remaining steps with ``perf_counter_ns``, summing the running vehicles per step.
The repeats of a case must end in the same state digest; otherwise the case is flagged
``nondeterministic``. With ``profile_dir``, the timed loop of the first repeat is profiled
with cProfile (``<suite>-<case>.prof``, open with ``snakeviz`` or ``pstats``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from urbanflow.benchmark.report import (
    BenchmarkReport,
    CaseResult,
    machine_key,
    now_utc,
    stat,
)
from urbanflow.benchmark.suites import BenchmarkCase

__all__ = ["run_case", "run_suites"]


def _peak_rss_mb() -> float:
    if sys.platform == "win32":
        import psutil

        return float(psutil.Process().memory_info().peak_wset) / 2**20
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru if sys.platform == "darwin" else ru * 1024) / 2**20  # B macOS, KiB Linux


def _measure(case_json: str, quick: bool, profile_path: str | None) -> dict[str, Any]:
    """One repeat, in a child process (JSON in, plain dict out)."""
    import tempfile
    from time import perf_counter_ns

    from urbanflow.scenario.generators import generate
    from urbanflow.scenario.scenario import Scenario
    from urbanflow.simulation import Simulation

    case = BenchmarkCase(**json.loads(case_json))
    warmup, timed = case.durations(quick)
    with tempfile.TemporaryDirectory(prefix="uf-bench-") as tmp:
        t0 = perf_counter_ns()
        scenario = (
            Scenario.load(case.path)
            if case.path is not None
            else generate(str(case.generator), **case.params)
        )
        record = str(Path(tmp) / "bench.ufr") if case.record else False
        sim = Simulation(
            scenario, seed=case.seed, duration=warmup + timed, record=record, **case.config
        )
        compile_s = (perf_counter_ns() - t0) / 1e9
        dt = sim.config.dt
        warm_steps, timed_steps = round(warmup / dt), round(timed / dt)
        sim.step(warm_steps)
        profiler = None
        if profile_path is not None:
            import cProfile

            profiler = cProfile.Profile()
        active_sum, active_max = 0, 0
        start = perf_counter_ns()
        if profiler is not None:
            profiler.enable()
        for _ in range(timed_steps):
            sim.step()
            n = len(sim.vehicles)
            active_sum += n
            active_max = max(active_max, n)
        wall = (perf_counter_ns() - start) / 1e9
        if profiler is not None and profile_path is not None:
            profiler.disable()
            profiler.dump_stats(profile_path)
        digest = sim.state_digest()
        n_lanes = len(sim.network.lane_ids)
        sim.close()
    return {
        "compile_s": compile_s,
        "wall_s": wall,
        "dt": dt,
        "timed_steps": timed_steps,
        "active_sum": active_sum,
        "active_max": active_max,
        "digest": digest,
        "peak_rss_mb": _peak_rss_mb(),
        "n_intersections": sum(
            1 for ix in scenario.resolved.network.intersections if ix.kind != "boundary"
        ),
        "n_lanes": n_lanes,
    }


def run_case(
    case: BenchmarkCase,
    *,
    repeats: int = 5,
    quick: bool = False,
    profile_dir: str | os.PathLike[str] | None = None,
) -> CaseResult:
    """Measure ``case`` ``repeats`` times, each in a fresh spawned process."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    case_json = json.dumps(dataclasses.asdict(case))
    runs: list[dict[str, Any]] = []
    for i in range(repeats):
        profile = None
        if profile_dir is not None and i == 0:
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            profile = str(Path(profile_dir) / f"{case.suite}-{case.name}.prof")
        with ProcessPoolExecutor(1, mp_context=get_context("spawn"), max_tasks_per_child=1) as pool:
            runs.append(pool.submit(_measure, case_json, quick, profile).result())
    first = runs[0]
    dt, steps = first["dt"], first["timed_steps"]
    wall = [r["wall_s"] for r in runs]
    warmup, _ = case.durations(quick)
    return CaseResult(
        suite=case.suite,
        case=case.name,
        params=case.params if case.path is None else {"path": case.path},
        config={**case.config, **({"record": True} if case.record else {})},
        n_intersections=first["n_intersections"],
        n_lanes=first["n_lanes"],
        warmup_s=warmup,
        timed_steps=steps,
        dt=dt,
        mean_active=sum(r["active_sum"] for r in runs) / (steps * len(runs)) if steps else 0.0,
        max_active=max(r["active_max"] for r in runs),
        compile_s=stat([r["compile_s"] for r in runs]),
        steps_per_s=stat([steps / w for w in wall]),
        vehicle_steps_per_s=stat([r["active_sum"] / r["wall_s"] for r in runs]),
        rtf=stat([steps * dt / w for w in wall]),
        wall_s=stat(wall),
        peak_rss_mb=stat([r["peak_rss_mb"] for r in runs]),
        state_digest=first["digest"],
        nondeterministic=len({r["digest"] for r in runs}) > 1,
    )


def run_suites(
    cases: Sequence[BenchmarkCase],
    *,
    repeats: int = 5,
    quick: bool = False,
    profile_dir: str | os.PathLike[str] | None = None,
    on_case: Callable[[CaseResult], None] | None = None,
) -> BenchmarkReport:
    """Measure every case in order and return the report (machine info included)."""
    from urbanflow.experiments.artifacts import environment_info

    results = []
    for case in cases:
        result = run_case(case, repeats=repeats, quick=quick, profile_dir=profile_dir)
        results.append(result)
        if on_case is not None:
            on_case(result)
    return BenchmarkReport(
        created_utc=now_utc(),
        machine={**environment_info(), "key": machine_key()},
        settings={"repeats": repeats, "quick": quick},
        cases=results,
    )
