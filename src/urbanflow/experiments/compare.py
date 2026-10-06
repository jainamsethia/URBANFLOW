"""Controller comparison: one scenario x controllers x seeds, run in parallel processes,
summarised with 95 % CIs and paired differences against the first controller (plan Q.5).

Every variant runs on the same seeds; demand is drawn at spawn from per-flow streams, so a
seed gives identical demand for every controller (common random numbers) and paired
differences are valid.

ponytail: an in-memory batch (no experiment store/DB, no resume); the plan's experiment
runner adds persistence and cancellation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbanflow.metrics.manager import METRIC_DIRECTIONS
from urbanflow.metrics.stats import mean_ci, paired_diff

if TYPE_CHECKING:
    from urbanflow.scenario.scenario import Scenario

__all__ = ["DEFAULT_METRICS", "CompareRequest", "compare_controllers", "run_comparison"]

DEFAULT_METRICS: Final = (
    "travel_time.mean",
    "delay.mean",
    "waiting_time_mean",
    "stops_mean",
    "throughput_vph",
    "queue.mean_total_veh",
    "vehicles.arrived",
)


class CompareRequest(BaseModel):
    """A comparison: scenario (bundled name or generator), controllers, seeds, duration."""

    model_config = ConfigDict(extra="forbid")

    scenario: str | None = Field(default=None, max_length=64)
    generator: str | None = Field(default=None, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    controllers: list[str] = Field(min_length=1, max_length=8)
    seeds: int = Field(default=3, ge=1, le=30)
    duration: float | None = Field(default=None, gt=0, le=86400)
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS), max_length=32)

    @model_validator(mode="after")
    def _one_source(self) -> CompareRequest:
        if (self.scenario is None) == (self.generator is None):
            raise ValueError('give exactly one of "scenario" or "generator"')
        if len(set(self.controllers)) != len(self.controllers):
            raise ValueError("controllers must be distinct")
        return self


def _run_one(job: tuple[str, str, int, float | None]) -> dict[str, Any]:
    """Child-process entry point: run one (controller, seed) and return its summary."""
    from urbanflow.scenario.scenario import Scenario
    from urbanflow.simulation import Simulation

    text, controller, seed, duration = job
    overrides: dict[str, Any] = {"seed": seed}
    if duration is not None:
        overrides["duration"] = duration
    sim = Simulation(Scenario.from_json(text), controllers={"*": controller}, **overrides)
    result = sim.run()
    return {
        "controller": controller,
        "seed": seed,
        "wall_time": result.wall_time,
        "summary": dict(result.summary),
    }


def _clean(value: float) -> float | None:
    return value if math.isfinite(value) else None


def run_comparison(req: CompareRequest, max_workers: int = 4) -> dict[str, Any]:
    """Resolve the request's scenario (bundled name or generator) and compare."""
    from urbanflow import Scenario, bundled, generate

    if req.generator is not None:
        scenario, name = generate(req.generator, **req.params), req.generator
    else:
        name = req.scenario or ""
        scenario = Scenario.load(bundled(name))
    return compare_controllers(
        scenario,
        req.controllers,
        seeds=req.seeds,
        duration=req.duration,
        metrics=req.metrics,
        max_workers=max_workers,
        name=name,
    )


def compare_controllers(
    scenario: Scenario,
    controllers: Sequence[str],
    *,
    seeds: int = 3,
    duration: float | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    max_workers: int = 4,
    name: str | None = None,
) -> dict[str, Any]:
    """Run every (controller, seed) pair of ``scenario`` and summarise (JSON-ready)."""
    text = scenario.to_json()
    seed_list = list(range(seeds))
    jobs = [(text, c, s, duration) for s in seed_list for c in controllers]  # seed-major
    workers = max(1, min(max_workers, len(jobs)))
    if workers == 1:
        runs = [_run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            runs = list(pool.map(_run_one, jobs))
    by = {(r["controller"], r["seed"]): r["summary"] for r in runs}
    base = controllers[0]
    table = []
    for metric in metrics:
        values = {
            c: [float(by[(c, s)].get(metric, math.nan)) for s in seed_list] for c in controllers
        }
        row: dict[str, Any] = {
            "metric": metric,
            "direction": METRIC_DIRECTIONS.get(metric),
            "by_controller": {},
            "vs_baseline": {},
        }
        for c, vals in values.items():
            mean, std, lo, hi = mean_ci(vals)
            row["by_controller"][c] = {
                "mean": _clean(mean),
                "std": _clean(std),
                "ci95": [_clean(lo), _clean(hi)],
                "values": [_clean(v) for v in vals],
            }
            if c == base:
                continue
            d, dlo, dhi, n = paired_diff(vals, values[base])
            base_mean = mean_ci(values[base])[0]
            row["vs_baseline"][c] = {
                "delta": _clean(d),
                "delta_pct": _clean(100.0 * d / base_mean) if base_mean else None,
                "ci95": [_clean(dlo), _clean(dhi)],
                "n_pairs": n,
                "significant": bool(
                    math.isfinite(dlo) and math.isfinite(dhi) and (dlo > 0 or dhi < 0)
                ),
            }
        table.append(row)
    return {
        "scenario": name or scenario.name,
        "controllers": list(controllers),
        "baseline": base,
        "seeds": seed_list,
        "duration": duration,
        "metrics": table,
        "runs": [
            {
                "controller": r["controller"],
                "seed": r["seed"],
                "wall_time": r["wall_time"],
                "summary": {k: _clean(float(v)) for k, v in r["summary"].items()},
            }
            for r in runs
        ],
    }
