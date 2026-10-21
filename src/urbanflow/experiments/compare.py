"""Controller comparison: one scenario x controllers x seeds, run in parallel processes,
summarised with 95 % CIs and paired differences against the first controller (plan Q.5).

Every variant runs on the same seeds; demand is drawn at spawn from per-flow streams, so a
seed gives identical demand for every controller (common random numbers) and paired
differences are valid.

:func:`compare_experiment` builds the same table from a stored experiment's completed
runs (variants instead of controllers), pairing over the seeds both variants completed;
:func:`timeseries_overlay` averages one column of their timeseries tables over the seeds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbanflow.metrics.manager import GLOBAL_COLUMNS, METRIC_DIRECTIONS
from urbanflow.metrics.stats import mean_ci, paired_diff

if TYPE_CHECKING:
    from urbanflow.experiments.store import ExperimentStore
    from urbanflow.scenario.scenario import Scenario

__all__ = [
    "DEFAULT_METRICS",
    "CompareRequest",
    "compare_controllers",
    "compare_experiment",
    "comparison_csv",
    "run_comparison",
    "summary_table",
    "timeseries_overlay",
]

DEFAULT_METRICS: Final = (
    "travel_time.mean",
    "delay.mean",
    "waiting_time_mean",
    "stops_mean",
    "throughput_vph",
    "queue.mean_total_veh",
    "vehicles.arrived",
)
TIMESERIES_COLUMNS: Final = tuple(c for c in GLOBAL_COLUMNS if c not in ("time", "window_s"))
"""Columns :func:`timeseries_overlay` accepts."""


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


def summary_table(
    by: Mapping[tuple[str, int], Mapping[str, float | None]],
    names: Sequence[str],
    seeds: Sequence[int],
    metrics: Sequence[str],
    baseline: str,
) -> list[dict[str, Any]]:
    """Per metric: mean, std and 95 % CI of every arm over ``seeds`` (``by[(arm, seed)]`` is
    a run's summary; missing runs and values count as NaN and are skipped) and the paired
    difference of every other arm against ``baseline`` over the seeds both have."""
    table = []
    for metric in metrics:
        values = {c: [_value(by.get((c, s)), metric) for s in seeds] for c in names}
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
            if c == baseline:
                continue
            d, dlo, dhi, n = paired_diff(vals, values[baseline])
            base_mean = mean_ci(values[baseline])[0]
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
    return table


def _value(summary: Mapping[str, float | None] | None, metric: str) -> float:
    value = None if summary is None else summary.get(metric)
    return math.nan if value is None else float(value)


def compare_experiment(
    store: ExperimentStore,
    experiment_id: str,
    *,
    metrics: Sequence[str] | None = None,
    baseline: str | None = None,
) -> dict[str, Any]:
    """The comparison table of a stored experiment's completed runs (JSON-ready);
    ``missing`` lists the (variant, seed) runs that did not complete."""
    from urbanflow.core.errors import ConfigError
    from urbanflow.experiments.spec import RunStatus

    rec = store.get_experiment(experiment_id)
    variants = [v.name for v in rec.spec.variants]
    base = baseline or rec.spec.baseline_name
    if base not in variants:
        raise ConfigError(f'baseline "{base}" is not a variant ({", ".join(variants)})')
    seeds = rec.spec.seed_list()
    runs = store.list_runs(experiment_id)
    by = {(r.variant, r.seed): r.summary or {} for r in runs if r.status is RunStatus.completed}
    return {
        "experiment": experiment_id,
        "name": rec.spec.name,
        "variants": variants,
        "baseline": base,
        "seeds": seeds,
        "status": rec.status.value,
        "missing": [f"{r.variant}/s{r.seed}" for r in runs if r.status is not RunStatus.completed],
        "metrics": summary_table(by, variants, seeds, list(metrics or DEFAULT_METRICS), base),
    }


def timeseries_overlay(
    store: ExperimentStore,
    experiment_id: str,
    column: str = "active",
    *,
    max_points: int = 400,
) -> dict[str, Any]:
    """Mean and 95 % CI over seeds of one ``timeseries`` column, per variant, from the
    completed runs' tables (JSON-ready; NaN as None).

    The runs of a variant sample at the same times, so rows align by index; a variant is
    cut to its shortest run and thinned to at most ``max_points`` samples."""
    import numpy as np

    from urbanflow.core.errors import ConfigError
    from urbanflow.experiments.spec import RunStatus
    from urbanflow.metrics.export import read_table

    if column not in TIMESERIES_COLUMNS:
        raise ConfigError(f'unknown column "{column}" (one of: {", ".join(TIMESERIES_COLUMNS)})')
    rec = store.get_experiment(experiment_id)
    done = store.list_runs(experiment_id, status=[RunStatus.completed])
    series = []
    for variant in (v.name for v in rec.spec.variants):
        files = [_timeseries_file(Path(r.artifacts_dir)) for r in done if r.variant == variant]
        tables = [read_table(f) for f in files if f is not None]
        n = min((len(t["time"]) for t in tables), default=0)
        if n == 0:
            continue
        rows = np.arange(0, n, max(1, -(-n // max_points)))
        values = np.stack([t[column][:n][rows] for t in tables])
        stats = [mean_ci(values[:, j].tolist()) for j in range(rows.size)]
        series.append(
            {
                "variant": variant,
                "runs": len(tables),
                "time": [float(x) for x in tables[0]["time"][rows]],
                "mean": [_clean(s[0]) for s in stats],
                "lo": [_clean(s[2]) for s in stats],
                "hi": [_clean(s[3]) for s in stats],
            }
        )
    return {"experiment": experiment_id, "column": column, "series": series}


def _timeseries_file(run_dir: Path) -> Path | None:
    for ext in ("csv", "parquet", "json"):
        path = run_dir / "tables" / f"timeseries.{ext}"
        if path.is_file():
            return path
    return None


def comparison_csv(report: Mapping[str, Any]) -> str:
    """One row per (metric, variant): mean, CI and the paired difference to the baseline."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        ["metric", "variant", "mean", "ci95_lo", "ci95_hi", "delta", "delta_pct",
         "delta_ci95_lo", "delta_ci95_hi", "n_pairs", "significant"]
    )  # fmt: skip
    for row in report["metrics"]:
        for name, stats in row["by_controller"].items():
            d = row["vs_baseline"].get(name)
            paired = (
                [d["delta"], d["delta_pct"], *d["ci95"], d["n_pairs"], d["significant"]]
                if d
                else [""] * 6
            )
            cells = [row["metric"], name, stats["mean"], *stats["ci95"], *paired]
            w.writerow(["" if c is None else c for c in cells])
    return buf.getvalue()


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
    table = summary_table(by, controllers, seed_list, metrics, base)
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
