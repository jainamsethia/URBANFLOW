"""Benchmark report: schema, statistics, regression check, files (plan S.2).

Each measured quantity is summarised over the repeats as median and interquartile range.
:func:`compare` flags a case only when its median throughput fell by more than the
threshold **and** the interquartile ranges do not overlap, so ordinary noise between runs
is not reported as a regression.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from urbanflow.core.errors import ConfigError, NotFoundError

__all__ = [
    "BenchmarkReport",
    "CaseResult",
    "Regression",
    "Stat",
    "compare",
    "load_report",
    "machine_key",
    "save_report",
    "stat",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Stat(_Model):
    """Median and quartiles over the repeats, plus every value."""

    median: float
    q1: float
    q3: float
    values: list[float]


def stat(values: list[float]) -> Stat:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return Stat(median=float(median), q1=float(q1), q3=float(q3), values=list(values))


class CaseResult(_Model):
    suite: str
    case: str
    params: dict[str, Any]
    config: dict[str, Any]
    n_intersections: int
    """Intersections that are not at the edge of the map."""
    n_lanes: int
    warmup_s: float
    timed_steps: int
    dt: float
    mean_active: float
    """Running vehicles per timed step (mean over steps and repeats)."""
    max_active: int
    compile_s: Stat
    """Scenario load + validation + network compile + simulation set-up, s."""
    steps_per_s: Stat
    vehicle_steps_per_s: Stat
    """Running vehicles summed over the timed steps, per wall second."""
    rtf: Stat
    """Real-time factor: timed simulated seconds per wall second."""
    wall_s: Stat
    peak_rss_mb: Stat
    state_digest: str
    nondeterministic: bool
    """True when the repeats did not all end in the same state digest."""


class BenchmarkReport(_Model):
    format: Literal["urbanflow.benchmark"] = "urbanflow.benchmark"
    version: Literal["1.0"] = "1.0"
    created_utc: str
    machine: dict[str, Any]
    settings: dict[str, Any]
    cases: list[CaseResult]

    @property
    def machine_key(self) -> str:
        return str(self.machine.get("key", ""))


class Regression(_Model):
    suite: str
    case: str
    metric: str
    baseline: float
    current: float
    change_pct: float


def machine_key() -> str:
    """``<system>-<machine>-py<major><minor>``, e.g. ``windows-arm64-py313``: results are
    compared only between runs with the same key."""
    py = "".join(platform.python_version_tuple()[:2])
    return f"{platform.system()}-{platform.machine()}-py{py}".lower()


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def compare(
    base: BenchmarkReport, new: BenchmarkReport, threshold: float = 0.10
) -> list[Regression]:
    """Cases of ``new`` whose median vehicle-steps/s (steps/s for an empty network) is below
    ``(1 - threshold)`` times ``base``'s, with non-overlapping interquartile ranges."""
    if not 0 < threshold < 1:
        raise ConfigError(f"threshold must be between 0 and 1 (got {threshold})")
    old = {(c.suite, c.case): c for c in base.cases}
    found = []
    for cur in new.cases:
        ref = old.get((cur.suite, cur.case))
        if ref is None:
            continue
        metric = "vehicle_steps_per_s" if ref.vehicle_steps_per_s.median > 0 else "steps_per_s"
        a: Stat = getattr(ref, metric)
        b: Stat = getattr(cur, metric)
        if b.median < a.median * (1 - threshold) and b.q3 < a.q1:
            found.append(
                Regression(
                    suite=cur.suite,
                    case=cur.case,
                    metric=metric,
                    baseline=a.median,
                    current=b.median,
                    change_pct=100 * (b.median / a.median - 1),
                )
            )
    return found


def save_report(report: BenchmarkReport, path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return out


def load_report(path: str | os.PathLike[str]) -> BenchmarkReport:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise NotFoundError(f"no benchmark report at {source}") from None
    try:
        return BenchmarkReport.model_validate_json(text)
    except ValueError as exc:
        raise ConfigError(f"{source} is not an UrbanFlow benchmark report: {exc}") from None
