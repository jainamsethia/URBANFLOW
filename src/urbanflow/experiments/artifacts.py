"""Run directories ``<workspace>/runs/<run_id>/`` and environment info (plan Q.3, AC 7.2).

A run directory holds ``spec.json`` (what was run: scenario identity, seed, resolved
config), ``scenario.json`` (an exact copy of the scenario), ``env.json``
(:func:`environment_info`), ``result.json`` (the lossless ``SimulationResult``) and
``summary.json``, the nested metric summary. ``summary.json`` is written **last**: its
presence marks a finished (or cleanly interrupted) run. Metric tables and replays join
the layout with the metrics and replay subsystems.
"""

from __future__ import annotations

import math
import os
import platform
import re
import secrets
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import psutil

from urbanflow._version import __version__
from urbanflow.core import constants as C
from urbanflow.core.errors import ConfigError
from urbanflow.results import RESULT_FILE, SimulationResult
from urbanflow.scenario.io import write_json
from urbanflow.scenario.scenario import Scenario

__all__ = ["RunArtifacts", "environment_info", "new_run_id", "summary_document"]

SPEC_FILE: Final = "spec.json"
SCENARIO_FILE: Final = "scenario.json"
ENV_FILE: Final = "env.json"
SUMMARY_FILE: Final = "summary.json"


def new_run_id(name: str, seed: int, *, now: datetime | None = None) -> str:
    """``<UTC yyyymmddThhmmss>-<name slug>-s<seed>-<4 hex>``, e.g.
    ``20260923T141502-grid-3x3-s7-a1f0``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[: C.RUN_ID_SLUG_MAX].strip("-")
    return f"{stamp}-{slug or 'run'}-s{seed}-{secrets.token_hex(C.RUN_ID_SUFFIX_BYTES)}"


def environment_info(accel: str = "numpy") -> dict[str, Any]:
    """Machine and software of this process (``env.json``)."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "numpy": metadata.version("numpy"),
        "urbanflow": __version__,
        "accel": accel,
    }


def summary_document(result: SimulationResult) -> dict[str, Any]:
    """``summary.json``: run identity plus the summary nested by its dotted keys (J.6)."""
    doc: dict[str, Any] = {
        "format": "urbanflow.metrics.summary",
        "version": "1.0",
        "run_id": result.run_id,
        "scenario_hash": result.scenario_hash,
        "seed": result.seed,
        "dt": result.config.dt,
        "duration_s": result.sim_time,
        "warmup_s": result.config.metrics.warmup,
        "steps": result.steps,
        "interrupted": result.interrupted,
    }
    for key, value in result.summary.items():
        *parents, leaf = key.split(".")
        node = doc
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = None if isinstance(value, float) and math.isnan(value) else value
    return doc


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """The files of one run directory (create with :meth:`create`)."""

    run_id: str
    directory: Path

    @classmethod
    def create(
        cls,
        runs_dir: str | os.PathLike[str],
        *,
        name: str,
        seed: int,
        directory: str | os.PathLike[str] | None = None,
    ) -> RunArtifacts:
        """A new run directory ``runs_dir/<run_id>/``, or exactly ``directory`` (which must
        be empty or absent: ``ConfigError`` otherwise)."""
        if directory is not None:
            target = Path(directory)
            if target.exists() and (not target.is_dir() or any(target.iterdir())):
                raise ConfigError(f"output directory {target} exists and is not empty")
            target.mkdir(parents=True, exist_ok=True)
            return cls(new_run_id(name, seed), target)
        root = Path(runs_dir)
        root.mkdir(parents=True, exist_ok=True)
        while True:
            run_id = new_run_id(name, seed)
            try:
                (root / run_id).mkdir()
            except FileExistsError:  # same second, same 4 hex: draw again
                continue
            return cls(run_id, root / run_id)

    @property
    def spec_path(self) -> Path:
        return self.directory / SPEC_FILE

    @property
    def scenario_path(self) -> Path:
        return self.directory / SCENARIO_FILE

    @property
    def env_path(self) -> Path:
        return self.directory / ENV_FILE

    @property
    def result_path(self) -> Path:
        return self.directory / RESULT_FILE

    @property
    def summary_path(self) -> Path:
        return self.directory / SUMMARY_FILE

    def write_spec(self, spec: Mapping[str, Any]) -> Path:
        """``spec.json``: ``spec`` (JSON-ready) with this run's id first."""
        return write_json(self.spec_path, {"run_id": self.run_id, **spec})

    def write_scenario(self, scenario: Scenario) -> Path:
        """``scenario.json``: a byte copy of the scenario file (its JSON if it has none)."""
        if scenario.path is not None and scenario.path.is_file():
            shutil.copyfile(scenario.path, self.scenario_path)
            return self.scenario_path
        return scenario.save(self.scenario_path)

    def write_env(self, accel: str = "numpy") -> Path:
        """``env.json``: :func:`environment_info`."""
        return write_json(self.env_path, environment_info(accel))

    def write_result(self, result: SimulationResult) -> SimulationResult:
        """``result.json``, then ``summary.json`` last; returns ``result`` with its run id."""
        stamped = replace(result, run_id=self.run_id)
        stamped.save(self.directory)
        write_json(self.summary_path, summary_document(stamped))
        return stamped
