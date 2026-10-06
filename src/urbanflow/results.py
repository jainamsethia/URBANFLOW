"""The result of a run: identity, config, summary metrics and provenance (plan AA 5.8).

``SimulationResult.summary`` uses the flattened dotted keys of the metrics summary (B.2
#13): ``vehicles.generated``, ``travel_time.mean``, ``throughput_vph`` and so on. A result
saves to ``<directory>/result.json`` and loads back losslessly (NaN is stored as null).
``state_digest`` identifies the final engine state (equal on the same platform exactly when
two runs ended in the same state). ``tables`` holds the metrics tables (``timeseries``,
``intersections``, ``trips``); :meth:`SimulationResult.export` writes them as CSV, JSON or
Parquet, and :meth:`save` stores them as CSV next to ``result.json``.
"""

from __future__ import annotations

import json
import math
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from urbanflow._version import __version__
from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import ConfigError, NotFoundError
from urbanflow.metrics.export import export_tables, read_table
from urbanflow.scenario.io import write_json

__all__ = ["RESULT_FILE", "SimulationResult", "provenance"]

RESULT_FILE: Final = "result.json"
_FORMAT: Final = "urbanflow.result"
_VERSION: Final = "1.0"


def provenance() -> dict[str, str]:
    """Versions and platform of this process: urbanflow, numpy, python, platform, machine."""
    return {
        "urbanflow": __version__,
        "numpy": metadata.version("numpy"),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _json_number(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else value


def _count(value: float) -> str:
    return f"{int(value):,}"


def _seconds(value: float, fmt: str = ".1f") -> str:
    return "n/a" if math.isnan(value) else f"{value:,{fmt}} s"


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """A comparable, serialisable run artifact (``Simulation.get_results()``)."""

    scenario_name: str
    scenario_hash: str
    """``Scenario.content_hash`` (sha256 hex)."""
    config: SimulationConfig
    """The resolved configuration of the run."""
    seed: int
    sim_time: float
    """Simulated time reached, s."""
    steps: int
    wall_time: float
    """Wall-clock seconds spent stepping."""
    interrupted: bool
    """The run was stopped by Ctrl-C (partial results)."""
    summary: Mapping[str, float]
    """Flattened metric summary (``travel_time.mean`` ...); NaN where undefined."""
    event_counts: Mapping[str, int]
    """Events per ``EventType`` name, cumulative since reset."""
    provenance: Mapping[str, str]
    """urbanflow, numpy, python, platform and machine of the producing process."""
    run_id: str | None = None
    """Set when the run was saved as run artifacts (CLI ``run``)."""
    controllers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """The configured signal controller of each signalised intersection, ``{id: {"type",
    "params"}}`` (the scenario's, or the ``controllers=`` override; a custom class or
    factory by its name, with its validated parameters). Commands issued during the run
    (``set_controller``, holds) are not included."""
    state_digest: str = ""
    """sha256 hex of the engine state at the end (``Simulation.state_digest()``; empty in
    results saved before the field existed, and after an interrupted step corrupted the
    state)."""
    tables: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict, repr=False)
    """Metrics tables by name: ``timeseries``, ``intersections``, ``trips`` (column ->
    array)."""

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (NaN summary values become null)."""
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
            "sim_time": self.sim_time,
            "steps": self.steps,
            "wall_time": self.wall_time,
            "interrupted": self.interrupted,
            "summary": {k: _json_number(v) for k, v in self.summary.items()},
            "event_counts": dict(self.event_counts),
            "provenance": dict(self.provenance),
            "controllers": {k: dict(v) for k, v in self.controllers.items()},
            "state_digest": self.state_digest,
            "config": self.config.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimulationResult:
        """Inverse of :meth:`to_dict`; ``ConfigError`` if ``data`` is not a result."""
        if data.get("format") != _FORMAT:
            raise ConfigError(f'not an UrbanFlow result: "format" must be "{_FORMAT}"')
        try:
            return cls(
                scenario_name=str(data["scenario_name"]),
                scenario_hash=str(data["scenario_hash"]),
                config=SimulationConfig.model_validate(data["config"]),
                seed=int(data["seed"]),
                sim_time=float(data["sim_time"]),
                steps=int(data["steps"]),
                wall_time=float(data["wall_time"]),
                interrupted=bool(data["interrupted"]),
                summary={k: math.nan if v is None else v for k, v in dict(data["summary"]).items()},
                event_counts={k: int(v) for k, v in dict(data["event_counts"]).items()},
                provenance={k: str(v) for k, v in dict(data["provenance"]).items()},
                run_id=data.get("run_id"),
                controllers={str(k): dict(v) for k, v in dict(data.get("controllers", {})).items()},
                state_digest=str(data.get("state_digest", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"malformed UrbanFlow result: {exc}") from None

    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Write ``<directory>/result.json`` (atomically); returns the directory."""
        target = Path(directory)
        export_tables(self.tables, target, "csv")
        write_json(target / RESULT_FILE, self.to_dict())
        return target

    def export(
        self,
        directory: str | os.PathLike[str],
        *,
        format: Literal["csv", "json", "parquet"] = "csv",
        tables: tuple[str, ...] | None = None,
    ) -> list[Path]:
        """Write the metrics tables (all, or ``tables``) as ``<directory>/<name>.<format>``.

        Parquet needs the ``data`` extra (``MissingDependencyError`` otherwise)."""
        chosen = {k: v for k, v in self.tables.items() if tables is None or k in tables}
        return export_tables(chosen, directory, format)

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> SimulationResult:
        """Read a result saved by :meth:`save` (``NotFoundError`` if there is none)."""
        path = Path(directory) / RESULT_FILE
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise NotFoundError(f"no saved result in {directory} ({RESULT_FILE} missing)") from None
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ConfigError(f"{path}: invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a JSON object")
        result = cls.from_dict(data)
        found = {
            name: read_table(Path(directory) / f"{name}.csv")
            for name in ("timeseries", "intersections", "trips")
            if (Path(directory) / f"{name}.csv").is_file()
        }
        return result if not found else _replace_tables(result, found)

    # ------------------------------------------------------------------ display
    @property
    def title(self) -> str:
        """``Results: <name> (seed <n>, <t> s)``, marked when interrupted."""
        tail = ", interrupted" if self.interrupted else ""
        return f"Results: {self.scenario_name} (seed {self.seed}, {self.sim_time:g} s{tail})"

    def rows(self) -> list[tuple[str, str]]:
        """``(metric, formatted value)`` rows of the results table (CLI and ``str``)."""
        s = self.summary
        return [
            (
                "Vehicles departed / arrived",
                f"{_count(s['vehicles.generated'])} / {_count(s['vehicles.arrived'])}",
            ),
            (
                "Vehicles en route / waiting",
                f"{_count(s['vehicles.en_route'])} / {_count(s['vehicles.backlog'])}",
            ),
            (
                "Throughput",
                "n/a" if math.isnan(s["throughput_vph"]) else f"{s['throughput_vph']:,.0f} veh/h",
            ),
            ("Mean travel time", _seconds(s["travel_time.mean"])),
            *(
                (label, fmt(s[key]))
                for label, key, fmt in (
                    ("Mean delay", "delay.mean", _seconds),
                    ("Mean waiting time", "waiting_time_mean", _seconds),
                    ("Mean stops per vehicle", "stops_mean", lambda v: f"{v:.2f}"),
                    ("Mean network queue", "queue.mean_total_veh", lambda v: f"{v:.1f} veh"),
                )
                if key in s and not math.isnan(s[key])
            ),
            ("Teleports", _count(s["vehicles.teleported"])),
            ("Wall time", _seconds(self.wall_time, ".2f")),
        ]

    def __str__(self) -> str:
        rows = [("Metric", "Value"), *self.rows()]
        left = max(len(name) for name, _ in rows)
        right = max(len(value) for _, value in rows)
        lines = [self.title, *(f"{name:<{left}}  {value:>{right}}" for name, value in rows)]
        return "\n".join(lines)


def _replace_tables(
    result: SimulationResult, tables: Mapping[str, Mapping[str, np.ndarray]]
) -> SimulationResult:
    from dataclasses import replace

    return replace(result, tables=dict(tables))
