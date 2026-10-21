"""Experiment specifications and their expansion into runs (plan Q.1).

An experiment file (``format: "urbanflow.experiment"``) names one scenario, a base config,
variants (each a config override, an optional scenario of its own, a signal-control policy
and a demand scale) and seeds. :func:`expand` turns it into one :class:`RunSpec` per
(seed, variant), **seed-major**, so a partial or cancelled experiment stays balanced across
variants. Every variant runs on the same seeds; demand is drawn at spawn from per-flow
streams, so a seed gives identical demand to every variant (common random numbers) and
paired comparisons are valid.

Each run's ``content_key`` hashes what determines its outcome (scenario content, config,
policy, seed, urbanflow major.minor): resuming an experiment skips runs whose key already
completed.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from urbanflow._version import __version__
from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import ConfigError, ValidationIssue, format_path

__all__ = [
    "ExperimentSpec",
    "ExperimentStatus",
    "PolicySpec",
    "RunSpec",
    "RunStatus",
    "ScenarioRef",
    "SeedRange",
    "VariantSpec",
    "expand",
    "load_experiment",
]

SLUG = r"^[a-z0-9][a-z0-9._-]{0,63}$"
"""Experiment and variant names (they become file and table names)."""


class ExperimentStatus(StrEnum):
    draft = "draft"
    queued = "queued"
    running = "running"
    completed = "completed"
    partial = "partial"
    failed = "failed"
    cancelled = "cancelled"
    interrupted = "interrupted"


class RunStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    interrupted = "interrupted"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)


class ScenarioRef(_Model):
    """Exactly one of a workspace file, a bundled scenario or a generator."""

    path: str | None = None
    """Scenario file, relative to the workspace."""
    bundled: str | None = None
    """Bundled scenario name."""
    generator: str | None = None
    """Generator name (with ``params``)."""
    params: dict[str, Any] = Field(default_factory=dict)
    """Generator parameters."""
    demand_scale: float = Field(default=1.0, gt=0)
    """Multiplies every flow's rate."""

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if sum(x is not None for x in (self.path, self.bundled, self.generator)) != 1:
            raise ValueError('give exactly one of "path", "bundled" or "generator"')
        if self.params and self.generator is None:
            raise ValueError('"params" needs "generator"')
        return self


class PolicySpec(_Model):
    """How the signals are controlled: as the scenario says, or by a registered controller."""

    kind: Literal["scenario", "controller"] = "scenario"
    controller: str | None = None
    """Registered controller name (``kind="controller"``)."""
    params: dict[str, Any] = Field(default_factory=dict)
    """Its parameters."""
    intersections: list[str] | None = None
    """Intersections it controls (default: every signalised one)."""

    @model_validator(mode="after")
    def _controller(self) -> Self:
        if (self.kind == "controller") != (self.controller is not None):
            raise ValueError('kind "controller" needs "controller", and only it may give one')
        return self

    def controllers(self) -> dict[str, Any] | None:
        """The ``Simulation(controllers=...)`` argument, or None."""
        if self.controller is None:
            return None
        ref = {"type": self.controller, "params": self.params}
        return dict.fromkeys(self.intersections or ["*"], ref)


class VariantSpec(_Model):
    """One arm of the experiment."""

    name: str = Field(pattern=SLUG)
    config: dict[str, Any] = Field(default_factory=dict)
    """Simulation config overrides (over the experiment's ``base_config``)."""
    scenario: ScenarioRef | None = None
    """Its own scenario (default: the experiment's)."""
    policy: PolicySpec = Field(default_factory=PolicySpec)
    demand_scale: float | None = Field(default=None, gt=0)
    """Overrides the scenario's ``demand_scale``."""


class SeedRange(_Model):
    start: int = Field(default=0, ge=0)
    count: int = Field(ge=1, le=1000)


class ExperimentSpec(_Model):
    """An experiment file (plan Q.1)."""

    format: Literal["urbanflow.experiment"] = "urbanflow.experiment"
    version: Literal["1.0"] = "1.0"
    name: str = Field(pattern=SLUG)
    description: str = ""
    scenario: ScenarioRef
    base_config: dict[str, Any] = Field(default_factory=dict)
    """Simulation config overrides shared by every variant."""
    variants: list[VariantSpec] = Field(min_length=1, max_length=64)
    baseline: str | None = None
    """Variant compared against (default: the first)."""
    seeds: list[int] | SeedRange = Field(default_factory=lambda: SeedRange(count=3))
    record_replay: bool | Literal["first_seed"] = False
    """Record a replay of every run, or of the first seed only."""
    export_format: Literal["csv", "parquet"] = "csv"
    """Format of the per-run metric tables."""
    max_workers: int | None = Field(default=None, ge=1, le=61)
    run_timeout_s: float | None = Field(default=None, gt=0)
    """A run still going after this wall time fails."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        if self.baseline is not None and self.baseline not in names:
            raise ValueError(f'baseline "{self.baseline}" is not a variant')
        seeds = self.seed_list()
        if len(set(seeds)) != len(seeds) or not seeds:
            raise ValueError("seeds must be distinct")
        return self

    def seed_list(self) -> list[int]:
        if isinstance(self.seeds, SeedRange):
            return list(range(self.seeds.start, self.seeds.start + self.seeds.count))
        return list(self.seeds)

    @property
    def baseline_name(self) -> str:
        return self.baseline or self.variants[0].name


class RunSpec(_Model):
    """One (variant, seed) run, everything a worker process needs (JSON-serialisable)."""

    run_id: str
    experiment_id: str
    variant: str
    seed: int
    scenario_json: str
    """The exact scenario (demand scale applied)."""
    scenario_hash: str
    config: dict[str, Any]
    """Config overrides (base, then variant, then the seed); the scenario supplies the rest."""
    policy: PolicySpec
    record_replay: bool
    export_format: Literal["csv", "parquet"]
    output_dir: str
    timeout_s: float | None
    content_key: str


def load_experiment(path: str | Path) -> ExperimentSpec:
    """Read and validate an experiment file; ``ConfigError`` with JSON paths otherwise."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"experiment file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{p}: invalid JSON: {exc}") from None
    return parse_experiment(data, source=str(p))


def parse_experiment(data: Any, *, source: str = "experiment") -> ExperimentSpec:
    try:
        return ExperimentSpec.model_validate(data)
    except ValidationError as exc:
        issues = tuple(
            ValidationIssue(format_path(err["loc"]), err["msg"], "E009") for err in exc.errors()
        )
        raise ConfigError(f"{source}: invalid experiment", issues) from None


def _load_scenario(ref: ScenarioRef, workspace: Path) -> Any:
    from urbanflow.scenario.generators import generate
    from urbanflow.scenario.io import bundled
    from urbanflow.scenario.scenario import Scenario

    if ref.bundled is not None:
        scenario = Scenario.load(bundled(ref.bundled))
    elif ref.generator is not None:
        scenario = generate(ref.generator, ref.params)
    else:
        root = workspace.resolve()
        path = (root / str(ref.path)).resolve()
        if not path.is_relative_to(root):
            raise ConfigError(f'scenario path "{ref.path}" leaves the workspace')
        scenario = Scenario.load(path)
    return scenario


def content_key(
    scenario_hash: str, config: Mapping[str, Any], policy: PolicySpec, seed: int
) -> str:
    major_minor = ".".join(__version__.split(".")[:2])
    doc = {
        "scenario_hash": scenario_hash,
        "config": dict(config),
        "policy": policy.model_dump(mode="json"),
        "seed": seed,
        "urbanflow": major_minor,
    }
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_experiment_id() -> str:
    return secrets.token_hex(6)


def expand(
    spec: ExperimentSpec, workspace: str | Path, experiment_id: str, runs_dir: str | Path
) -> list[RunSpec]:
    """One :class:`RunSpec` per (seed, variant), seed-major. Config overrides are validated
    here (``ConfigError`` naming ``variants[i].config``), scenarios loaded once each."""
    ws = Path(workspace)
    cache: dict[str, tuple[str, str]] = {}
    runs: list[RunSpec] = []
    resolved: list[tuple[VariantSpec, str, str, dict[str, Any]]] = []
    for i, variant in enumerate(spec.variants):
        ref = variant.scenario or spec.scenario
        if variant.demand_scale is not None:
            ref = ref.model_copy(update={"demand_scale": variant.demand_scale})
        key = ref.model_dump_json()
        if key not in cache:
            scenario = _load_scenario(ref, ws)
            if ref.demand_scale != 1.0:
                scenario = scenario.scale_demand(ref.demand_scale)
            cache[key] = (scenario.to_json(), scenario.content_hash)
        text, digest = cache[key]
        config = {**spec.base_config, **variant.config}
        try:
            SimulationConfig.model_validate(config)
        except ValidationError as exc:
            err = exc.errors()[0]
            where = format_path(("variants", i, "config", *err["loc"]))
            raise ConfigError(f"{where}: {err['msg']}") from None
        resolved.append((variant, text, digest, config))
    seeds = spec.seed_list()
    for k, seed in enumerate(seeds):
        for variant, text, digest, config in resolved:
            run_config = {**config, "seed": seed}
            run_id = f"{experiment_id}-{variant.name}-s{seed}"
            record = spec.record_replay is True or (spec.record_replay == "first_seed" and k == 0)
            runs.append(
                RunSpec(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    variant=variant.name,
                    seed=seed,
                    scenario_json=text,
                    scenario_hash=digest,
                    config=run_config,
                    policy=variant.policy,
                    record_replay=record,
                    export_format=spec.export_format,
                    output_dir=str(Path(runs_dir) / run_id),
                    timeout_s=spec.run_timeout_s,
                    content_key=content_key(digest, run_config, variant.policy, seed),
                )
            )
    return runs


def safe_name(name: str) -> str:
    """A file-system-safe slug of ``name``."""
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-") or "experiment"
