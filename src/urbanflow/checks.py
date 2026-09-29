"""Deep validation: the registry-backed and compile checks of stage X (plan E.8, AA 5.1).

``Scenario.load`` checks a file on its own (stages L, S, P, D, Q). :func:`check` adds what
needs the component registries and the network compiler:

* **E904** unknown ``simulation.car_following`` or ``simulation.router`` (did-you-mean hint
  and the available names); ``lane_change_model`` joins when lane changing lands;
* **E903** ``vehicle_types[i].model_params`` the car-following model's ``Params`` rejects
  (unknown names, or bad values reported with the structural codes);
* the compile diagnostics: **E802**, **E905**, **E806** (errors) and **W302**, **W304**.

``Simulation.__init__`` and ``urbanflow validate`` (``--deep``, the default) both run it.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from urbanflow.core.config import SimulationConfig, resolve_config
from urbanflow.core.errors import ScenarioValidationError, Severity, ValidationIssue, suggest
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.network.compiler import compile_network
from urbanflow.routing import router_registry
from urbanflow.scenario.io import parse_json, read_text
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.validate import (
    ValidationReport,
    issue,
    issues_from_pydantic,
    validate_data,
)
from urbanflow.vehicles import car_following_registry

__all__ = ["check", "deep_check"]


def check(
    scenario: Scenario | str | os.PathLike[str],
    config: SimulationConfig | None = None,
    *,
    strict: bool = False,
) -> ValidationReport:
    """Validate ``scenario`` (a :class:`Scenario` or a file) as a run would use it.

    ``config`` is layered over the scenario's ``simulation`` block (set fields only), as
    ``Simulation(scenario, config)`` does. The report lists the file-level issues, then
    the deep ones; ``strict=True`` turns warnings into errors. A missing file raises
    ``NotFoundError``; an invalid ``config`` raises ``ConfigError``.
    """
    report, _ = deep_check(scenario, config)
    if strict:
        promoted = tuple(replace(i, severity=Severity.error) for i in report.issues)
        report = replace(report, issues=promoted)
    return report


def deep_check(
    scenario: Scenario | str | os.PathLike[str],
    config: SimulationConfig | None = None,
    *,
    check_router: bool = True,
) -> tuple[ValidationReport, CompiledNetwork | None]:
    """:func:`check` plus the compiled network (None if the scenario does not compile).

    ``Simulation`` reuses the network instead of compiling twice; ``check_router=False``
    skips E904 for the router when a router instance was passed in.
    """
    if not isinstance(scenario, Scenario):
        loaded = _load(Path(scenario))
        if isinstance(loaded, ValidationReport):
            return loaded, None
        scenario = loaded
    cfg = resolve_config(("scenario", scenario.spec.simulation), ("config", config))
    issues = [*scenario.issues, *_registry_issues(scenario, cfg, check_router)]
    net: CompiledNetwork | None = None
    try:
        net = compile_network(scenario)
    except ScenarioValidationError as exc:
        issues += exc.issues
    else:
        issues += net.report.warnings
    return ValidationReport(tuple(issues), scenario.spec, scenario.resolved), net


def _load(path: Path) -> Scenario | ValidationReport:
    """The validated scenario of a file, or the report of a file that does not validate."""
    try:
        report = validate_data(parse_json(read_text(path)))
    except ScenarioValidationError as exc:  # load stage: size, encoding, JSON syntax
        return ValidationReport(exc.issues)
    if not report.ok or report.spec is None or report.resolved is None:
        return report
    return Scenario(report.spec, report.resolved, path=path, issues=report.warnings)


def _unknown(kind: str, name: str, available: list[str], field: str) -> ValidationIssue:
    return issue(
        "E904",
        ("simulation", field),
        kind=kind,
        name=name,
        hint=suggest(name, available),
        list=", ".join(available) or "none",
    )


def _registry_issues(
    scenario: Scenario, cfg: SimulationConfig, check_router: bool
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if check_router and cfg.router not in router_registry:
        issues.append(_unknown("router", cfg.router, router_registry.names(), "router"))
    name = cfg.car_following
    if name not in car_following_registry:
        available = car_following_registry.names()
        issues.append(_unknown("car-following model", name, available, "car_following"))
        return issues
    params = car_following_registry.get(name).Params
    known = sorted(params.model_fields)
    for i, vt in enumerate(scenario.spec.vehicle_types):
        base = ("vehicle_types", i, "model_params")
        unknown = [p for p in vt.model_params if p not in known]
        issues += [
            issue("E903", (*base, p), p=p, m=name, list=", ".join(known) or "none") for p in unknown
        ]
        if unknown:
            continue
        try:
            params.model_validate(dict(vt.model_params))
        except ValidationError as exc:
            issues += issues_from_pydantic(exc, prefix=base, root=params)
    return issues
