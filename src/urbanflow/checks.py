"""Deep validation: the registry-backed and compile checks of stage X (plan E.8, AA 5.1).

``Scenario.load`` checks a file on its own (stages L, S, P, D, Q). :func:`check` adds what
needs the resolved run config, the component registries and the network compiler:

* **E506**, **E507** re-checked with the run's final ``dt`` and ``duration`` (Python or
  CLI overrides can break a flow that the scenario's own ``simulation`` block allows);
* **E904** unknown ``simulation.car_following`` or ``simulation.router`` (did-you-mean hint
  and the available names); ``lane_change_model`` joins when lane changing lands;
* **E903** ``vehicle_types[i].model_params`` the car-following model's ``Params`` rejects
  (unknown names, or bad values reported with the structural codes);
* **E901** unknown ``signal.controller.type`` and **E902** controller ``params`` its
  ``Params`` rejects (the converted pydantic message at the parameter's path); the same
  for ``Simulation(controllers=...)`` overrides given by name or ``{"type", "params"}``
  (paths ``controllers.<key>``);
* the compile diagnostics: **E802**, **E905**, **E806** (errors) and **W302**, **W304**;
* **E505** (second message): an integer ``depart_lane`` of a flow or trip with no
  connection to the route's next road (for origin/destination items, the router's first
  leg). Without lane changes such a vehicle could never leave its first lane.

``Simulation.__init__`` and ``urbanflow validate`` (``--deep``, the default) both run it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from urbanflow.core.config import SimulationConfig, resolve_config
from urbanflow.core.errors import (
    NotFoundError,
    ScenarioValidationError,
    Severity,
    ValidationIssue,
    suggest,
)
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.network.compiler import compile_network
from urbanflow.network.graph import build_road_graph
from urbanflow.routing import Router, router_registry, valid_mask
from urbanflow.scenario.io import parse_json, read_text
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import ControllerSpec, FlowSpec, TripSpec
from urbanflow.scenario.validate import (
    ValidationReport,
    demand_config_issues,
    issue,
    issues_from_pydantic,
    validate_data,
)
from urbanflow.signals import ControllerRef, controller_registry
from urbanflow.vehicles import car_following_registry
from urbanflow.vehicles.types import model_param_issues

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
    router: Router | None = None,
    controllers: Mapping[str, ControllerRef] | None = None,
) -> tuple[ValidationReport, CompiledNetwork | None]:
    """:func:`check` plus the compiled network (None if the scenario does not compile).

    ``Simulation`` reuses the network instead of compiling twice. ``router`` is the router
    instance a run uses instead of ``config.router`` (no E904 for the router name then);
    ``controllers`` are its ``controllers=`` overrides (E901/E902 for the ones given by name
    or mapping; instances and factories are checked when built).
    """
    if not isinstance(scenario, Scenario):
        loaded = _load(Path(scenario))
        if isinstance(loaded, ValidationReport):
            return loaded, None
        scenario = loaded
    cfg = resolve_config(("scenario", scenario.spec.simulation), ("config", config))
    issues = [
        *scenario.issues,
        *demand_config_issues(scenario.resolved.demand, cfg.dt, cfg.duration),
    ]
    issues += _registry_issues(scenario, cfg, check_router=router is None)
    issues += _controller_issues(scenario, controllers or {})
    net: CompiledNetwork | None = None
    try:
        net = compile_network(scenario)
    except ScenarioValidationError as exc:
        issues += exc.issues
    else:
        issues += net.report.warnings
        issues += _depart_lane_issues(scenario, net, cfg, router)
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
    scenario: Scenario, cfg: SimulationConfig, *, check_router: bool
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
    for i, vt in enumerate(scenario.spec.vehicle_types):
        base = ("vehicle_types", i, "model_params")
        issues += model_param_issues(vt.model_params, params, name, base)[0]
    return issues


def _controller_issues(
    scenario: Scenario, overrides: Mapping[str, ControllerRef]
) -> list[ValidationIssue]:
    """E901/E902 of the named overrides and of the resolved signals' controllers they do
    not replace (an override may rescue a scenario whose configured controller is unknown)."""
    issues: list[ValidationIssue] = []
    for i, ix in enumerate(scenario.resolved.network.intersections):
        if ix.signal is not None and "*" not in overrides and ix.id not in overrides:
            base = ("network", "intersections", i, "signal", "controller")
            issues += _spec_issues(ix.signal.controller, base, (*base, "type"))
    for key, ref in overrides.items():
        loc: tuple[str | int, ...] = ("controllers", key)
        if isinstance(ref, ControllerSpec):
            issues += _spec_issues(ref, loc, (*loc, "type"))
        elif isinstance(ref, str | Mapping):
            data: Mapping[str, Any] = {"type": ref} if isinstance(ref, str) else ref
            try:
                spec = ControllerSpec.model_validate(dict(data))
            except ValidationError as exc:
                issues += issues_from_pydantic(exc, prefix=loc, root=ControllerSpec)
                continue
            issues += _spec_issues(spec, loc, loc if isinstance(ref, str) else (*loc, "type"))
    return issues


def _spec_issues(
    spec: ControllerSpec, base: tuple[str | int, ...], type_path: tuple[str | int, ...]
) -> list[ValidationIssue]:
    """E901 if ``spec.type`` is not registered, else E902 for what its ``Params`` rejects."""
    if spec.type not in controller_registry:
        names = controller_registry.names()
        hint, listed = suggest(spec.type, names), ", ".join(names) or "none"
        return [issue("E901", type_path, t=spec.type, hint=hint, list=listed)]
    params = controller_registry.get(spec.type).Params
    try:
        params.model_validate(dict(spec.params))
    except ValidationError as exc:
        converted = issues_from_pydantic(exc, prefix=(*base, "params"), root=params)
        return [replace(x, code="E902") for x in converted]
    return []


def _depart_lane_issues(
    scenario: Scenario, net: CompiledNetwork, cfg: SimulationConfig, router: Router | None
) -> list[ValidationIssue]:
    """E505 (second message) for integer depart lanes that cannot reach the next road.

    Origin/destination items are checked on the first leg of ``router`` (else of
    ``cfg.router``, built on first use; skipped if that name is unknown, E904).
    """
    idx = net.road_index
    issues: list[ValidationIssue] = []
    ready = False
    demand = scenario.resolved.demand
    items: list[tuple[str, int, FlowSpec | TripSpec]] = [
        *(("flows", i, f) for i, f in enumerate(demand.flows)),
        *(("trips", i, t) for i, t in enumerate(demand.trips)),
    ]
    for kind, i, item in items:
        k = item.depart_lane
        if isinstance(k, str):
            continue
        if isinstance(item, FlowSpec) and item.routes:
            paths = [tuple(idx[r] for r in choice.roads) for choice in item.routes]
        elif item.route:
            paths = [tuple(idx[r] for r in item.route)]
        elif item.origin and item.destination:
            if not ready:
                if router is None and cfg.router in router_registry:
                    router = router_registry.get(cfg.router)()
                if router is not None:
                    router.reset(net, build_road_graph(net), cfg)
                ready = True
            if router is None:
                continue
            via = tuple(idx[v] for v in item.via)
            try:
                paths = [router.route(idx[item.origin], idx[item.destination], via)]
            except NotFoundError:  # E504 already covers the static graph
                continue
        else:
            continue
        for p in paths:
            if len(p) > 1 and not valid_mask(net, p[0], p[1]) >> k & 1:
                road, nxt = net.road_ids[p[0]], net.road_ids[p[1]]
                loc = ("demand", kind, i, "depart_lane")
                issues.append(issue("E505", loc, variant=1, k=k, road=road, next=nxt))
                break
    return issues
