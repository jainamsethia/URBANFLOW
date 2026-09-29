"""Scenario generators: a registry of pure functions of their parameters (plan E.10).

Built-in generators register when this package is imported; plugins register through the
``urbanflow.plugins`` entry-point group, which the registry loads on the first miss.
Generators write ``meta.generator`` provenance and never set ``simulation.seed``.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from urbanflow._version import __version__
from urbanflow.core.errors import ScenarioValidationError
from urbanflow.core.registry import Registry
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import GeneratorInfo, ScenarioSpec
from urbanflow.scenario.validate import issues_from_pydantic

__all__ = [
    "BUNDLED_SCENARIOS",
    "GENERATORS",
    "GeneratorEntry",
    "generate",
    "list_generators",
    "register_generator",
]


@dataclass(frozen=True, slots=True)
class GeneratorEntry:
    """A registered generator: ``fn(params) -> Scenario`` and its pydantic params model."""

    fn: Callable[[Any], Scenario]
    params: type[BaseModel]
    description: str

    @property
    def source(self) -> str:
        """``builtin`` or the top-level package that registered the generator."""
        package = self.fn.__module__.split(".")[0]
        return "builtin" if package == "urbanflow" else package


GENERATORS: Final[Registry[GeneratorEntry]] = Registry("generator")

BUNDLED_SCENARIOS: Final[Mapping[str, tuple[str, Mapping[str, Any]]]] = MappingProxyType(
    {"single_intersection": ("single_intersection", MappingProxyType({}))}
)
"""Bundled scenario name -> (generator, params); see ``scripts/regen_bundled_scenarios.py``."""


def register_generator[F: Callable[..., Scenario]](
    name: str, *, params: type[BaseModel], description: str | None = None
) -> Callable[[F], F]:
    """Decorator registering ``fn(params) -> Scenario`` as generator ``name``.

    ``description`` defaults to the first line of the function's docstring.
    """

    def decorator(fn: F) -> F:
        text = description or (inspect.getdoc(fn) or name).splitlines()[0]
        GENERATORS.register(name, GeneratorEntry(fn, params, text))
        return fn

    return decorator


def generate(
    name: str, params: BaseModel | Mapping[str, Any] | None = None, /, **kw: Any
) -> Scenario:
    """Run generator ``name`` with ``params`` (a model or mapping) updated by ``kw``.

    Parameter errors raise ``ScenarioValidationError`` with paths such as ``params.rows``;
    an unknown name raises ``NotFoundError`` listing the available generators.
    """
    entry = GENERATORS.get(name)
    data = params.model_dump() if isinstance(params, BaseModel) else dict(params or {})
    try:
        validated = entry.params.model_validate(_with_defaults(entry.params, {**data, **kw}))
    except ValidationError as exc:
        issues = issues_from_pydantic(exc, prefix=("params",), root=entry.params)
        raise ScenarioValidationError(issues, f"generator {name}") from None
    scenario = entry.fn(validated)
    info = GeneratorInfo(
        name=name, params=validated.model_dump(mode="json"), urbanflow_version=__version__
    )

    def stamped(spec: ScenarioSpec) -> ScenarioSpec:
        return spec.model_copy(update={"meta": spec.meta.model_copy(update={"generator": info})})

    return Scenario(stamped(scenario.spec), stamped(scenario.resolved), issues=scenario.issues)


def _with_defaults(model: type[BaseModel], data: Mapping[str, Any]) -> dict[str, Any]:
    """Partial nested params (``turn_ratios.far=0.2``) update the field's default model."""
    out = dict(data)
    for name, info in model.model_fields.items():
        value = out.get(name)
        if isinstance(value, Mapping) and not info.is_required():
            default = info.get_default(call_default_factory=True)
            if isinstance(default, BaseModel):
                out[name] = {**default.model_dump(), **value}
    return out


def list_generators() -> list[tuple[str, GeneratorEntry]]:
    """Sorted ``(name, entry)`` pairs, including plugin generators."""
    return GENERATORS.items()


for _module in ("single_intersection",):  # built-ins register on import
    importlib.import_module(f"{__name__}.{_module}")
