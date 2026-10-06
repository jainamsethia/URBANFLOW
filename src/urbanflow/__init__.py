"""UrbanFlow: Python-first microscopic city traffic simulation and research workbench.

Public names are imported lazily (PEP 562) so that ``import urbanflow`` and the CLI start
fast and do not pull in numpy or networkx until a feature needs them.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from urbanflow._version import __version__

# Library etiquette: never emit log records unless the application configures logging.
logging.getLogger("urbanflow").addHandler(logging.NullHandler())

# name -> (module, attribute). Grows phase by phase; every entry must resolve (tested).
_LAZY: dict[str, tuple[str, str]] = {
    "UrbanFlowError": ("urbanflow.core.errors", "UrbanFlowError"),
    "ScenarioValidationError": ("urbanflow.core.errors", "ScenarioValidationError"),
    "ValidationIssue": ("urbanflow.core.errors", "ValidationIssue"),
    "ConfigError": ("urbanflow.core.errors", "ConfigError"),
    "NotFoundError": ("urbanflow.core.errors", "NotFoundError"),
    "SimulationError": ("urbanflow.core.errors", "SimulationError"),
    "CommandError": ("urbanflow.core.errors", "CommandError"),
    "InvariantViolation": ("urbanflow.core.errors", "InvariantViolation"),
    "ReplayFormatError": ("urbanflow.core.errors", "ReplayFormatError"),
    "MissingDependencyError": ("urbanflow.core.errors", "MissingDependencyError"),
    "SimulationConfig": ("urbanflow.core.config", "SimulationConfig"),
    "Simulation": ("urbanflow.simulation", "Simulation"),
    "SimulationResult": ("urbanflow.results", "SimulationResult"),
    "Snapshot": ("urbanflow.snapshot", "Snapshot"),
    "check": ("urbanflow.checks", "check"),
    "EventType": ("urbanflow.core.events", "EventType"),
    "VehicleStatus": ("urbanflow.core.types", "VehicleStatus"),
    "SignalState": ("urbanflow.core.types", "SignalState"),
    "TurnKind": ("urbanflow.core.types", "TurnKind"),
    "IntersectionKind": ("urbanflow.core.types", "IntersectionKind"),
    "Scenario": ("urbanflow.scenario.scenario", "Scenario"),
    "ScenarioBuilder": ("urbanflow.scenario.builder", "ScenarioBuilder"),
    "ValidationReport": ("urbanflow.scenario.validate", "ValidationReport"),
    "generate": ("urbanflow.scenario.generators", "generate"),
    "bundled": ("urbanflow.scenario.io", "bundled"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'urbanflow' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # pragma: no cover - static typing only
    from urbanflow.checks import (
        check as check,
    )
    from urbanflow.core.config import (
        SimulationConfig as SimulationConfig,
    )
    from urbanflow.core.errors import (
        CommandError as CommandError,
    )
    from urbanflow.core.errors import (
        ConfigError as ConfigError,
    )
    from urbanflow.core.errors import (
        InvariantViolation as InvariantViolation,
    )
    from urbanflow.core.errors import (
        MissingDependencyError as MissingDependencyError,
    )
    from urbanflow.core.errors import (
        NotFoundError as NotFoundError,
    )
    from urbanflow.core.errors import (
        ReplayFormatError as ReplayFormatError,
    )
    from urbanflow.core.errors import (
        ScenarioValidationError as ScenarioValidationError,
    )
    from urbanflow.core.errors import (
        SimulationError as SimulationError,
    )
    from urbanflow.core.errors import (
        UrbanFlowError as UrbanFlowError,
    )
    from urbanflow.core.errors import (
        ValidationIssue as ValidationIssue,
    )
    from urbanflow.core.events import (
        EventType as EventType,
    )
    from urbanflow.core.types import (
        IntersectionKind as IntersectionKind,
    )
    from urbanflow.core.types import (
        SignalState as SignalState,
    )
    from urbanflow.core.types import (
        TurnKind as TurnKind,
    )
    from urbanflow.core.types import (
        VehicleStatus as VehicleStatus,
    )
    from urbanflow.results import (
        SimulationResult as SimulationResult,
    )
    from urbanflow.scenario.builder import (
        ScenarioBuilder as ScenarioBuilder,
    )
    from urbanflow.scenario.generators import (
        generate as generate,
    )
    from urbanflow.scenario.io import (
        bundled as bundled,
    )
    from urbanflow.scenario.scenario import (
        Scenario as Scenario,
    )
    from urbanflow.scenario.validate import (
        ValidationReport as ValidationReport,
    )
    from urbanflow.simulation import (
        Simulation as Simulation,
    )
    from urbanflow.snapshot import (
        Snapshot as Snapshot,
    )
