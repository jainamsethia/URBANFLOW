"""Exception hierarchy, validation issues and friendly error formatting.

Every error a user can trigger is an :class:`UrbanFlowError` with a stable ``exit_code``
(used by the CLI). Validation problems carry :class:`ValidationIssue` records whose
``path`` is a JSON-style location such as ``network.intersections[3].signal.phases[2]``.
"""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType
from typing import Any

__all__ = [
    "CommandError",
    "ConfigError",
    "InvariantViolation",
    "MissingDependencyError",
    "NotFoundError",
    "ReplayFormatError",
    "ScenarioValidationError",
    "Severity",
    "SimulationError",
    "UrbanFlowError",
    "ValidationIssue",
    "format_issues",
    "format_path",
    "require",
    "suggest",
]


class Severity(StrEnum):
    """Severity of a :class:`ValidationIssue`."""

    error = "error"
    warning = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem found while validating a scenario, config or request."""

    path: str
    message: str
    code: str
    severity: Severity = Severity.error

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": str(self.severity),
        }


class UrbanFlowError(Exception):
    """Base class of every error UrbanFlow raises on purpose."""

    exit_code: int = 1


class ScenarioValidationError(UrbanFlowError):
    """A scenario (or imported file) has one or more validation errors.

    ``str(err)`` renders the friendly multi-line report; ``err.to_dict()`` is the
    machine-readable form used by the server (HTTP 422) and ``--format json``.
    """

    exit_code = 3

    def __init__(self, issues: Iterable[ValidationIssue], source: str | None = None) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        self.source = source
        super().__init__(self._render())

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.error)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.warning)

    def to_dict(self) -> dict[str, Any]:
        return {
            "errors": [i.to_dict() for i in self.errors],
            "warnings": [i.to_dict() for i in self.warnings],
        }

    def _render(self) -> str:
        shown = self.errors or self.issues  # warnings only appear when promoted (--strict)
        return format_issues(shown)

    def __str__(self) -> str:
        return self._render()


class ConfigError(UrbanFlowError):
    """Invalid configuration, settings or CLI value."""

    exit_code = 4

    def __init__(self, message: str, issues: Iterable[ValidationIssue] = ()) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        text = message if not self.issues else f"{message}\n{format_issues(self.issues, header='')}"
        super().__init__(text.rstrip())


class NotFoundError(UrbanFlowError, LookupError):
    """An unknown file, entity id, run id or registry name."""

    exit_code = 5


class SimulationError(UrbanFlowError):
    """Lifecycle misuse, restore mismatch or corrupted simulation state."""

    exit_code = 6


class CommandError(SimulationError, ValueError):
    """An invalid control command (bad argument, impossible action)."""

    exit_code = 6


class InvariantViolation(SimulationError):
    """A debug-mode invariant check failed (see plan F.7)."""

    exit_code = 7

    def __init__(
        self,
        rule: str,
        step: int,
        uids: Sequence[int] = (),
        details: str = "",
    ) -> None:
        self.rule = rule
        self.step = step
        self.uids = tuple(int(u) for u in uids)
        self.details = details
        shown = ", ".join(str(u) for u in self.uids[:10])
        more = "" if len(self.uids) <= 10 else f" (+{len(self.uids) - 10} more)"
        msg = f"invariant {rule} violated at step {step}"
        if shown:
            msg += f" by vehicles uid {shown}{more}"
        if details:
            msg += f": {details}"
        super().__init__(msg)


class ReplayFormatError(UrbanFlowError):
    """A replay file is malformed, unsafe or written by an unsupported version."""

    exit_code = 8


class MissingDependencyError(UrbanFlowError, ImportError):
    """An optional dependency is required for this feature but is not installed."""

    exit_code = 9

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"this feature needs the optional package {package!r}; "
            f'install it with: pip install "urbanflow[{extra}]"  (or: uv sync --extra {extra})'
        )


_TYPE_SEGMENTS = frozenset({"int", "float", "str", "bool", "list", "dict", "tuple", "none"})


def format_path(loc: Sequence[str | int]) -> str:
    """Render a location tuple as a JSON-style path.

    >>> format_path(("network", "intersections", 3, "signal", "phases", 2))
    'network.intersections[3].signal.phases[2]'
    >>> format_path(("demand", "flows", 0, "type_mix", "city bus"))
    'demand.flows[0].type_mix["city bus"]'
    >>> format_path(())
    '$'

    Pydantic union-branch segments (``function-after[...]``, ``int``, ``str``...) are
    dropped so users see the path of *their* document.
    """
    out = ""
    for seg in loc:
        if isinstance(seg, int):
            out += f"[{seg}]"
            continue
        if "[" in seg or "(" in seg or seg.lower() in _TYPE_SEGMENTS:
            continue
        if seg.isidentifier():  # keywords too: JSON paths use `.from` (E.8)
            out += f".{seg}" if out else seg
        else:
            escaped = seg.replace("\\", "\\\\").replace('"', '\\"')
            out += f'["{escaped}"]'
    return out or "$"


def format_issues(
    issues: Iterable[ValidationIssue], header: str = "Scenario validation failed:"
) -> str:
    """Render issues as the friendly report shown by the CLI.

    ``Scenario validation failed:`` followed by one ``  - path: message`` line per issue.
    """
    lines = [header] if header else []
    lines += [f"  - {i.path}: {i.message}" for i in issues]
    return "\n".join(lines)


def suggest(name: str, candidates: Iterable[str]) -> str:
    """Return ``' (did you mean "x"?)'`` for the closest candidate, or ``''``."""
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return f' (did you mean "{matches[0]}"?)' if matches else ""


def require(module: str, extra: str) -> ModuleType:
    """Import an optional module or raise :class:`MissingDependencyError` with an install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(module.split(".")[0], extra) from exc
