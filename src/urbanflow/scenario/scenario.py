"""The :class:`Scenario` object: a validated, immutable scenario and its resolved form (AA 5.2)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from urbanflow.core.constants import SHORT_HASH_LENGTH
from urbanflow.core.errors import ValidationIssue
from urbanflow.core.logging import log_duration
from urbanflow.scenario.derive import resolve
from urbanflow.scenario.io import CURRENT_VERSION, parse_json, read_text, write_json
from urbanflow.scenario.schema import ScenarioSpec, scenario_json_schema
from urbanflow.scenario.validate import ValidationReport, validate_data, validate_spec

if TYPE_CHECKING:
    from urbanflow.scenario.builder import ScenarioBuilder

__all__ = ["Scenario"]

_log = logging.getLogger("urbanflow.scenario")  # AB 6.4 event loggers
_validate_log = logging.getLogger("urbanflow.scenario.validate")


class Scenario:
    """A validated scenario: the declared ``spec`` and the fully explicit ``resolved`` spec.

    Scenarios are immutable; :meth:`edit` returns a builder for changes. Warnings found while
    loading are kept in ``issues``. Construct with :meth:`load`, :meth:`from_dict`,
    :meth:`from_json`, :meth:`from_spec` or :meth:`generate`.
    """

    def __init__(
        self,
        spec: ScenarioSpec,
        resolved: ScenarioSpec | None = None,
        *,
        path: Path | None = None,
        issues: tuple[ValidationIssue, ...] = (),
    ) -> None:
        """Wrap ``spec`` without validating it (use the constructors for validation)."""
        self.spec = spec
        self.resolved = resolved if resolved is not None else resolve(spec)
        self.path = path
        self.issues = issues

    # ------------------------------------------------------------------ constructors
    @classmethod
    def _from_report(cls, report: ValidationReport, source: str, path: Path | None) -> Scenario:
        report.raise_for_errors(source=source)
        if report.spec is None or report.resolved is None:  # pragma: no cover - errors raised
            raise RuntimeError("validation returned no spec")
        for warning in report.warnings:
            _validate_log.warning(
                "%s: %s: %s",
                source,
                warning.path,
                warning.message,
                extra={"code": warning.code, "path": warning.path},
            )
        return cls(report.spec, report.resolved, path=path, issues=report.warnings)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Scenario:
        """Load and validate a scenario file.

        Raises ``ScenarioValidationError`` (all errors, with JSON paths) or
        ``NotFoundError`` (missing file, with a did-you-mean hint).
        """
        file = Path(path)
        with log_duration(_log, "scenario loaded", path=str(file)) as extra:
            data = parse_json(read_text(file))
            scenario = cls._from_report(validate_data(data), str(file), file)
            if _log.isEnabledFor(logging.INFO):  # the hash costs a canonical dump
                extra |= {  # "scenario" = the name (LogRecord reserves the key "name")
                    "scenario": scenario.name,
                    "hash": scenario.short_hash,
                    "counts": scenario.summary(),
                }
        return scenario

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str = "<dict>") -> Scenario:
        """Validate a scenario document given as a mapping."""
        return cls._from_report(validate_data(data), source, None)

    @classmethod
    def from_json(cls, text: str | bytes, *, source: str = "<string>") -> Scenario:
        """Validate a scenario document given as JSON text."""
        return cls._from_report(validate_data(parse_json(text)), source, None)

    @classmethod
    def from_spec(cls, spec: ScenarioSpec) -> Scenario:
        """Validate an already structurally valid spec."""
        return cls._from_report(validate_spec(spec), "<spec>", None)

    @classmethod
    def generate(cls, name: str, /, **params: Any) -> Scenario:
        """Run a registered generator, e.g. ``Scenario.generate("single_intersection")``."""
        from urbanflow.scenario.generators import generate

        return generate(name, params)

    @staticmethod
    def json_schema() -> dict[str, Any]:
        """JSON Schema of the scenario format (draft 2020-12)."""
        return scenario_json_schema()

    # ------------------------------------------------------------------ views
    @property
    def name(self) -> str:
        return self.spec.meta.name

    @cached_property
    def content_hash(self) -> str:
        """sha256 of the canonical JSON of the resolved spec, excluding ``meta``."""
        data = self.resolved.model_dump(mode="json", exclude={"meta"})
        text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @property
    def short_hash(self) -> str:
        """First 12 hex characters of :attr:`content_hash`."""
        return self.content_hash[:SHORT_HASH_LENGTH]

    def to_dict(self, *, resolved: bool = False) -> dict[str, Any]:
        """JSON-ready dict; the declared form keeps only the fields that were set.

        ``version`` is always the current format version (E.7 §1.7).
        """
        if resolved:
            data = self.resolved.model_dump(mode="json")
        else:
            data = self.spec.model_dump(mode="json", exclude_unset=True)
        data["version"] = f"{CURRENT_VERSION[0]}.{CURRENT_VERSION[1]}"
        return data

    def to_json(self, *, resolved: bool = False, indent: int | None = 2) -> str:
        """JSON text of :meth:`to_dict`."""
        return json.dumps(self.to_dict(resolved=resolved), indent=indent, ensure_ascii=False)

    def save(
        self, path: str | os.PathLike[str], *, resolved: bool = False, indent: int = 2
    ) -> Path:
        """Write the scenario atomically; ``resolved=True`` writes the fully explicit spec."""
        return write_json(path, self.to_dict(resolved=resolved), indent=indent)

    def edit(self) -> ScenarioBuilder:
        """A :class:`ScenarioBuilder` initialised from this scenario."""
        from urbanflow.scenario.builder import ScenarioBuilder

        return ScenarioBuilder.from_scenario(self)

    def summary(self) -> dict[str, int]:
        """Entity counts of the resolved scenario."""
        net, demand = self.resolved.network, self.resolved.demand
        signals = [ix.signal for ix in net.intersections if ix.signal is not None]
        return {
            "intersections": len(net.intersections),
            "roads": len(net.roads),
            "lanes": sum(len(r.lanes) for r in net.roads),
            "movements": sum(len(ix.movements or ()) for ix in net.intersections),
            "connections": sum(
                len(m.connections or ()) for ix in net.intersections for m in ix.movements or ()
            ),
            "signals": len(signals),
            "phases": sum(len(s.phases or ()) for s in signals),
            "vehicle_types": len(self.resolved.vehicle_types),
            "flows": len(demand.flows),
            "trips": len(demand.trips),
            "transit_lines": len(demand.transit),
        }

    def __repr__(self) -> str:
        return f"Scenario(name={self.name!r}, hash={self.short_hash})"
