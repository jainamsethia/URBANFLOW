"""Scenario files: size-capped reading, strict JSON parsing, migration, atomic writes (E.7 §1.7).

These are the load-stage (L) checks of the validation pipeline (E.8): every failure raises
:class:`~urbanflow.core.errors.ScenarioValidationError` with one E000/E010-E013 issue.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn

from urbanflow.core.constants import MAX_JSON_DEPTH, MAX_SCENARIO_BYTES
from urbanflow.core.errors import NotFoundError, ScenarioValidationError, suggest
from urbanflow.scenario.schema import FORMAT

if TYPE_CHECKING:
    from urbanflow.core.errors import ValidationIssue

__all__ = [
    "CURRENT_VERSION",
    "MIGRATIONS",
    "bundled",
    "bundled_names",
    "migrate",
    "parse_json",
    "read_text",
    "version_issue",
    "write_json",
]

CURRENT_VERSION: Final = (1, 0)
MIGRATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
"""from-version "MAJOR.MINOR" -> function returning the document at its next version."""

_log = logging.getLogger(__name__)
_MB: Final = 1024 * 1024
_BOM: Final = chr(0xFEFF)  # a str input may still carry the UTF-8 byte-order mark
_NON_FINITE = re.compile(r'"(?:[^"\\]|\\.)*"|(-?Infinity|NaN)')


def _issue(code: str, path: str, *, variant: int = 0, **fmt: Any) -> ValidationIssue:
    from urbanflow.scenario.validate import issue  # validate imports this module

    return issue(code, path, variant=variant, **fmt)


def _fail(code: str, path: str, *, variant: int = 0, **fmt: Any) -> NoReturn:
    raise ScenarioValidationError([_issue(code, path, variant=variant, **fmt)])


def _version_str(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def _check_size(size: int) -> None:
    if size > MAX_SCENARIO_BYTES:
        _fail("E013", "$", size=size / _MB, limit=MAX_SCENARIO_BYTES // _MB)


def read_text(path: str | os.PathLike[str]) -> str:
    """Read a scenario file (UTF-8, optional BOM) after checking the 64 MiB size cap.

    Raises :class:`NotFoundError` (with a did-you-mean hint from sibling ``.json`` files)
    when the file does not exist.
    """
    file = Path(path)
    try:
        _check_size(file.stat().st_size)
        raw = file.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        siblings = [str(p) for p in file.parent.glob("*.json")] if file.parent.is_dir() else []
        hint = suggest(str(file), siblings).replace('"', "")
        raise NotFoundError(f"scenario file not found: {file}{hint}") from None
    except IsADirectoryError:
        raise NotFoundError(f"scenario path is a directory, not a file: {file}") from None
    return _decode(raw)


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        _fail("E000", "$", variant=1, detail=f"{exc.reason} at byte {exc.start}")


class _NonFinite(ValueError):
    """Raised by the JSON parser for NaN, Infinity and -Infinity."""


def _reject_constant(_token: str) -> NoReturn:
    raise _NonFinite("non-finite constant")


def parse_json(text: str | bytes) -> dict[str, Any]:
    """Parse a scenario document; NaN/Infinity and non-object roots are rejected.

    Every failure is an E000/E010/E013 :class:`ScenarioValidationError`, including parser
    limits such as integer literals over Python's digit limit or too deep nesting.
    """
    if isinstance(text, bytes):
        _check_size(len(text))
        text = _decode(text)
    else:
        _check_size(len(text.encode("utf-8")))
    text = text.removeprefix(_BOM)
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        _fail("E000", "$", detail=exc.msg, line=exc.lineno, col=exc.colno)
    except _NonFinite:
        match = next((m for m in _NON_FINITE.finditer(text) if m.group(1)), None)
        line = text.count("\n", 0, match.start()) + 1 if match else 1
        _fail("E000", "$", variant=2, line=line)
    except (ValueError, RecursionError) as exc:  # int digit limit, nesting depth
        _fail("E000", "$", variant=3, detail=str(exc))
    if _too_deep(data):  # pydantic cannot serialise (hash) deeper documents
        _fail("E000", "$", variant=3, detail=f"nesting deeper than {MAX_JSON_DEPTH} levels")
    if not isinstance(data, dict):
        _fail("E010", "format", got=f"a JSON {type(data).__name__}")
    return data


def _too_deep(value: Any) -> bool:
    """True if arrays/objects nest deeper than MAX_JSON_DEPTH (level by level, no recursion)."""
    level = [value]
    for _ in range(MAX_JSON_DEPTH):
        level = [
            child
            for node in level
            if isinstance(node, dict | list)
            for child in (node.values() if isinstance(node, dict) else node)
        ]
        if not level:
            return False
    return any(isinstance(node, dict | list) for node in level)


def _parse_version(value: object) -> tuple[int, int] | None:
    if isinstance(value, str) and re.fullmatch(r"\d+\.\d+", value):
        major, minor = value.split(".")
        return int(major), int(minor)
    return None


def version_issue(raw: object) -> ValidationIssue | None:
    """E011/E012 for a version this release cannot read, else None.

    The current version and older minor versions of the current major are readable (minor
    bumps only add optional fields); a newer minor gives E011, anything else E012.
    """
    version = _parse_version(raw)
    if version is not None and version[0] == CURRENT_VERSION[0]:
        if version[1] <= CURRENT_VERSION[1]:
            return None
        return _issue("E011", "version", v=json.dumps(raw), cur=_version_str(CURRENT_VERSION))
    return _issue("E012", "version", v=json.dumps(raw), cur=_version_str(CURRENT_VERSION))


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Check ``format``/``version`` and upgrade the document to :data:`CURRENT_VERSION`.

    Migrations chain through :data:`MIGRATIONS` (logged at INFO). A newer minor version
    raises E011 (upgrade urbanflow); a version without a migration path raises E012.
    Older minor versions of the current major are compatible (minor bumps only add
    optional fields). The input is not modified.
    """
    fmt = data.get("format")
    if fmt != FORMAT:
        _fail("E010", "format", got="nothing" if fmt is None else json.dumps(fmt))
    doc = dict(data)
    seen: set[str] = set()
    while True:
        raw = doc.get("version")
        if _parse_version(raw) == CURRENT_VERSION:
            return doc
        if isinstance(raw, str) and raw in MIGRATIONS and raw not in seen:
            seen.add(raw)
            doc = dict(MIGRATIONS[raw](doc))
            new = doc.get("version")
            _log.info(
                "migrated scenario from version %s to %s", raw, new, extra={"from": raw, "to": new}
            )
            continue
        found = version_issue(raw)
        if found is not None:
            raise ScenarioValidationError([found])
        doc["version"] = _version_str(CURRENT_VERSION)
        return doc


def write_json(path: str | os.PathLike[str], obj: Any, *, indent: int | None = 2) -> Path:
    """Write ``obj`` as UTF-8 JSON atomically (temp file in the same directory + replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def _bundled_dir() -> Path:
    return Path(str(resources.files("urbanflow.scenario") / "bundled"))


def bundled_names() -> list[str]:
    """Names of the bundled scenarios shipped with the package."""
    return sorted(p.stem for p in _bundled_dir().glob("*.json"))


def bundled(name: str) -> Path:
    """Path of a bundled (read-only, package data) scenario, e.g. ``bundled("grid_3x3")``.

    Only the listed names resolve, so a name can never address a path outside the package.
    """
    names = bundled_names()
    if name not in names:
        raise NotFoundError(
            f'unknown bundled scenario "{name}"{suggest(name, names)} '
            f"(available: {', '.join(names) or 'none'})"
        )
    return _bundled_dir() / f"{name}.json"
