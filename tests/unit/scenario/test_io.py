"""Load stage: reading, JSON parsing, migration, atomic writes, bundled files (plan E.7 §1.7)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from urbanflow.core.errors import NotFoundError, ScenarioValidationError
from urbanflow.scenario import io


def _issue(exc: pytest.ExceptionInfo[ScenarioValidationError]) -> tuple[str, str, str]:
    (found,) = exc.value.issues
    return found.code, found.path, found.message


def test_read_text_tolerates_a_bom(tmp_path: Path) -> None:
    file = tmp_path / "s.json"
    file.write_bytes(b'\xef\xbb\xbf{"a": 1}')
    assert io.parse_json(io.read_text(file)) == {"a": 1}
    assert io.parse_json('﻿{"a": 1}') == {"a": 1}
    assert io.parse_json(b'\xef\xbb\xbf{"a": 1}') == {"a": 1}


def test_missing_file_suggests_a_sibling(tmp_path: Path) -> None:
    (tmp_path / "grid.json").write_text("{}", encoding="utf-8")
    with pytest.raises(NotFoundError) as info:
        io.read_text(tmp_path / "gird.json")
    assert "scenario file not found" in str(info.value)
    assert f"(did you mean {tmp_path / 'grid.json'}?)" in str(info.value)
    with pytest.raises(NotFoundError):
        io.read_text(tmp_path / "nowhere" / "x.json")


def test_invalid_utf8(tmp_path: Path) -> None:
    file = tmp_path / "s.json"
    file.write_bytes(b'{"a": "\xff"}')
    with pytest.raises(ScenarioValidationError) as info:
        io.read_text(file)
    code, path, message = _issue(info)
    assert (code, path) == ("E000", "$")
    assert message.startswith("file is not valid UTF-8:")


def test_invalid_json_reports_line_and_column() -> None:
    with pytest.raises(ScenarioValidationError) as info:
        io.parse_json('{\n  "a": 1\n  "b": 2}')
    code, _, message = _issue(info)
    assert code == "E000"
    assert message.startswith("invalid JSON: ") and message.endswith("(line 3, column 3)")


@pytest.mark.parametrize(("text", "line"), [('{"a": NaN}', 1), ('{\n"a": 1,\n"b": -Infinity}', 3)])
def test_non_finite_constants_are_rejected(text: str, line: int) -> None:
    with pytest.raises(ScenarioValidationError) as info:
        io.parse_json(text)
    assert _issue(info) == ("E000", "$", f"NaN/Infinity are not allowed (line {line})")


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('{"x": ' + "1" * 5000 + "}", "integer string conversion"),  # Python's digit limit
        # nesting depth: RecursionError up to Python 3.13, our depth cap on 3.14+
        ('{"x": ' + "[" * 100_000 + "]" * 100_000 + "}", ("recursion", "nesting deeper than")),
        ('{"x": ' + "[" * 64 + "]" * 64 + "}", "nesting deeper than 64 levels"),  # hashable
    ],
    ids=["digit-limit", "nesting", "depth-cap"],
)
def test_parser_limits_are_load_errors(text: str, fragment: str | tuple[str, ...]) -> None:
    """Every json.loads failure is an E000 issue, never an internal error."""
    with pytest.raises(ScenarioValidationError) as info:
        io.parse_json(text)
    code, path, message = _issue(info)
    assert (code, path) == ("E000", "$")
    fragments = (fragment,) if isinstance(fragment, str) else fragment
    assert message.startswith("invalid JSON: ") and any(f in message for f in fragments)


def test_nesting_up_to_the_cap_is_fine() -> None:
    assert io.parse_json('{"x": ' + "[" * 63 + "]" * 63 + "}")["x"]


def test_nan_inside_a_string_is_fine() -> None:
    assert io.parse_json('{"a": "NaN", "b": "x\\" Infinity"}') == {"a": "NaN", "b": 'x" Infinity'}


def test_root_must_be_an_object() -> None:
    with pytest.raises(ScenarioValidationError) as info:
        io.parse_json("[1, 2]")
    assert _issue(info)[:2] == ("E010", "format")


def test_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io, "MAX_SCENARIO_BYTES", 3 * 1024 * 1024)
    file = tmp_path / "big.json"
    file.write_bytes(b" " * (5 * 1024 * 1024))
    with pytest.raises(ScenarioValidationError) as info:
        io.read_text(file)
    assert _issue(info) == ("E013", "$", "file is 5.0 MB; the limit is 3 MB")
    with pytest.raises(ScenarioValidationError):
        io.parse_json(" " * (4 * 1024 * 1024))


@pytest.mark.parametrize(
    ("doc", "code", "path"),
    [
        ({"format": "cityflow", "version": "1.0"}, "E010", "format"),
        ({"version": "1.0"}, "E010", "format"),
        ({"format": "urbanflow.scenario", "version": "1.3"}, "E011", "version"),
        ({"format": "urbanflow.scenario", "version": "2.0"}, "E012", "version"),
        ({"format": "urbanflow.scenario", "version": "0.9"}, "E012", "version"),
        ({"format": "urbanflow.scenario", "version": "one"}, "E012", "version"),
        ({"format": "urbanflow.scenario"}, "E012", "version"),
    ],
)
def test_format_and_version_checks(doc: dict[str, Any], code: str, path: str) -> None:
    with pytest.raises(ScenarioValidationError) as info:
        io.migrate(doc)
    assert _issue(info)[:2] == (code, path)


def test_messages_name_the_versions() -> None:
    with pytest.raises(ScenarioValidationError) as info:
        io.migrate({"format": "urbanflow.scenario", "version": "1.3"})
    assert (
        _issue(info)[2] == 'scenario version "1.3" is newer than supported 1.0; upgrade urbanflow'
    )
    with pytest.raises(ScenarioValidationError) as info:
        io.migrate({"format": "x"})
    assert '(got "x")' in _issue(info)[2]


def test_current_version_passes_through_unchanged() -> None:
    doc = {"format": "urbanflow.scenario", "version": "1.0", "x": [1]}
    out = io.migrate(doc)
    assert out == doc and out is not doc


def test_older_minor_versions_are_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io, "CURRENT_VERSION", (1, 2))
    out = io.migrate({"format": "urbanflow.scenario", "version": "1.0"})
    assert out["version"] == "1.2"


def test_migrations_chain(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def from_08(d: dict[str, Any]) -> dict[str, Any]:
        return {**d, "version": "0.9", "steps": [*d.get("steps", []), "0.8"]}

    def from_09(d: dict[str, Any]) -> dict[str, Any]:
        return {**d, "version": "1.0", "steps": [*d["steps"], "0.9"]}

    monkeypatch.setattr(io, "MIGRATIONS", {"0.8": from_08, "0.9": from_09})
    doc = {"format": "urbanflow.scenario", "version": "0.8"}
    with caplog.at_level(logging.INFO, logger="urbanflow.scenario.io"):
        out = io.migrate(doc)
    assert out["version"] == "1.0" and out["steps"] == ["0.8", "0.9"]
    assert doc == {"format": "urbanflow.scenario", "version": "0.8"}
    steps = [r for r in caplog.records if "migrated scenario" in r.message]
    assert [(r.__dict__["from"], r.__dict__["to"]) for r in steps] == [
        ("0.8", "0.9"),
        ("0.9", "1.0"),
    ]


def test_migration_loops_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io, "MIGRATIONS", {"0.9": lambda d: dict(d)})
    with pytest.raises(ScenarioValidationError) as info:
        io.migrate({"format": "urbanflow.scenario", "version": "0.9"})
    assert _issue(info)[0] == "E012"


def test_write_json_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "s.json"
    assert io.write_json(target, {"name": "ü", "n": [1, 2]}) == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"name": "ü", "n": [1, 2]}
    assert target.read_bytes().endswith(b"\n") and b"\r\n" not in target.read_bytes()
    with pytest.raises(TypeError):
        io.write_json(target, {"bad": object()})
    assert json.loads(target.read_text(encoding="utf-8"))["n"] == [1, 2]  # untouched
    assert [p.name for p in target.parent.iterdir()] == ["s.json"]  # no temp files left


def test_bundled_scenarios() -> None:
    names = io.bundled_names()
    assert "single_intersection" in names
    path = io.bundled("single_intersection")
    assert path.is_file() and path.suffix == ".json"
    with pytest.raises(NotFoundError) as info:
        io.bundled("single_intersectoin")
    assert '(did you mean "single_intersection"?)' in str(info.value)
    assert "available: corridor, grid_3x3, grid_4x4, single_intersection" in str(info.value)


@pytest.mark.parametrize(
    "name", ["../scenario", "../../scenario/bundled/single_intersection", "/etc/passwd", ""]
)
def test_bundled_names_cannot_escape_the_package(name: str) -> None:
    with pytest.raises(NotFoundError, match="unknown bundled scenario"):
        io.bundled(name)


def test_version_issue() -> None:
    assert io.version_issue("1.0") is None
    found = io.version_issue("1.4")
    assert found is not None and (found.code, found.path) == ("E011", "version")
    assert [io.version_issue(v).code for v in ("2.0", "x", None)] == ["E012"] * 3  # type: ignore[union-attr]
