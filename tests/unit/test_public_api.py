"""The public API surface: every exported name resolves, imports stay lazy."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import urbanflow

REPO = Path(__file__).resolve().parents[2]


def test_every_public_name_resolves() -> None:
    for name in urbanflow.__all__:
        assert getattr(urbanflow, name) is not None, name


def test_unknown_attribute_raises() -> None:
    try:
        urbanflow.no_such_thing  # noqa: B018
    except AttributeError as exc:
        assert "no_such_thing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError")


def test_version_matches_pyproject() -> None:
    meta = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert urbanflow.__version__ == meta["project"]["version"]


def _modules_after_import(statement: str) -> set[str]:
    code = f"import sys; {statement}; print(','.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return set(out.stdout.strip().split(","))


def test_package_import_is_light() -> None:
    mods = _modules_after_import("import urbanflow")
    assert "numpy" not in mods
    assert "pydantic" not in mods


def test_cli_import_does_not_load_numpy() -> None:
    mods = _modules_after_import("import urbanflow.cli.main")
    assert "numpy" not in mods
    assert "networkx" not in mods
