"""Every example script with a ``main()`` runs in quick mode from the repository root (AE.2).

Scripts are discovered under ``examples/``, so new example directories are picked up
without touching this file.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _has_main(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)


SCRIPTS = sorted(p for p in EXAMPLES.rglob("*.py") if _has_main(p))


def test_examples_are_discovered() -> None:
    assert EXAMPLES / "single_intersection" / "run.py" in SCRIPTS
    assert EXAMPLES / "single_intersection" / "inspect_vehicles.py" in SCRIPTS
    for directory in {p.parent for p in SCRIPTS}:
        assert (directory / "README.md").is_file(), directory


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.relative_to(EXAMPLES).as_posix())
def test_example_runs(
    script: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("URBANFLOW_EXAMPLE_QUICK", "1")
    monkeypatch.chdir(ROOT)
    name = "example_" + "_".join(script.relative_to(EXAMPLES).with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)  # importing runs nothing: main() is guarded
    assert not capsys.readouterr().out
    assert module.main() in (None, 0)  # the exit code of `python script.py`
    assert capsys.readouterr().out.strip()
