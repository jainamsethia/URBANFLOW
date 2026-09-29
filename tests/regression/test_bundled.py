"""Bundled scenarios (plan B.2 #21): load cleanly and equal their regenerated content."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

import urbanflow
from urbanflow.scenario.generators import BUNDLED_SCENARIOS
from urbanflow.scenario.io import bundled_names

REPO = Path(__file__).resolve().parents[2]


def _regen_script() -> Any:
    path = REPO / "scripts" / "regen_bundled_scenarios.py"
    spec = importlib.util.spec_from_file_location("regen_bundled_scenarios", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_bundled_file_is_registered() -> None:
    assert sorted(BUNDLED_SCENARIOS) == bundled_names()


@pytest.mark.parametrize("name", sorted(BUNDLED_SCENARIOS))
def test_bundled_scenario(name: str) -> None:
    path = urbanflow.bundled(name)
    scenario = urbanflow.Scenario.load(path)
    assert scenario.issues == ()
    assert path.read_text(encoding="utf-8") == _regen_script().render(name)


def test_regen_check_passes() -> None:
    assert _regen_script().main(["--check"]) == 0
