"""Fixtures shared by the scenario tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

DEMO_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "scenarios" / "demo.json"
_DEMO: dict[str, Any] = json.loads(DEMO_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def demo_path() -> Path:
    """The E.7 §1.8 example scenario file (valid, 0 issues)."""
    return DEMO_PATH


@pytest.fixture
def demo_data() -> dict[str, Any]:
    """A fresh, mutable copy of the E.7 §1.8 example."""
    return copy.deepcopy(_DEMO)
