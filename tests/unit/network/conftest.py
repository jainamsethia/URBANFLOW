"""Fixtures shared by the network tests: the E.7 §1.8 demo, compiled once per module."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.scenario import Scenario

DEMO_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "scenarios" / "demo.json"
_DEMO: dict[str, Any] = json.loads(DEMO_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def demo_data() -> dict[str, Any]:
    """A fresh, mutable copy of the demo scenario document."""
    return copy.deepcopy(_DEMO)


@pytest.fixture(scope="module")
def demo() -> Scenario:
    return Scenario.load(DEMO_PATH)


@pytest.fixture(scope="module")
def demo_net(demo: Scenario) -> CompiledNetwork:
    """The compiled demo: a signalised 4-arm junction J with 2-lane roads (radius 8.4 m)."""
    return compile_network(demo)
