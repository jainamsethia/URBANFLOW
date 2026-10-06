"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, settings

if TYPE_CHECKING:
    from urbanflow.engine import Engine
    from urbanflow.scenario import Scenario, ScenarioBuilder

# Property tests check correctness, not speed: no per-example deadline (slow CI runners and
# coverage tracing otherwise turn timing into failures).
settings.register_profile("urbanflow", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("urbanflow")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """On GitHub Actions, turn each failure into a public ``::error::`` annotation."""
    if not os.environ.get("GITHUB_ACTIONS") or not report.failed:
        return
    text = str(report.longrepr).strip().splitlines()
    tail = " | ".join(line.strip() for line in text[-4:] if line.strip())
    message = tail.replace("%", "%25").replace("\r", "").replace("\n", " ")[:900]
    print(f"\n::error title={report.nodeid} ({report.when})::{message}")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An empty workspace directory, isolated from the developer's environment."""
    ws = tmp_path / "ws"
    ws.mkdir()
    for var in (
        "URBANFLOW_WORKSPACE",
        "URBANFLOW_CONFIG",
        "URBANFLOW_CONFIG_FILE",  # pydantic-settings also reads the prefixed field name
        "URBANFLOW_DATABASE_URL",
        "URBANFLOW_HOST",
        "URBANFLOW_PORT",
        "URBANFLOW_API_TOKEN",
        "URBANFLOW_LOG_LEVEL",
        "URBANFLOW_LOG_FORMAT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(ws)
    yield ws


@pytest.fixture(autouse=True)
def _reset_urbanflow_logger() -> Iterator[None]:
    """Keep logging configuration from leaking between tests."""
    import logging

    logger = logging.getLogger("urbanflow")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def make_engine() -> Callable[..., Engine]:
    """``make_engine(scenario, controllers=None, **config)``: an Engine with
    ``debug_checks=True`` by default (``controllers``: overrides by intersection index).

    The config resolves the scenario's ``simulation`` block, then the overrides; the car
    following model and router come from their registries, as the facade does.
    """
    from urbanflow.core.config import resolve_config
    from urbanflow.core.rng import RngStreams
    from urbanflow.engine import Engine
    from urbanflow.network import compile_network
    from urbanflow.routing import router_registry
    from urbanflow.signals import build_programs
    from urbanflow.vehicles import VehicleTypes, car_following_registry

    def make(
        scenario: Scenario, *, controllers: Mapping[int, Any] | None = None, **overrides: Any
    ) -> Engine:
        config = resolve_config(
            ("scenario", scenario.resolved.simulation),
            ("test", {"debug_checks": True, **overrides}),
        )
        net = compile_network(scenario)
        model = car_following_registry.get(config.car_following)
        types = VehicleTypes.from_specs(net.vehicle_types, model.Params, model=config.car_following)
        return Engine(
            net,
            config,
            rng=RngStreams(config.seed),
            router=router_registry.get(config.router)(),
            car_following=model(),
            types=types,
            demand=scenario.resolved.demand,
            signals=build_programs(net, scenario.resolved.network),
            controllers=controllers,
        )

    return make


ARMS = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}


@pytest.fixture
def junction_builder() -> Callable[..., ScenarioBuilder]:
    """``junction_builder(kind="uncontrolled", arm=200.0, lanes=1, **simulation)``.

    A 4-arm junction "J" with boundary nodes E/N/W/S at ``arm`` m, roads ``{arm}_in`` and
    ``{arm}_out``, derived movements and no demand (add flows or trips before ``build``).
    """
    from urbanflow.scenario import ScenarioBuilder

    def make(
        kind: str = "uncontrolled", arm: float = 200.0, lanes: int = 1, **simulation: Any
    ) -> ScenarioBuilder:
        b = ScenarioBuilder("junction", **simulation)
        b.intersection("J", (0.0, 0.0), kind=kind)
        for name, (dx, dy) in ARMS.items():
            b.boundary(name, (dx * arm, dy * arm))
            b.road(f"{name}_in", name, "J", lanes=lanes)
            b.road(f"{name}_out", "J", name, lanes=lanes)
        return b

    return make
