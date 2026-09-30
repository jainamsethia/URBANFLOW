"""Signalised junctions with hand-written phases, and a driver for runtime + controller."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pytest

from urbanflow.core.events import EventBuffer
from urbanflow.core.types import Stage
from urbanflow.network import CompiledNetwork, compile_network
from urbanflow.scenario import Scenario, ScenarioBuilder
from urbanflow.signals import (
    ControllerContext,
    ControllerSetup,
    LaneData,
    SignalController,
    SignalProgram,
    SignalRuntime,
    build_programs,
    create_controller,
)

ARMS = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}

# 1-lane 4-arm junction J (right-hand traffic): W_in heads east, so W_in->N_out is a left
# turn, W_in->S_out a right turn; E_in->S_out is the left turn opposing W_in->E_out.
EW = {"W_in->E_out": "G", "E_in->W_out": "G", "W_in->N_out": "g", "E_in->S_out": "g"}
NS = {"N_in->S_out": "G", "S_in->N_out": "G"}
EW_PLUS = {**EW, "W_in->N_out": "G", "W_in->S_out": "G"}  # nothing lost from EW; g -> G
EW_LEFT = {"W_in->E_out": "G", "E_in->W_out": "g", "W_in->N_out": "G"}  # E_in->S_out lost


def junction_scenario(
    phases: Sequence[Mapping[str, Any]] | None = None, *, arm: float = 200.0, **signal: Any
) -> Scenario:
    """4-arm junction "J" (1 lane) whose signal has ``phases`` (default EW 30 s, NS 30 s)
    and the other ``SignalSpec`` fields in ``signal``."""
    b = ScenarioBuilder("signal_junction")
    b.intersection("J", (0.0, 0.0), kind="signalized")
    for name, (dx, dy) in ARMS.items():
        b.boundary(name, (dx * arm, dy * arm))
        b.road(f"{name}_in", name, "J")
        b.road(f"{name}_out", "J", name)
    if phases is None:
        phases = [{"id": "EW", "green": EW, "duration": 30}, {"id": "NS", "green": NS}]
    data = b.to_spec().model_dump(mode="json", by_alias=True, exclude_none=True)
    data["network"]["intersections"][0]["signal"] = {"phases": list(phases), **signal}
    return Scenario.from_dict(data)


def program_of(
    phases: Sequence[Mapping[str, Any]] | None = None, **signal: Any
) -> tuple[CompiledNetwork, SignalProgram]:
    scenario = junction_scenario(phases, **signal)
    net = compile_network(scenario)
    (prog,) = build_programs(net, scenario.resolved.network)
    return net, prog


@pytest.fixture(scope="session")
def ew_ns() -> tuple[CompiledNetwork, SignalProgram]:
    """Phases EW (30 s) and NS (30 s); yellow 3, all-red 1, min green 5."""
    return program_of()


def runtime_for(net: CompiledNetwork, *programs: SignalProgram) -> SignalRuntime:
    return SignalRuntime(programs, n_intersections=net.n_intersections, n_movements=net.n_movements)


class Driver:
    """Steps a runtime with a controller, as engine sub-step 1 does (no vehicles)."""

    def __init__(
        self,
        runtime: SignalRuntime,
        controller: SignalController,
        program: SignalProgram,
        *,
        dt: float = 1.0,
        params: Any = None,
        initial: bool = True,
    ) -> None:
        self.runtime, self.controller, self.program, self.dt = runtime, controller, program, dt
        self.events = EventBuffer()
        self.step_count = 0
        j = program.intersection
        if params is None:
            params = type(controller).Params()
        setup = ControllerSetup(
            program=program,
            params=params,
            rng=np.random.default_rng(0),
            dt=dt,
            initial=initial,
            _place=lambda *a, **k: runtime.place(j, *a, **k),
        )
        controller.reset(setup)

    def step(self) -> None:
        rt, j, t = self.runtime, self.program.intersection, self.step_count * self.dt
        self.events.clear()
        if rt.stage[j] == Stage.green.code:
            ctx = ControllerContext(
                time=t,
                dt=self.dt,
                program=self.program,
                phase=int(rt.phase[j]),
                stage=Stage.green,
                green_elapsed=float(rt.green_elapsed[j]),
                stage_elapsed=float(rt.stage_elapsed[j]),
                lanes=cast(LaneData, None),  # fixed_time and external never read lanes
                rng=np.random.default_rng(0),
            )
            q = self.controller.decide(ctx)
            if q is not None:
                rt.request(j, q)
        self.step_count += 1
        rt.advance(self.dt, self.events, step=self.step_count, time=t)

    def state(self) -> tuple[Stage, int, int]:
        j = self.program.intersection
        rt = self.runtime
        return Stage.from_code(int(rt.stage[j])), int(rt.phase[j]), int(rt.target[j])


@pytest.fixture
def make_driver() -> Callable[..., Driver]:
    def make(net: CompiledNetwork, program: SignalProgram, ref: Any, **kwargs: Any) -> Driver:
        controller, params = create_controller(ref)
        return Driver(runtime_for(net, program), controller, program, params=params, **kwargs)

    return make
