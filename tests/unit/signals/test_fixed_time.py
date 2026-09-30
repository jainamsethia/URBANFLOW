"""Fixed-time control: realised cycle C_q and offsets, incl. positions in yellow and all-red."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from urbanflow.core.errors import ConfigError, SimulationError
from urbanflow.core.types import Stage
from urbanflow.signals import ControllerSetup, FixedTime, controller_registry, create_controller
from urbanflow.signals.controllers import FixedTimeParams, cycle_position, realised_cycle

from .conftest import EW, EW_PLUS, NS, Driver, program_of

GREEN, YELLOW, ALL_RED = Stage.green, Stage.yellow, Stage.all_red
PHASES = [{"green": EW, "duration": 30}, {"green": NS, "duration": 25}]


def _cycle(driver: Driver, steps: int) -> list[tuple[Stage, int, int]]:
    out = []
    for _ in range(steps):
        driver.step()
        out.append(driver.state())
    return out


def _rle(states: list[tuple[Stage, int, int]]) -> list[tuple[Stage, int, int]]:
    out: list[tuple[Stage, int, int]] = []
    for stage, phase, _ in states:
        if out and out[-1][:2] == (stage, phase):
            out[-1] = (stage, phase, out[-1][2] + 1)
        else:
            out.append((stage, phase, 1))
    return out


def test_registered_with_offset_params() -> None:
    assert controller_registry.get("fixed_time") is FixedTime
    controller, params = create_controller({"type": "fixed_time", "params": {"offset": 12.5}})
    assert isinstance(controller, FixedTime) and params.model_dump() == {"offset": 12.5}
    with pytest.raises(ConfigError, match="ofset: Extra inputs"):
        create_controller({"type": "fixed_time", "params": {"ofset": 1}})
    with pytest.raises(ConfigError):
        create_controller({"type": "fixed_time", "params": {"offset": math.nan}})


def test_realised_cycle_formula() -> None:
    _, prog = program_of(PHASES)
    assert realised_cycle(prog, 1.0) == 30 + 25 + 2 * (3 + 1)
    # n(x) = ceil(x/dt - 1e-9): 30.1/0.4 -> 76, 25/0.4 -> 63 (62.5), 2.5/0.4 -> 7, 0.7/0.4 -> 2
    _, prog = program_of(
        [{"green": EW, "duration": 30.1}, {"green": NS, "duration": 25}], yellow=2.5, all_red=0.7
    )
    assert realised_cycle(prog, 0.4) == pytest.approx((76 + 63 + 2 * (7 + 2)) * 0.4)
    # d_p = max(duration, min_green) (validation keeps duration >= min_green; the
    # controller does not rely on it)
    _, prog = program_of(PHASES)
    short = replace(prog, duration=np.array([3.0, 25.0]))
    assert realised_cycle(short, 1.0) == 5 + 25 + 2 * (3 + 1)
    # no intergreen after a phase whose successor keeps every green movement
    _, prog = program_of([{"green": EW, "duration": 30}, {"green": EW_PLUS, "duration": 30}])
    assert realised_cycle(prog, 1.0) == 30 + 30 + (3 + 1)


def test_the_cycle_runs_step_by_step(make_driver: Callable[..., Driver]) -> None:
    net, prog = program_of(PHASES)
    driver = make_driver(net, prog, "fixed_time")
    one = [
        (GREEN, 0, 30),
        (YELLOW, 0, 3),
        (ALL_RED, 0, 1),
        (GREEN, 1, 25),
        (YELLOW, 1, 3),
        (ALL_RED, 1, 1),
    ]
    assert _rle(_cycle(driver, 3 * 63)) == one * 3


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (0.0, (GREEN, 0, -1, 0.0, 0.0)),
        (10.0, (GREEN, 1, -1, 19.0, 19.0)),  # u = 53: 19 s into green 1 (starts at 34)
        (32.0, (YELLOW, 0, 1, 1.0, 31.0)),  # u = 31: the 2nd yellow step after green 0
        (30.0, (ALL_RED, 0, 1, 0.0, 33.0)),  # u = 33
        (0.5, (ALL_RED, 1, 0, 0.0, 28.0)),  # u = 62.5 -> step 62, all-red after phase 1
        (63.0 + 10.0, (GREEN, 1, -1, 19.0, 19.0)),  # modulo the cycle
        (-53.0, (GREEN, 1, -1, 19.0, 19.0)),
    ],
)
def test_cycle_position(offset: float, expected: tuple[Stage, int, int, float, float]) -> None:
    _, prog = program_of(PHASES)
    assert cycle_position(prog, 1.0, offset) == expected


@pytest.mark.parametrize("dt", [1.0, 0.5])
@pytest.mark.parametrize("offset", [0.0, 7.0, 31.25, 33.5, 61.0, -5.5])
def test_offsets_never_drift(make_driver: Callable[..., Driver], dt: float, offset: float) -> None:
    """The state applied in step k is the cycle position of ``k dt - offset``, forever."""
    net, prog = program_of(PHASES)
    driver = make_driver(net, prog, {"type": "fixed_time", "params": {"offset": offset}}, dt=dt)
    for k in range(int(4 * realised_cycle(prog, dt) / dt)):
        driver.step()
        stage, phase, target, _, _ = cycle_position(prog, dt, offset - k * dt)
        assert driver.state() == (stage, phase, target), k


def test_green_wave_offsets_between_intersections(make_driver: Callable[..., Driver]) -> None:
    net, prog = program_of(PHASES)
    starts: list[list[int]] = []
    for offset in (0.0, 15.0):
        driver = make_driver(net, prog, {"type": "fixed_time", "params": {"offset": offset}})
        k_start = []
        for _ in range(5 * 63):
            before = driver.state()
            driver.step()
            if driver.state()[:2] == (GREEN, 0) and before[:2] != (GREEN, 0):
                k_start.append(driver.step_count - 1)  # the step green 0 is first applied
        starts.append(k_start)
    assert starts[0] == [63, 126, 189, 252]
    assert starts[1] == [15, 78, 141, 204, 267]  # 15 s later in every cycle


def test_mid_run_install_continues_from_the_current_state(
    make_driver: Callable[..., Driver],
) -> None:
    net, prog = program_of(PHASES)
    params = {"type": "fixed_time", "params": {"offset": 10.0}}
    driver = make_driver(net, prog, params, initial=False)
    assert driver.state() == (GREEN, 0, -1)  # not moved to the offset position
    assert _rle(_cycle(driver, 31))[0] == (GREEN, 0, 30)
    setup = ControllerSetup(
        prog, FixedTimeParams(), np.random.default_rng(0), 1.0, False, _place=print
    )
    with pytest.raises(SimulationError, match="only allowed at a simulation reset"):
        setup.set_initial(0)
