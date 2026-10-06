"""Write and register your own signal controller, then compare it with max-pressure.

    uv run python examples/traffic_signal_control/custom_controller.py

The ``longest_queue`` controller switches, every ``check_every`` seconds of green, to the
phase that serves the most halted vehicles. The engine still enforces min-green, yellow
and all-red, so a controller can never create an unsafe transition.
"""

from __future__ import annotations

import os
from typing import cast

import numpy as np
from pydantic import BaseModel

from urbanflow import Simulation, bundled
from urbanflow.signals import (
    ControllerBase,
    ControllerContext,
    ControllerSetup,
    register_controller,
)


class LongestQueueParams(BaseModel):
    check_every: float = 10.0
    """Seconds of green between decisions."""


@register_controller("longest_queue")
class LongestQueue(ControllerBase):
    """Switch to the phase serving the most halted vehicles."""

    Params = LongestQueueParams

    def reset(self, setup: ControllerSetup) -> None:
        self.check_every = cast(LongestQueueParams, setup.params).check_every

    def decide(self, ctx: ControllerContext) -> int | None:
        if ctx.green_elapsed < max(ctx.min_green, self.check_every):
            return None
        best = int(np.argmax(ctx.phase_movements @ ctx.movement_halting))
        return best if best != ctx.phase else None


def main() -> None:
    duration = 300 if os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1" else 1800
    for controller in ("fixed_time", "longest_queue", "max_pressure"):
        sim = Simulation.from_scenario(
            bundled("grid_3x3"), controllers={"*": controller}, seed=3, duration=duration
        )
        s = sim.run().summary
        print(
            f"{controller:14s} travel {s['travel_time.mean']:6.1f} s  "
            f"delay {s['delay.mean']:5.1f} s"
            f"  waiting {s['waiting_time_mean']:5.1f} s  arrived {s['vehicles.arrived']:.0f}"
        )


if __name__ == "__main__":
    main()
