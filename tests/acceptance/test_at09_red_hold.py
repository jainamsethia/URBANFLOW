"""AT-09 (user TEST 2): a red light holds approach A (plan section V, F.3, H.2).

While a 50-vehicle platoon streams through on green, a manual hold
(``sim.signals.hold_phase``) turns W -> E yellow, all-red and then red for 120 s. No
vehicle on A crosses its stop line while red, except vehicles that committed before the
red (during yellow), and the queue front stops 0.4-0.6 m before the line (the stop-line
obstacle leaves 0.5 m). The run is built in ``conftest.py`` with the public API only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

pytestmark = pytest.mark.acceptance


def test_at09_red_holds_approach_a(red_then_green: Callable[[float], Any]) -> None:
    run = red_then_green(1.0)
    assert run.crossed_in_yellow  # close vehicles proceed through the yellow
    assert run.crossed_in_red <= run.committed_at_red
    assert run.queue >= 30  # the rest of the platoon queues behind the line
    assert run.lane_length - 0.6 <= run.front_position <= run.lane_length - 0.4
    assert run.front_speed < 1e-6
