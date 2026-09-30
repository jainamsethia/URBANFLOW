"""AT-10 (user TEST 3): after AT-09, switch approach A to green (plan section V, G.7).

Queued vehicles proceed within 2 s of the green, the queue clears, and the discharge
(saturation) flow, measured from the headways of the 5th to the 15th queued vehicle, is
1500-1900 veh/h/lane at every recommended step length. The car's time gap T is calibrated
for this band (B.2 #25): about 1584 / 1616 / 1636 veh/h at dt 1.0 / 0.5 / 0.2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

pytestmark = pytest.mark.acceptance


def test_at10_queue_discharges(red_then_green: Callable[[float], Any]) -> None:
    run = red_then_green(1.0)
    assert run.green_time > 0 and run.first_move_time - run.green_time <= 2.0
    assert len(run.crossings) == run.queue >= 30  # the queue clears
    assert run.sim.lanes["W_in_0"].vehicle_count == 0
    assert run.crossings[0] - run.green_time <= 2.0  # the front crosses at once


@pytest.mark.parametrize("dt", [1.0, 0.5, 0.2])
def test_at10_discharge_flow_band(red_then_green: Callable[[float], Any], dt: float) -> None:
    run = red_then_green(dt)
    assert run.first_move_time - run.green_time <= 2.0
    assert 1500.0 <= run.discharge_vph() <= 1900.0
