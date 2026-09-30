"""AT-11: yellow onset with one vehicle 10 m and one 60 m from the stop line (plan V, F.3).

Both drive at 13.9 m/s on the two lanes of approach W (dt 0.2, ``external`` control,
public API only); the near vehicle proceeds and the far one stops. The mechanisms (F.3):

* "near" is already committed when the yellow starts: in green a vehicle commits once it
  is inside its decision distance ``v^2/(2b) + v dt + 5 m``, about 56 m at 13.9 m/s, so
  it crosses on the intergreen as any committed vehicle does (committed vehicles never
  re-check the light, H.5). An *uncommitted* vehicle 10 m out at 13.9 m/s cannot arise
  in green; the dilemma branch for one (9.7 m/s^2 > ``YELLOW_MAX_DECEL``: proceed) is
  covered by ``tests/unit/engine/test_admission.py::test_yellow_dilemma*`` and the
  engine-level ``tests/integration/engine/test_signals.py``.
* "far" is still uncommitted at 60 m; stopping needs 1.6 m/s^2 <= ``YELLOW_MAX_DECEL``,
  so the yellow rule holds it and it stops at the line (0.5 m before it).
"""

from __future__ import annotations

import pytest

from urbanflow import ScenarioBuilder, Simulation
from urbanflow.core.types import Stage

pytestmark = pytest.mark.acceptance

ARMS = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}
DET = {"speed_factor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}}
DT, SPEED = 0.2, 13.89  # the speed limit: v0 with speed factor 1
MOVEMENT = "W_in->E_out"


def _gap(sim: Simulation, vid: str) -> float | None:
    """Distance of the vehicle's front to the stop line of approach W; None if it is not
    on the approach (not inserted yet, or past the line)."""
    car = sim.vehicles.get(vid)
    if car is None or car.road != "W_in" or car.lane is None or car.position is None:
        return None
    return sim.lanes[car.lane].length - car.position


def test_at11_yellow_dilemma() -> None:
    b = ScenarioBuilder("dilemma", dt=DT, duration=120.0)
    b.vehicle_type("det", **DET)
    b.intersection("J", (0.0, 0.0), kind="signalized")
    for name, (dx, dy) in ARMS.items():
        b.boundary(name, (dx * 300.0, dy * 300.0))
        b.road(f"{name}_in", name, "J", lanes=2)
        b.road(f"{name}_out", "J", name, lanes=2)
    # "far" departs 3.6 s later: 50 m behind at 13.89 m/s
    b.trip("near", 0.0, route=["W_in", "E_out"], vehicle_type="det", depart_lane=0)
    b.trip("far", 3.6, route=["W_in", "E_out"], vehicle_type="det", depart_lane=1)
    sim = Simulation(b.build(), controllers={"J": "external"}, debug_checks=True)
    phases = sim.signals.phases("J")
    green = next(p.index for p in phases if MOVEMENT in p.green)
    red = next(p.index for p in phases if MOVEMENT not in p.green)
    sim.signals.set_phase("J", green)

    step = SPEED * DT
    while (gap := _gap(sim, "near")) is None or gap > 10.0 + step / 2:
        sim.step()
    near_gap, far_gap = _gap(sim, "near"), _gap(sim, "far")
    assert near_gap == pytest.approx(10.0, abs=step / 2)
    assert far_gap == pytest.approx(60.0, abs=step)
    assert sim.vehicles["far"].speed == pytest.approx(SPEED, abs=0.05)
    raw = sim.state.raw  # live slot views
    committed = {int(u) for u in raw["uid"][raw["active"] & raw["committed"]].tolist()}
    assert sim.vehicles.uid("near") in committed  # committed in green (decision zone)
    assert sim.vehicles.uid("far") not in committed  # the yellow rule decides it

    sim.signals.request_phase("J", red)  # the yellow starts in the next step
    sim.step()
    assert sim.signals["J"].stage is Stage.yellow
    assert sim.signals["J"].movement_states[MOVEMENT] == "y"
    committed = {int(u) for u in raw["uid"][raw["active"] & raw["committed"]].tolist()}
    assert sim.vehicles.uid("far") not in committed  # held: v^2/(2d) <= YELLOW_MAX_DECEL
    crossed_at = None
    for _ in range(round(20.0 / DT)):
        if crossed_at is None and _gap(sim, "near") is None:
            crossed_at = sim.signals["J"].movement_states[MOVEMENT]
        assert _gap(sim, "far") is not None  # never crosses
        sim.step()
    assert crossed_at == "y"  # the near vehicle cleared the line during the yellow
    assert sim.signals["J"].movement_states[MOVEMENT] == "r"
    assert sim.vehicles["far"].speed < 1e-6
    assert 0.4 <= (_gap(sim, "far") or 0.0) <= 0.6
