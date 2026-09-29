"""Acceleration, anticipation, overrides and the safe-speed cap (plan G.1-G.3)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest
from pydantic import BaseModel

from urbanflow.core import constants as C
from urbanflow.core.types import FloatArray
from urbanflow.engine.longitudinal import (
    Longitudinal,
    anticipation,
    expire_overrides,
    longitudinal_step,
)
from urbanflow.vehicles import IDM, CarFollowingInputs, CarFollowingModel, ParamArrays

from .conftest import World

WE = ("W_in", "E_out")
RNG = np.random.default_rng(0)


def _step(
    w: World,
    dt: float,
    *,
    time: float = 0.0,
    stop_at: float | None = None,
    model: CarFollowingModel | None = None,
) -> Longitudinal:
    """One longitudinal update of every vehicle, optionally with an obstacle at ``stop_at``."""
    run = w.run()
    expire_overrides(w.veh, run, time)
    inf = np.full(run.size, np.inf)
    obstacle, cap = inf, inf
    if stop_at is not None:
        d = stop_at - w.veh.pos[run]
        obstacle, cap = d + w.types.min_gap[w.veh.type_idx[run]] - C.STOP_LINE_CLEARANCE, d
    cf = model if model is not None else IDM()
    return longitudinal_step(
        w.net, w.veh, w.types, cf, run, w.leaders(), obstacle, cap, dt=dt, rng=RNG
    )


@pytest.mark.parametrize("dt", [0.25, 0.5, 1.0])
def test_stops_half_a_metre_before_the_obstacle(junction: World, dt: float) -> None:
    w = junction
    line = float(w.net.link_length[w.link("W_in_0")])
    h = w.place("W_in_0", 20.0, 13.0, route=WE)
    decel = []
    for _ in range(int(200 / dt)):
        lon = _step(w, dt, stop_at=line)
        w.veh.pos[h] += lon.dx[0]
        decel.append(-float(w.veh.accel[h]))
        assert lon.violations == 0
    assert line - 0.6 <= w.veh.pos[h] <= line - 0.4
    assert w.veh.speed[h] == pytest.approx(0.0, abs=1e-3)
    assert max(decel) <= C.IDM_DECEL  # comfortable: R.2 "decel <= b"


def test_anticipation_toward_a_slower_connector(junction: World) -> None:
    w = junction
    lane = float(w.net.link_length[w.link("W_in_0")])
    right = w.link("W_in_0->S_out_0")
    v_c = float(w.net.link_speed_limit[right])
    near = w.place("W_in_0", lane - 30.0, 13.0, route=("W_in", "S_out"))
    far = w.place("W_in_0", lane - 200.0, 13.0, route=("W_in", "S_out"))
    straight = w.place("E_in_0", lane - 30.0, 13.0, route=("E_in", "W_out"))
    on_conn = w.place("N_in_0->W_out_0", 1.0, 2.0, route=("N_in", "W_out"))
    assert w.veh.next_conn[near] == right
    a_lim, v0_end = anticipation(w.net, w.veh, w.types, w.run(), 1.0)
    r = (13.0**2 - v_c**2) / (2 * 30.0)
    assert r > C.ANTICIPATION_DECEL_FACTOR * C.IDM_DECEL
    assert w.at(a_lim, near) == pytest.approx(-min(r, C.IDM_EMERGENCY_DECEL))
    assert math.isinf(w.at(a_lim, far))  # r = 0.4 < b/2
    assert math.isinf(w.at(a_lim, straight))  # the straight connector is not slower
    assert math.isinf(w.at(a_lim, on_conn))  # heading to a faster lane
    assert w.at(v0_end, near) == pytest.approx(w.veh.v0[near])  # 30 m: out of reach
    # the override replaces the desired speed on the next link too
    w.veh.speed_override[straight] = 5.0
    a_lim, _ = anticipation(w.net, w.veh, w.types, w.run(), 1.0)
    r = (13.0**2 - 25.0) / (2 * 30.0)
    assert w.at(a_lim, straight) == pytest.approx(-r)


def test_anticipation_in_the_last_step_before_a_slower_link(junction: World) -> None:
    """Within reach of the link, brake at r below the b/2 threshold; no overshoot of v_n."""
    w = junction
    lane = float(w.net.link_length[w.link("W_in_0")])
    v_c = float(w.net.link_speed_limit[w.link("W_in_0->S_out_0")])
    d, v = 2.7, v_c + 0.8
    fast = w.place("W_in_0", lane - d, v, route=("W_in", "S_out"))
    slow = w.place("E_in_0", lane - 1.5, v_c - 1.0, route=("E_in", "N_out"))  # within reach
    r = (v**2 - v_c**2) / (2 * d)
    assert r < C.ANTICIPATION_DECEL_FACTOR * C.IDM_DECEL  # the plain rule would not brake
    a_lim, v0_end = anticipation(w.net, w.veh, w.types, w.run(), 1.0)
    assert w.at(a_lim, fast) == pytest.approx(-r)
    assert w.at(v0_end, fast) == pytest.approx(v_c)
    assert math.isinf(w.at(a_lim, slow))
    _step(w, 1.0)
    assert w.veh.speed[fast] <= v_c  # enters the connector at no more than its v0
    assert w.veh.speed[slow] <= w.at(v0_end, slow) + 1e-12  # accelerates to v_n at most
    assert w.at(v0_end, slow) == pytest.approx(w.net.link_speed_limit[w.link("E_in_0->N_out_0")])


def test_override_replaces_v0_until_it_expires(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", 0.0, 10.0, route=WE)
    w.veh.speed_override[h] = 4.0
    w.veh.override_until[h] = 30.0
    for n in range(30):
        _step(w, 1.0, time=float(n))
    assert w.veh.speed[h] == pytest.approx(4.0, abs=0.05)
    assert w.veh.speed_override[h] == 4.0  # persistent until override_until
    _step(w, 1.0, time=30.0)
    assert math.isnan(w.veh.speed_override[h]) and math.isinf(w.veh.override_until[h])
    assert w.veh.speed[h] > 4.0  # accelerating back toward v0


def test_never_overshoots_v0(junction: World) -> None:
    w = junction
    v0 = C.SPEED_LIMIT
    h = w.place("W_in_0", 0.0, v0 - 0.01, route=WE)
    assert w.veh.v0[h] == pytest.approx(v0)
    _step(w, 2.0)
    assert w.veh.speed[h] <= v0 + 1e-12


def test_safety_cap_violation_is_counted(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", 50.0, 15.0, route=WE)
    w.place("W_in_0", 57.5, 0.0, route=WE)  # 2.5 m ahead, stopped
    lon = _step(w, 1.0)
    assert lon.violations == 1
    assert w.veh.speed[h] >= 15.0 - C.IDM_EMERGENCY_DECEL - 1e-9  # the b_F floor
    assert w.veh.accel[h] == pytest.approx(w.veh.speed[h] - 15.0)


def test_safe_cap_keeps_the_follower_behind(junction: World) -> None:
    w = junction
    f = w.place("W_in_0", 20.0, 14.0, route=WE)
    lead = w.place("W_in_0", 80.0, 14.0, route=WE)
    w.veh.speed_override[lead] = 0.0  # the leader brakes to a stop
    for _ in range(40):
        lon = _step(w, 1.0)
        w.veh.pos[w.run()] += lon.dx
        assert lon.violations == 0
        assert w.veh.pos[lead] - w.veh.length[lead] - w.veh.pos[f] > C.SAFETY_MARGIN - 1e-9


class _Wild:
    """A buggy model: huge accelerations of alternating sign."""

    name: ClassVar[str] = "wild"
    Params: ClassVar[type[BaseModel]] = IDM.Params
    sign = 1.0

    def acceleration(self, x: CarFollowingInputs, p: ParamArrays) -> FloatArray:  # noqa: ARG002
        return np.full(x.speed.size, 50.0 * self.sign)


def test_model_output_is_clamped(junction: World) -> None:
    w = junction
    h = w.place("W_in_0", 20.0, 5.0, route=WE)
    model = _Wild()
    _step(w, 1.0, model=model)
    assert w.veh.accel[h] == pytest.approx(C.IDM_ACCEL)
    model.sign = -1.0
    _step(w, 1.0, model=model)
    assert w.veh.accel[h] == pytest.approx(-C.IDM_EMERGENCY_DECEL)
