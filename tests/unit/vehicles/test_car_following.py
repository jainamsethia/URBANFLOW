"""IDM, the safe-speed cap and the ballistic update (plan G.1-G.3, G.6, R.2 physics tests)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from urbanflow.core import constants as C
from urbanflow.core.errors import ConfigError, NotFoundError
from urbanflow.core.types import FloatArray
from urbanflow.network import CompiledNetwork
from urbanflow.vehicles import (
    IDM,
    CarFollowingInputs,
    ParamArrays,
    VehicleTable,
    VehicleTypes,
    ballistic,
    car_following_registry,
    desired_speed,
    no_overshoot,
    register_car_following,
    safe_speed,
    stop_budget,
)

A, B, B_EMERG, T, S0, DELTA = 1.5, 2.0, 6.0, 1.5, 2.0, 4.0
V0 = 13.89


def _params(n: int) -> ParamArrays:
    full = lambda x: np.full(n, x)  # noqa: E731
    return ParamArrays(
        {
            "a": full(A),
            "b": full(B),
            "T": full(T),
            "s0": full(S0),
            "delta": full(DELTA),
            "b_emerg": full(B_EMERG),
            "b_hat": full(B_EMERG),
        }
    )


def _idm(v: FloatArray, gap: FloatArray, v_lead: FloatArray, v0: float = V0) -> FloatArray:
    n = len(v)
    x = CarFollowingInputs(
        speed=v,
        v0=np.full(n, v0),
        gap=gap,
        leader_speed=v_lead,
        dt=1.0,
        uid=np.arange(n, dtype=np.uint32),
        rng=np.random.default_rng(0),
    )
    return IDM().acceleration(x, _params(n))


def _one(v: float, gap: float, v_lead: float, v0: float = V0) -> float:
    return float(_idm(np.array([v]), np.array([gap]), np.array([v_lead]), v0)[0])


# --------------------------------------------------------------------------- IDM (G.1)
@pytest.mark.parametrize("v", [1.0, 4.0, 8.0, 11.0, 13.0])
def test_idm_equilibrium_gap(v: float) -> None:
    """R.2: the zero-acceleration gap is s_e(v) = (s0 + vT) / sqrt(1 - (v/v0)^delta) (1%)."""
    s_e = (S0 + v * T) / math.sqrt(1 - (v / V0) ** DELTA)
    lo, hi = 0.02, 1e4
    for _ in range(100):  # bisection on the monotone a(gap) at dv = 0
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _one(v, mid, v) < 0 else (lo, mid)
    assert lo == pytest.approx(s_e, rel=0.01)
    assert _one(v, 0.99 * s_e, v) < 0 < _one(v, 1.01 * s_e, v)


def test_free_road_terms() -> None:
    inf = np.full(3, np.inf)
    acc = _idm(np.array([0.0, V0, 2 * V0]), inf, np.full(3, np.nan))  # no leader: s = inf
    assert acc[0] == pytest.approx(A)
    assert acc[1] == pytest.approx(0.0)
    # IIDM bound: far above v0 the free term is -b/a, i.e. comfortable braking, not -b_emerg
    assert acc[2] == pytest.approx(-B)


def test_overlap_gives_emergency_braking() -> None:
    gaps = np.array([C.IDM_OVERLAP_GAP, 0.0, -1.0])
    acc = _idm(np.zeros(3), gaps, np.zeros(3))
    assert acc.tolist() == [-B_EMERG] * 3


def test_zero_desired_speed_is_finite() -> None:
    acc = _idm(np.array([0.0, 5.0]), np.full(2, np.inf), np.zeros(2), v0=0.0)
    assert acc.tolist() == [-B, -B]


@given(
    v=st.floats(0, 40),
    gap=st.floats(-5, 500) | st.just(math.inf),
    v_lead=st.floats(0, 40),
    v0=st.floats(1, 40),
)
def test_idm_output_is_clamped(v: float, gap: float, v_lead: float, v0: float) -> None:
    acc = _one(v, gap, v_lead, v0)
    assert math.isfinite(acc)
    assert -B_EMERG <= acc <= A


def test_no_overshoot_of_v0() -> None:
    acc = no_overshoot(np.array([1.5, 1.5, -1.0, 1.5]), np.array([13.0, 15.0, 13.0, 0.0]),
                       np.array([13.5, 13.5, 13.5, 13.5]), 1.0)  # fmt: skip
    assert acc.tolist() == pytest.approx([0.5, 0.0, -1.0, 1.5])


def test_idm_is_registered_and_models_can_be_added() -> None:
    assert car_following_registry.get("idm") is IDM
    assert IDM.name == "idm"

    @register_car_following("test_ovm")
    class OVM:
        Params: ClassVar[type[BaseModel]] = BaseModel

        def acceleration(self, x: CarFollowingInputs, p: ParamArrays) -> FloatArray:
            return np.zeros_like(x.speed) * p.a[0]

    assert OVM.name == "test_ovm"  # type: ignore[attr-defined]
    assert car_following_registry.get("test_ovm") is OVM
    with pytest.raises(ConfigError, match="already registered"):
        car_following_registry.register("test_ovm", IDM)
    with pytest.raises(NotFoundError, match="idm"):
        car_following_registry.get("idn")


# --------------------------------------------------------------------------- G.2 cap
def test_stop_budget_and_safe_speed_root() -> None:
    budget = stop_budget(np.array([30.0, np.inf]), np.array([10.0, np.nan]), np.array([5.0, 5.0]))
    assert budget[0] == pytest.approx(30 + 100 / 10 - C.SAFETY_MARGIN)
    assert budget[1] == np.inf
    v, dt, b_hat = np.array([12.0]), 1.0, np.array([4.0])
    r = safe_speed(v, budget[:1], b_hat, dt)
    # the root satisfies (v + v')/2 dt + v'^2 / (2 b_hat) = C exactly
    assert (v + r) / 2 * dt + r**2 / (2 * b_hat) == pytest.approx(budget[:1])
    assert safe_speed(v, np.array([np.inf]), b_hat, dt)[0] == np.inf
    # radicand < 0 -> -inf (the in-step stop branch of ballistic takes over)
    assert safe_speed(np.array([10.0]), np.array([-5.0]), b_hat, dt)[0] == -np.inf


def _stop_time_positions(v: float, b: float, times: FloatArray) -> FloatArray:
    t = np.minimum(times, v / b)
    return v * t - b * t * t / 2


@example(v=20.0, v_lead=10.0, slack=0.0, b_f=8.0, b_l=2.0, dt=1.0, a_frac=1.0)  # G.2 dip case
@given(
    v=st.floats(0, 30),
    v_lead=st.floats(0, 30),
    slack=st.floats(0, 50),
    b_f=st.floats(1, 9),
    b_l=st.floats(1, 9),
    dt=st.sampled_from([0.1, 0.5, 1.0, 2.0]),
    a_frac=st.floats(0, 1),
)
@settings(max_examples=300, deadline=None)
def test_safe_speed_never_overlaps_under_worst_case_braking(
    v: float, v_lead: float, slack: float, b_f: float, b_l: float, dt: float, a_frac: float
) -> None:
    """R.2: with the invariant v^2/(2 b_hat) <= C, the capped step followed by b_hat braking
    never overlaps the leader braking at b_L, at any intermediate time (incl. b_F > b_L)."""
    b_hat = min(b_f, b_l)
    gap = v * v / (2 * b_hat) - v_lead * v_lead / (2 * b_l) + C.SAFETY_MARGIN + slack
    assume(gap >= 0)
    a = -b_f + a_frac * (b_f + 3.0)  # any model output in [-b_F, 3]
    budget = stop_budget(np.array([gap]), np.array([v_lead]), np.array([b_l]))
    r = safe_speed(np.array([v]), budget, np.array([b_hat]), dt)
    v1, dx, violation = ballistic(np.array([v]), np.array([a]), budget, r, np.array([b_f]), dt)
    assert not violation[0]  # the invariant makes the cap feasible
    v1_, dx_ = float(v1[0]), float(dx[0])
    horizon = dt + v1_ / b_hat + v_lead / b_l + dt
    t = np.unique(np.r_[np.linspace(0, horizon, 4001), dt, v_lead / b_l])
    lead = gap + _stop_time_positions(v_lead, b_l, t)
    if v1_ > 0:  # constant acceleration in the step, then braking at b_hat
        acc = (v1_ - v) / dt
        inside = t <= dt
        follow = np.where(
            inside, v * t + acc * t * t / 2, dx_ + _stop_time_positions(v1_, b_hat, t - dt)
        )
    else:  # stopped inside the step
        b_star = v * v / (2 * dx_) if dx_ > 0 else math.inf
        follow = dx_ if dx_ == 0 else _stop_time_positions(v, b_star, t)
    assert np.min(lead - follow) >= -1e-6


# --------------------------------------------------------------------------- G.3 update
def _step(v: float, a: float, budget: float, b_f: float, dt: float, b_hat: float | None = None):
    vv, bb = np.array([v]), np.array([budget])
    r = safe_speed(vv, bb, np.array([b_hat or b_f]), dt)
    v1, dx, bad = ballistic(vv, np.array([a]), bb, r, np.array([b_f]), dt)
    return float(v1[0]), float(dx[0]), bool(bad[0])


def test_ballistic_moving_branch() -> None:
    v1, dx, bad = _step(10.0, 1.0, 1e9, 6.0, 1.0)  # free road: v' = v + a dt
    assert (v1, dx, bad) == (11.0, 10.5, False)
    v1, dx, bad = _step(10.0, 1.0, 12.0, 6.0, 1.0)  # capped by the safe speed
    assert 4.0 < v1 < 10.0  # above the floor v - b_F dt
    assert (10 + v1) / 2 + v1**2 / 12 == pytest.approx(12.0)
    assert dx == pytest.approx((10 + v1) / 2)
    assert not bad


def test_ballistic_floor_counts_a_violation() -> None:
    # C too small to stop from 20 m/s (r = -inf): the floor v - b_F dt binds; a vehicle
    # faster than b_F dt never "stops inside the step" harder than b_F
    for a in (-6.0, 1.0):
        v1, dx, bad = _step(20.0, a, 5.0, 6.0, 1.0)
        assert v1 == pytest.approx(14.0)
        assert dx == pytest.approx(17.0)
        assert bad
    v1, dx, bad = _step(20.0, 1.0, 30.0, 6.0, 1.0)  # r > 0 but below the floor
    assert (v1, bad) == (pytest.approx(14.0), True)


def test_braking_exactly_at_b_hat_is_not_a_violation() -> None:
    # along the b_hat = b_F braking plan the root equals the floor up to rounding
    for v in (0.7, 1.0, 13.9, 29.3):
        for dt in (0.1, 0.5, 1.0):
            budget = v * v / (2 * 6.0)
            v1, _, bad = _step(v, -6.0, budget, 6.0, dt)
            assert not bad
            assert v1 == pytest.approx(max(v - 6.0 * dt, 0.0), abs=1e-6)


def test_ballistic_in_step_stop() -> None:
    # v < b_hat dt with a small budget: stop inside the step, exactly within C
    v1, dx, bad = _step(2.0, 0.0, 0.8, 6.0, 1.0)
    b_star = min(6.0, max(0.0, 2.0, 4 / 1.6, 1e-9))
    assert (v1, bad) == (0.0, False)
    assert dx == pytest.approx(4 / (2 * b_star))
    assert dx <= 0.8
    assert _step(0.0, -1.0, -0.3, 6.0, 0.5) == (0.0, 0.0, False)  # standing stays put
    v1, dx, bad = _step(3.0, 0.0, 0.1, 6.0, 1.0)  # cannot stop within C even at b_F
    assert (v1, bad) == (0.0, True)
    assert dx == pytest.approx(9 / 12)


@given(
    v=st.floats(0, 3),
    a_frac=st.floats(0, 1),
    budget=st.floats(-0.5, 2),
    dt=st.sampled_from([0.1, 0.5, 1.0, 2.0]),
    b_f=st.floats(1, 9),
    b_hat_frac=st.floats(0.1, 1),
)
@settings(max_examples=500)
def test_ballistic_low_speed_band(
    v: float, a_frac: float, budget: float, dt: float, b_f: float, b_hat_frac: float
) -> None:
    """R.2 / G.3: v in [0,3], a in [-b_emerg, a_max], C in [-0.5, 2], dt in {0.1,0.5,1,2}:
    dx finite and >= 0, v' >= 0, and dx <= max(C, 0) whenever v^2/(2 max(C,1e-9)) <= b_F
    (up to the kernel's 1e-9 floor on C)."""
    a = -b_f + a_frac * (b_f + A)
    v1, dx, _ = _step(v, a, budget, b_f, dt, b_hat=b_hat_frac * b_f)
    assert math.isfinite(dx)
    assert dx >= 0
    assert v1 >= 0
    if v * v / (2 * max(budget, C.BALLISTIC_FLOOR)) <= b_f:
        assert dx <= max(budget, 0.0) + C.BALLISTIC_FLOOR


# --------------------------------------------------------------------------- R7
def test_desired_speed(corridor_net: CompiledNetwork) -> None:
    types = VehicleTypes.from_specs(corridor_net.vehicle_types, IDM.Params)
    veh = VehicleTable(4)
    car, emergency = types.index["car"], types.index["emergency"]
    h = [veh.alloc(x) for x in ("a", "b", "c")]
    veh.type_idx[h] = [car, car, emergency]
    veh.speed_factor[h] = [0.9, 1.1, 1.3]
    lane = corridor_net.lane_of("W_J1", 0)
    veh.link[h] = lane
    limit = corridor_net.link_speed_limit[lane]
    got = desired_speed(corridor_net, veh, types, np.array(h))
    expected = [min(36.1, f * limit) for f in (0.9, 1.1, 1.3)]
    assert got == pytest.approx(expected, rel=1e-6)  # speed_factor is stored as float32
    slow = corridor_net.n_lanes  # a connector: its curvature limit applies
    other = desired_speed(corridor_net, veh, types, np.array(h[:1]), np.array([slow]))
    assert other[0] == pytest.approx(min(36.1, 0.9 * corridor_net.link_speed_limit[slow]))
