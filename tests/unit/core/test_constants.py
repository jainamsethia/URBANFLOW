"""Physical sanity of the default numbers (plan G.7, E.5, E.7, G.9, AG.1 R9)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urbanflow.core import constants as C
from urbanflow.core.types import VehicleClass


def _car() -> dict[str, float]:
    return {
        "length": C.VEHICLE_LENGTH,
        "width": C.VEHICLE_WIDTH,
        "max_speed": C.VEHICLE_MAX_SPEED,
        "accel": C.IDM_ACCEL,
        "decel": C.IDM_DECEL,
        "emergency_decel": C.IDM_EMERGENCY_DECEL,
        "min_gap": C.IDM_MIN_GAP,
    }


def _builtin(name: str) -> dict[str, float]:
    params = _car()
    params.update({k: v for k, v in C.BUILTIN_VEHICLE_TYPES[name].items() if isinstance(v, float)})
    return params


def test_plan_values() -> None:
    # A few load-bearing numbers quoted verbatim by the plan.
    assert (C.DECISION_MARGIN, C.GAP_ACCEPT_MARGIN, C.ETA_END_ACCEL_FACTOR) == (5.0, 1.0, 0.5)
    assert (C.YELLOW_MAX_DECEL, C.SAFETY_MARGIN, C.TURN_LATERAL_ACCEL) == (3.0, 0.5, 2.0)
    assert (C.MAX_VEHICLE_WIDTH, C.LATERAL_MARGIN, C.MIN_LANE_LENGTH) == (2.6, 0.4, 5.0)
    assert math.isclose(C.CONFLICT_WIDTH, 3.0)
    assert (C.MITER_LIMIT, C.LANE_WIDTH, C.SPEED_LIMIT, C.SETBACK) == (4.0, 3.2, 13.89, 2.0)
    assert (C.LC_COOLDOWN, C.LC_VISUAL_DURATION, C.LC_MANDATORY_BIAS) == (3.0, 2.0, 1.0)
    assert (C.QUEUE_FRONT_TOLERANCE_M, C.VEH_SPACING_REF_M, C.METRICS_INTERVAL_S) == (
        10.0,
        7.5,
        10.0,
    )
    assert math.isclose(C.CONNECTOR_SAMPLE_ANGLE, math.pi / 16)


@pytest.mark.parametrize("name", sorted(C.BUILTIN_VEHICLE_TYPES))
def test_builtin_types_are_physical(name: str) -> None:
    p = _builtin(name)
    assert p["emergency_decel"] >= p["decel"] > 0  # assumption A1 (G.2)
    assert p["accel"] > 0
    assert 0 < p["width"] <= C.MAX_VEHICLE_WIDTH  # conflict zones cover every built-in
    assert 0 < p["length"] <= C.VEHICLE_LENGTH_MAX
    assert 0 < p["max_speed"] <= C.SPEED_LIMIT_MAX
    assert 0 <= p["min_gap"] <= C.MIN_GAP_MAX
    # The dilemma threshold lies between comfortable and emergency braking.
    assert p["decel"] <= C.YELLOW_MAX_DECEL <= p["emergency_decel"]


def test_speed_factor_ordering() -> None:
    assert C.SPEED_FACTOR_MIN <= C.SPEED_FACTOR_MEAN <= C.SPEED_FACTOR_MAX
    sf = C.BUILTIN_VEHICLE_TYPES["emergency"]["speed_factor"]
    assert sf["min"] <= sf["mean"] <= sf["max"]


def test_saturation_flow_matches_webster_default() -> None:
    # B.2 #25 (G.7 corrected): the built-in car's IDM equilibrium flow q(v) = v / (s_e + l),
    # s_e = (s0 + vT) / sqrt(1 - (v/v0)^delta), peaks at Webster's ~1800 veh/h/lane at the
    # default 13.9 m/s limit (T = 1.5 s peaks at ~1470: G.7 had dropped the delta term).
    # The measured queue discharge of this T is AT-10's (1500-1900 veh/h/lane).
    v0 = C.SPEED_LIMIT
    v = np.linspace(0.01, v0, 100_000, endpoint=False)
    s_e = (C.IDM_MIN_GAP + v * C.IDM_HEADWAY) / np.sqrt(1 - (v / v0) ** C.IDM_DELTA)
    flow = C.SECONDS_PER_HOUR * v / (s_e + C.VEHICLE_LENGTH)
    assert flow.max() == pytest.approx(C.WEBSTER_SATURATION_FLOW, rel=0.01)


def test_defaults_within_their_bounds() -> None:
    assert C.DT_MIN <= C.DT_RECOMMENDED_MIN <= C.DT <= C.DT_RECOMMENDED_MAX <= C.DT_MAX
    assert C.LANE_WIDTH_MIN <= C.LANE_WIDTH <= C.LANE_WIDTH_MAX
    assert C.SPEED_LIMIT_MIN <= C.SPEED_LIMIT <= C.SPEED_LIMIT_MAX
    assert 0 <= C.SIGNAL_YELLOW <= C.SIGNAL_INTERGREEN_MAX
    assert 0 <= C.SIGNAL_ALL_RED <= C.SIGNAL_INTERGREEN_MAX
    assert 0 <= C.DEFAULT_MIN_GREEN_S < C.SIGNAL_MAX_GREEN <= C.MAX_GREEN_MAX
    assert 0 < C.PHASE_DURATION <= C.PHASE_DURATION_MAX
    assert C.WEBSTER_CYCLE_MIN < C.WEBSTER_CYCLE_MAX
    assert C.STRAIGHT_MAX_ANGLE < C.UTURN_MIN_ANGLE <= math.pi
    assert C.LOW_ANGLE_CROSSING < C.STRAIGHT_MAX_ANGLE
    assert C.MITER_LIMIT > 1


def test_energy_table_covers_every_class() -> None:
    assert set(C.ENERGY_DEFAULTS) == {v.value for v in VehicleClass}
    for vclass in VehicleClass:
        params = C.ENERGY_DEFAULTS[vclass]  # StrEnum members index str keys
        assert params.fuel in C.CO2_G_PER_MJ
        assert 0 < params.efficiency < 1
        assert params.mass_kg > 0 and params.cda_m2 > 0 and params.idle_kw > 0


def test_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        C.ENERGY_DEFAULTS["car"] = C.ENERGY_DEFAULTS["bus"]
    with pytest.raises(TypeError):
        C.BUILTIN_VEHICLE_TYPES["bus"]["length"] = 1.0
