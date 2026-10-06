"""IDM lane capacity (core/capacity.py) and its two users: the compiler and W501."""

from __future__ import annotations

import numpy as np
import pytest

from urbanflow.core import constants as C
from urbanflow.core.capacity import idm_capacity_vph
from urbanflow.network import compile_network
from urbanflow.scenario import ScenarioBuilder


def _brute(v0: float, headway: float, min_gap: float, length: float, delta: float) -> float:
    """``3600 max v / (s_e(v) + l)`` on a 10^6-point grid."""
    v = v0 * np.arange(1, 1_000_000) / 1_000_000
    gap = (min_gap + v * headway) / np.sqrt(1.0 - (v / v0) ** delta)
    return float(3600.0 * (v / (gap + length)).max())


@pytest.mark.parametrize(
    ("v0", "headway", "min_gap", "length", "delta"),
    [
        (13.89, C.IDM_HEADWAY, C.IDM_MIN_GAP, C.VEHICLE_LENGTH, C.IDM_DELTA),
        (2.0, C.IDM_HEADWAY, C.IDM_MIN_GAP, C.VEHICLE_LENGTH, C.IDM_DELTA),
        (33.3, 1.5, 2.5, 12.0, C.IDM_DELTA),  # a bus on a motorway
        (8.0, 1.0, 1.0, 4.0, 2.0),
    ],
)
def test_matches_a_brute_force_maximisation(
    v0: float, headway: float, min_gap: float, length: float, delta: float
) -> None:
    got = idm_capacity_vph(v0, headway=headway, min_gap=min_gap, length=length, delta=delta)
    assert got == pytest.approx(_brute(v0, headway, min_gap, length, delta), rel=1e-5)


def test_built_in_car_at_the_default_speed_limit() -> None:
    """~1790 veh/h/lane at 13.89 m/s with T = 1.1 s (the lead's figure, B.2 #25), well
    below 3600 / (T + (l + s0)/v) = 2244, which drops the (v/v0)^delta term."""
    q = idm_capacity_vph(C.SPEED_LIMIT)
    assert q == pytest.approx(1792.26, abs=0.01)
    simple = 3600 / (C.IDM_HEADWAY + (C.VEHICLE_LENGTH + C.IDM_MIN_GAP) / C.SPEED_LIMIT)
    assert simple == pytest.approx(2244.4, abs=0.1) and q < simple
    # monotone in the speed limit, and in the time gap the other way
    assert idm_capacity_vph(10.0) < q < idm_capacity_vph(20.0)
    assert idm_capacity_vph(C.SPEED_LIMIT, headway=1.5) < q


def _one_road(rate: float, lanes: int = 1) -> ScenarioBuilder:
    b = ScenarioBuilder("capacity")
    b.boundary("A", (0.0, 0.0))
    b.boundary("B", (300.0, 0.0))
    b.road("AB", "A", "B", lanes=lanes)
    b.flow("f", route=["AB"], rate=rate)
    return b


def test_road_capacity_and_w501_use_the_same_formula() -> None:
    cap = idm_capacity_vph(C.SPEED_LIMIT)
    net = compile_network(_one_road(100.0, lanes=2).build())
    assert net.road_capacity_vph.tolist() == [pytest.approx(2 * cap)]
    below = _one_road(cap - 10).build()
    assert "W501" not in {i.code for i in below.issues}
    above = _one_road(cap + 10).build()
    w501 = [i for i in above.issues if i.code == "W501"]
    assert w501 and f"~{cap:.0f} veh/h" in w501[0].message


def test_w501_averages_a_type_mix_by_headway() -> None:
    b = _one_road(100.0)
    b.flow("mixed", route=["AB"], rate=100.0, type_mix={"car": 1.0, "bus": 1.0})
    bus = C.BUILTIN_VEHICLE_TYPES["bus"]
    q_bus = idm_capacity_vph(
        C.SPEED_LIMIT,
        headway=C.IDM_HEADWAY,
        min_gap=float(bus["min_gap"]),  # type: ignore[arg-type]
        length=float(bus["length"]),  # type: ignore[arg-type]
    )
    mix = 1 / (0.5 / idm_capacity_vph(C.SPEED_LIMIT) + 0.5 / q_bus)
    b.update("flow", "mixed", rate=mix + 10)
    issues = b.build().issues
    msgs = [i.message for i in issues if i.code == "W501"]
    assert len(msgs) == 1 and f"~{mix:.0f} veh/h" in msgs[0]


def test_w501_uses_each_types_desired_speed_and_delta() -> None:
    """P4 review: W501 used the road speed limit and delta = 4 for every type, while the
    engine's v0 is min(max_speed, f limit): a bus (max 25 m/s) on a 33.3 m/s road got
    1900 instead of ~1705 veh/h, a delta = 1 type 1792 instead of ~1343 veh/h."""
    bus = C.BUILTIN_VEHICLE_TYPES["bus"]
    q_bus = idm_capacity_vph(
        float(bus["max_speed"]),  # type: ignore[arg-type]
        headway=C.IDM_HEADWAY,
        min_gap=float(bus["min_gap"]),  # type: ignore[arg-type]
        length=float(bus["length"]),  # type: ignore[arg-type]
    )
    q_soft = idm_capacity_vph(C.SPEED_LIMIT, delta=1.0)
    assert q_soft < 0.8 * idm_capacity_vph(C.SPEED_LIMIT)
    for vtype, speed, cap in (("bus", 33.3, q_bus), ("soft", C.SPEED_LIMIT, q_soft)):
        b = ScenarioBuilder("capacity")
        b.vehicle_type("soft", model_params={"delta": 1.0})
        b.boundary("A", (0.0, 0.0))
        b.boundary("B", (300.0, 0.0))
        b.road("AB", "A", "B", speed_limit=speed)
        b.flow("f", route=["AB"], rate=cap + 10, vehicle_type=vtype)
        msgs = [i.message for i in b.build().issues if i.code == "W501"]
        assert len(msgs) == 1 and f"~{cap:.0f} veh/h" in msgs[0], (vtype, msgs)
        b.update("flow", "f", rate=cap - 10)
        assert "W501" not in {i.code for i in b.build().issues}
