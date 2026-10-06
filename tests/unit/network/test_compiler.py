"""compile_network: counts, ids, layout, determinism, diagnostics (plan E.5 rules 1-8)."""

from __future__ import annotations

import copy
import logging
import math
import zlib
from typing import Any

import numpy as np
import pytest

import urbanflow
from urbanflow.core.constants import (
    DETECTOR_LENGTH,
    TURN_LATERAL_ACCEL,
)
from urbanflow.core.errors import ScenarioValidationError, Severity
from urbanflow.core.types import IntersectionKind, TurnKind
from urbanflow.network import CompiledNetwork, CompileReport, compile_network
from urbanflow.scenario import Scenario, derive
from urbanflow.scenario.schema import ScenarioSpec

N_IN, S_OUT, E_OUT, W_OUT = 0, 3, 5, 7  # road indices in the demo


def _compile_error(scenario: Scenario) -> list[tuple[str, str, str]]:
    with pytest.raises(ScenarioValidationError) as info:
        compile_network(scenario)
    return [(i.code, i.path, i.message) for i in info.value.issues]


def _chain(gap: float, flows: list[dict[str, Any]]) -> dict[str, Any]:
    """A -> K1 -> K2 -> B, one lane each; K1/K2 derive to uncontrolled with radius 5.2 m."""
    return {
        "format": "urbanflow.scenario",
        "version": "1.0",
        "meta": {"name": "chain"},
        "network": {
            "intersections": [
                {"id": "A", "point": [-200, 0]},
                {"id": "K1", "point": [0, 0]},
                {"id": "K2", "point": [gap, 0]},
                {"id": "B", "point": [gap + 200, 0]},
            ],
            "roads": [
                {"id": "A_K1", "from": "A", "to": "K1", "lanes": [{}]},
                {"id": "K1_K2", "from": "K1", "to": "K2", "lanes": [{}]},
                {"id": "K2_B", "from": "K2", "to": "B", "lanes": [{}]},
            ],
        },
        "demand": {"flows": flows},
    }


# ---------------------------------------------------------------------------- layout
def test_counts_match_the_spec(demo: Scenario, demo_net: CompiledNetwork) -> None:
    counts = demo.summary()
    assert demo_net.n_intersections == counts["intersections"]
    assert demo_net.n_roads == counts["roads"]
    assert demo_net.n_lanes == counts["lanes"]
    assert demo_net.n_movements == counts["movements"]
    assert demo_net.n_conn == counts["connections"]
    assert demo_net.int_program.tolist() == [0, -1, -1, -1, -1]
    assert demo_net.int_kind[0] == IntersectionKind.signalized.code


def test_ids_and_indices(demo: Scenario, demo_net: CompiledNetwork) -> None:
    net = demo_net
    assert net.road_ids == tuple(r.id for r in demo.resolved.network.roads)
    assert net.int_ids == ("J", "N", "E", "S", "W")
    for road in net.road_ids:
        for k in range(2):
            assert net.link_ids[net.lane_of(road, k)] == f"{road}_{k}"
    assert net.link_ids[16] == "E_in_1->N_out_1"  # connectors follow movement order
    assert all(net.link_index[lid] == k for k, lid in enumerate(net.link_ids))
    assert all(net.mov_index[mid] == k for k, mid in enumerate(net.mov_ids))
    assert net.mov_ids[:3] == ("E_in->N_out", "E_in->S_out", "E_in->W_out")
    assert [net.link_ids[c] for c in net.road_pair_conns[(N_IN, S_OUT)]] == [
        "N_in_0->S_out_0",
        "N_in_1->S_out_1",
    ]
    assert net.vehicle_types == demo.resolved.vehicle_types


def test_road_pair_mask(demo_net: CompiledNetwork) -> None:
    mask = demo_net.road_pair_mask
    assert (mask[(N_IN, E_OUT)], mask[(N_IN, S_OUT)], mask[(N_IN, W_OUT)]) == (0b01, 0b11, 0b10)
    assert len(mask) == demo_net.n_movements
    assert (N_IN, 1) not in mask  # no U-turns


def test_boundary_exits(demo_net: CompiledNetwork) -> None:
    net = demo_net
    exits = [net.link_ids[k] for k in np.flatnonzero(net.link_is_exit)]
    assert exits == [f"{r}_{k}" for r in ("N_out", "S_out", "E_out", "W_out") for k in (0, 1)]
    assert net.link_intersection[net.lane_of("N_in", 0)] == 0
    assert net.link_intersection[net.lane_of("N_out", 0)] == net.int_index["N"]


def test_canonical_approach_order_is_clockwise_from_north(demo_net: CompiledNetwork) -> None:
    net = demo_net
    ins = net.int_in_lanes[net.int_in_ptr[0] : net.int_in_ptr[1]]
    outs = net.int_out_lanes[net.int_out_ptr[0] : net.int_out_ptr[1]]
    assert [net.link_ids[k] for k in ins] == [
        f"{a}_in_{k}" for a in ("N", "E", "S", "W") for k in (0, 1)
    ]
    assert [net.link_ids[k] for k in outs] == [
        f"{a}_out_{k}" for a in ("N", "E", "S", "W") for k in (0, 1)
    ]


def test_golden_summary(demo: Scenario, demo_net: CompiledNetwork) -> None:
    net = demo_net
    assert np.bincount(net.conf_kind, minlength=3).tolist() == [36, 8, 8]
    np.testing.assert_allclose(net.road_length, 191.6)  # 200 m minus J's radius 8.4 m
    assert float(net.link_length.sum()) == pytest.approx(3285.326, abs=1e-3)
    assert sorted(set(np.round(net.link_length[16:], 3).tolist())) == [5.647, 15.685, 16.8]
    assert net.origin.tolist() == [-200.0, -200.0]
    assert net.bbox.tolist() == [0.0, 0.0, 400.0, 400.0]
    np.testing.assert_allclose(net.int_point[0], [200.0, 200.0])
    assert net.scenario_hash == demo.content_hash
    assert net.geometry_crc == zlib.crc32(demo.content_hash.encode("ascii"))


def test_limits_capacity_and_detectors(demo_net: CompiledNetwork) -> None:
    net = demo_net
    # delta-aware IDM peak flow of the built-in car at 13.89 m/s: 1792.26 veh/h/lane (not
    # 3600 / (T + (l + s0)/v) = 2244 veh/h/lane, which drops the (v/v0)^delta term)
    np.testing.assert_allclose(net.road_capacity_vph, 2 * 1792.2556, rtol=1e-6)
    np.testing.assert_allclose(net.lane_detector_start, 191.6 - DETECTOR_LENGTH)
    assert net.link_speed_limit[net.lane_of("W_in", 1)] == 11.11
    left = net.link_index["N_in_0->E_out_0"]  # ~quarter circle of radius 10 m
    assert net.link_speed_limit[left] == pytest.approx(math.sqrt(TURN_LATERAL_ACCEL * 10), rel=0.03)
    assert net.link_speed_limit[net.link_index["W_in_1->E_out_1"]] == 11.11  # min(v_in, v_out)


def test_static_ranks() -> None:
    for kind, expected in (("signalized", {0}), ("uncontrolled", {1})):
        net = compile_network(urbanflow.generate("single_intersection", kind=kind))
        assert set(net.mov_static_rank.tolist()) == expected
    sc = urbanflow.generate("single_intersection", kind="priority")
    net = compile_network(sc)
    movements = sc.resolved.network.intersections[0].movements or ()
    for mov, rank in zip(movements, net.mov_static_rank, strict=True):
        if mov.priority == "minor":
            assert rank == 1
        else:
            assert rank == (2 if mov.turn in (TurnKind.left, TurnKind.uturn) else 3)


def test_determinism(demo: Scenario, demo_net: CompiledNetwork) -> None:
    again = compile_network(Scenario.from_spec(demo.spec))
    for name, arr in demo_net.arrays().items():
        other = again.arrays()[name]
        assert arr.dtype == other.dtype and np.array_equal(arr, other), name
    assert dict(again.road_pair_conns) == dict(demo_net.road_pair_conns)
    assert again.link_ids == demo_net.link_ids and again.geometry_crc == demo_net.geometry_crc


def test_left_hand_traffic_mirrors_lanes(demo_data: dict[str, Any]) -> None:
    right = compile_network(Scenario.from_dict(demo_data))
    demo_data["network"]["drive_side"] = "left"
    left = compile_network(Scenario.from_dict(demo_data))
    for net, sign in ((right, -1), (left, 1)):
        world = net.link_points(net.lane_of("N_in", 0)) + net.origin  # southbound
        np.testing.assert_allclose(world[:, 0], sign * 1.6)


def test_trimming_follows_the_polyline(demo_data: dict[str, Any]) -> None:
    # N_in ends 1 m from J after a 3.16 m kink: the 7.4 m trim runs around the kink along
    # arc length and ends on the straight part (CityFlow would overshoot along the kink)
    demo_data["network"]["roads"][0]["points"] = [[0, 200], [0, 3], [-1, 0]]
    net = compile_network(Scenario.from_dict(demo_data))
    kink = math.hypot(1, 3)
    assert net.road_length[N_IN] == pytest.approx(197 + kink - 7.4)
    end = net.link_points(net.lane_of("N_in", 0))[-1] + net.origin
    np.testing.assert_allclose(end, [-1.6, 3 + 7.4 - kink], atol=1e-9)


def test_report_holds_warnings_only(demo_net: CompiledNetwork) -> None:
    assert demo_net.report == CompileReport()
    with pytest.raises(ValueError, match="warnings only"):
        CompileReport((urbanflow.ValidationIssue("$", "x", "E905"),))


# ---------------------------------------------------------------------------- diagnostics
def test_w302_protected_crossing(demo_data: dict[str, Any]) -> None:
    demo_data["network"]["intersections"][0]["signal"]["phases"][0]["green"]["N_in->E_out"] = "G"
    (warning,) = compile_network(Scenario.from_dict(demo_data)).report.warnings
    assert (warning.code, warning.path, warning.severity) == (
        "W302",
        "network.intersections[0].signal.phases[0]",
        Severity.warning,
    )
    assert warning.message == (
        'movements "N_in->E_out" and "S_in->N_out" cross but are both protected (G); '
        "make one permissive (g)"
    )


def test_w302_on_derived_phases_points_at_the_signal(
    demo_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Derived phases are not in the file: W302 is reported at the signal instead."""

    def all_protected(group: tuple[str, ...], by_road: Any, _far: Any, _near: Any) -> Any:
        return [{str(m.id): "G" for road in group for m in by_road.get(road, ())}]

    templates = {**derive.SIGNAL_TEMPLATES, "two_phase": all_protected}
    monkeypatch.setattr(derive, "SIGNAL_TEMPLATES", templates)
    del demo_data["network"]["intersections"][0]["signal"]["phases"]
    warnings = compile_network(Scenario.from_dict(demo_data)).report.warnings
    assert warnings and {(w.code, w.path) for w in warnings} == {
        ("W302", "network.intersections[0].signal")
    }


def test_compile_is_logged(demo: Scenario, caplog: pytest.LogCaptureFixture) -> None:
    """AB 6.4: "network compiled" (INFO) with lanes, connectors, conflicts, duration_ms."""
    with caplog.at_level(logging.INFO, logger="urbanflow.network"):
        net = compile_network(demo)
    (record,) = [r for r in caplog.records if r.message == "network compiled"]
    assert record.name == "urbanflow.network"
    fields = record.__dict__
    assert (fields["lanes"], fields["connectors"], fields["conflicts"]) == (
        net.n_lanes,
        net.n_conn,
        net.n_conflicts,
    )
    assert fields["duration_ms"] >= 0


def test_e802_short_road_is_rejected_even_without_validation() -> None:
    spec = ScenarioSpec.model_validate(_chain(14.0, []))
    found = _compile_error(Scenario(spec))
    assert found == [
        (
            "E802",
            "network.roads[1]",
            'road is 14.0 m long but intersections "K1" and "K2" reserve 5.2 m + 5.2 m; '
            "lanes would be 3.6 m (minimum 5.0 m)",
        )
    ]


def test_e905_hairpin_lane() -> None:
    data = _chain(100.0, [])
    data["network"]["intersections"] = [
        {"id": "H1", "point": [0, 0]},
        {"id": "H2", "point": [0, -30]},
    ]
    data["network"]["roads"] = [
        {
            "id": "H",
            "from": "H1",
            "to": "H2",
            "points": [[0, 0], [60, 0], [60, -2], [40, -2], [40, -30], [0, -30]],
            "lanes": [{}],
        }
    ]
    scenario = Scenario.from_dict(data)  # sharp bends only warn (W802)
    ((code, path, message),) = _compile_error(scenario)
    assert (code, path) == ("E905", "network.roads[0]")
    assert message.startswith("lane geometry is degenerate: lane 0 reverses direction")


def test_e905_zero_length_connector(demo_data: dict[str, Any]) -> None:
    ix = demo_data["network"]["intersections"][0]
    ix["radius"] = 0
    ix["movements"] = [
        {"from_road": "N_in", "to_road": "S_out", "connections": [{"from_lane": 0, "to_lane": 0}]}
    ]
    ix["signal"]["phases"] = [{"green": {"N_in->S_out": "G"}}]
    demo_data["demand"] = {}
    ((code, path, message),) = _compile_error(Scenario.from_dict(demo_data))
    assert (code, path) == ("E905", "network.intersections[0].movements[0].connections[0]")
    assert 'connector "N_in_0->S_out_0": the connector has zero length' in message
    derived = copy.deepcopy(demo_data)
    del derived["network"]["intersections"][0]["movements"]
    del derived["network"]["intersections"][0]["signal"]["phases"]
    paths = {path for _, path, _ in _compile_error(Scenario.from_dict(derived))}
    assert paths == {"network.intersections[0]"}  # derived movements are not in the file


BUS = {"vehicle_type": "bus", "rate": 60}


@pytest.mark.parametrize(
    "flow",
    [
        {"id": "f", "route": ["A_K1", "K1_K2", "K2_B"], **BUS},  # connector target lane
        {"id": "f", "origin": "A_K1", "destination": "K2_B", **BUS},  # OD: shortest path
        {"id": "f", "route": ["K1_K2", "K2_B"], **BUS},  # first-road lane
        {
            "id": "f",
            "routes": [{"roads": ["K1_K2"], "weight": 1}],
            "type_mix": {"bus": 1},
            "rate": 60,
        },
    ],
)
def test_e806_lane_too_short_for_a_routed_type(flow: dict[str, Any]) -> None:
    scenario = Scenario.from_dict(_chain(24.0, [flow]))  # K1_K2 lanes: 24 - 2 * 5.2 = 13.6 m
    assert _compile_error(scenario) == [
        (
            "E806",
            "network.roads[1].lanes[0]",
            'lane "K1_K2_0" is 13.6 m long but vehicle type "bus" routed through it needs '
            "14.5 m (length + min_gap); don't-block-the-box admission (F.3) could never admit "
            "it and insertion would place it past the lane end",
        )
    ]


def test_e806_trips_transit_and_short_types_pass() -> None:
    car = {"id": "c", "route": ["A_K1", "K1_K2", "K2_B"], "rate": 60}
    data = _chain(24.0, [car])
    data["demand"]["trips"] = [{"id": "t", "depart": 0, "origin": "A_K1", "destination": "K2_B"}]
    net = compile_network(Scenario.from_dict(data))  # cars need 7 m
    assert net.report.warnings == ()
    data["demand"]["transit"] = [
        {
            "id": "L",
            "route": ["A_K1", "K1_K2"],
            "headway": 600,
            "stops": [{"road": "A_K1", "position": 50}],
        }
    ]
    assert [c for c, *_ in _compile_error(Scenario.from_dict(data))] == ["E806"]


def test_errors_name_the_source(demo: Scenario) -> None:
    spec = ScenarioSpec.model_validate(_chain(14.0, []))
    with pytest.raises(ScenarioValidationError) as info:
        compile_network(Scenario(spec))
    assert info.value.source == "chain"
    with pytest.raises(ScenarioValidationError) as info:
        compile_network(Scenario(spec, path=demo.path))
    assert info.value.source == str(demo.path)
