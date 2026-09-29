"""ScenarioBuilder: fluent construction, local checks and edits (plan E.9)."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from urbanflow import generate
from urbanflow.core.errors import ScenarioValidationError
from urbanflow.core.types import IntersectionKind
from urbanflow.scenario import Scenario, ScenarioBuilder
from urbanflow.scenario.schema import FlowSpec, TransitLineSpec, TripSpec, VehicleTypeSpec


def grid_2x2() -> ScenarioBuilder:
    """The plan E.9 example, verbatim apart from the final save."""
    s = 300.0
    b = ScenarioBuilder("grid-2x2", duration=3600)
    for r in range(2):
        for c in range(2):
            b.intersection(f"J{r}{c}", (c * s, r * s))
        b.boundary(f"W{r}", (-s / 2, r * s)).boundary(f"E{r}", (1.5 * s, r * s))
    for c in range(2):
        b.boundary(f"S{c}", (c * s, -s / 2)).boundary(f"N{c}", (c * s, 1.5 * s))
    for r in range(2):
        b.two_way(f"W{r}", f"J{r}0", lanes=2).two_way(f"J{r}0", f"J{r}1", lanes=2)
        b.two_way(f"J{r}1", f"E{r}", lanes=2)
    for c in range(2):
        b.two_way(f"S{c}", f"J0{c}", lanes=2).two_way(f"J0{c}", f"J1{c}", lanes=2)
        b.two_way(f"J1{c}", f"N{c}", lanes=2)
    for r in range(2):
        b.flow(
            f"we{r}", origin=f"W{r}_J{r}0", destination=f"J{r}1_E{r}", rate=500, arrival="poisson"
        )
        b.flow(
            f"ew{r}", origin=f"E{r}_J{r}1", destination=f"J{r}0_W{r}", rate=500, arrival="poisson"
        )
    for c in range(2):
        b.flow(
            f"sn{c}", origin=f"S{c}_J0{c}", destination=f"J1{c}_N{c}", rate=400, arrival="poisson"
        )
    b.signal_all(template="two_phase", green=30)
    return b


def one_issue(exc: pytest.ExceptionInfo[ScenarioValidationError]) -> tuple[str, str]:
    (found,) = exc.value.issues
    return found.code, found.path


@pytest.fixture
def demo_builder(demo_data: dict[str, Any]) -> ScenarioBuilder:
    return Scenario.from_dict(demo_data).edit()


def test_grid_example_builds_cleanly(tmp_path: Any) -> None:
    scenario = grid_2x2().build()
    assert scenario.issues == ()
    counts = scenario.summary()
    assert (counts["intersections"], counts["roads"], counts["signals"]) == (12, 24, 4)
    assert all(
        p.duration == 30
        for ix in scenario.spec.network.intersections
        if ix.signal
        for p in ix.signal.phases
    )
    path = scenario.save(tmp_path / "grid2x2.json")
    assert Scenario.load(path).content_hash == scenario.content_hash


def test_round_trip_through_the_builder(demo_data: dict[str, Any]) -> None:
    for scenario in (
        Scenario.from_dict(demo_data),
        generate("single_intersection", kind="priority"),
    ):
        again = ScenarioBuilder.from_scenario(scenario).build()
        assert again.content_hash == scenario.content_hash
        assert again.spec == scenario.spec
        assert ScenarioBuilder.from_scenario(scenario.spec).to_spec() == scenario.spec


@pytest.mark.parametrize(
    ("call", "code", "path"),
    [
        (lambda b: b.intersection("J", (1, 1)), "E101", "id"),
        (lambda b: b.boundary("Q", (1, "x")), "E003", "point[1]"),
        (lambda b: b.road("N_in", "N", "J"), "E102", "id"),
        (lambda b: b.road("x", "N", "JJ"), "E103", "to"),
        (lambda b: b.road("x", "N", "J", lanes=0), "E004", "lanes"),
        (lambda b: b.road("x", "N", "J", lanes=[{"width": 9}]), "E004", "lanes[0].width"),
        (lambda b: b.road("x", "N", "J", lanez=2), "E002", "lanez"),
        (lambda b: b.movement("J", "N_in", "S_ot"), "E203", "to_road"),
        (lambda b: b.movement("JJ", "N_in", "S_out"), "E021", "intersection"),
        (lambda b: b.vehicle_type("city_bus"), "E401", "id"),
        (lambda b: b.vehicle_type("v", length=-1), "E004", "length"),
        (lambda b: b.flow("ns", route=["N_in"], rate=1), "E501", "id"),
        (lambda b: b.flow("f", route=["N_in"]), "E007", "$"),
        (lambda b: b.trip("bus1", 0, route=["N_in"]), "E501", "id"),
        (lambda b: b.simulation(dt=9), "E004", "dt"),
        (lambda b: b.update("flow", "ns", rate=-5), "E004", "rate"),
        (lambda b: b.update("road", "N_in", id="X"), "E024", "id"),
        (lambda b: b.remove("trip", "nope"), "E021", "id"),
        (lambda b: b.materialize("JJ"), "E021", "intersection"),
        (lambda b: b.vehicle_type("truck", speed_factor={"mean": -1}), "E004", "speed_factor.mean"),
    ],
)
def test_local_checks_raise_one_issue(
    demo_builder: ScenarioBuilder, call: Any, code: str, path: str
) -> None:
    with pytest.raises(ScenarioValidationError) as info:
        call(demo_builder)
    assert one_issue(info) == (code, path)


def test_movement_duplicates(demo_builder: ScenarioBuilder) -> None:
    demo_builder.movement("J", "N_in", "S_out", connections=[(0, 0)], turn="straight")
    with pytest.raises(ScenarioValidationError) as info:
        demo_builder.movement("J", "N_in", "S_out")
    assert one_issue(info) == ("E205", "to_road")
    with pytest.raises(ScenarioValidationError) as info:
        demo_builder.movement("J", "N_in", "E_out", id="N_in->S_out")
    assert one_issue(info) == ("E204", "id")
    assert "network.intersections[0].movements[0]" in info.value.issues[0].message


def test_two_way_is_atomic(demo_builder: ScenarioBuilder) -> None:
    with pytest.raises(ScenarioValidationError):
        demo_builder.two_way("N", "J", ids=("new_road", "N_in"))  # the second id exists
    assert "new_road" not in {r.id for r in demo_builder.to_spec().network.roads}


def test_materialize_copies_the_derived_form(demo_builder: ScenarioBuilder) -> None:
    derived = demo_builder.build().resolved.network.intersections[0].movements
    demo_builder.materialize("J")
    assert demo_builder.to_spec().network.intersections[0].movements == derived


def test_vehicle_types_and_lane_edits(demo_builder: ScenarioBuilder) -> None:
    demo_builder.update("road", "N_out", lanes=[{}, {}, {}], lane_width=3.5)
    demo_builder.vehicle_type("van", length=6.0, vclass="truck")
    demo_builder.flow("van_flow", route=["N_in", "S_out"], rate=30, vehicle_type="van")
    # a partial override of a built-in type: only the fields set replace the truck's values
    demo_builder.vehicle_type("truck", emergency_decel=1.8)
    scenario = demo_builder.build()
    assert scenario.issues == ()
    road = next(r for r in scenario.spec.network.roads if r.id == "N_out")
    assert (len(road.lanes), road.lane_width) == (3, 3.5)
    types = {vt.id: vt for vt in scenario.resolved.vehicle_types}
    assert types["van"].length == 6.0
    assert (types["truck"].decel, types["truck"].emergency_decel) == (1.5, 1.8)


def test_invalid_built_in_override_fails_at_build(demo_builder: ScenarioBuilder) -> None:
    demo_builder.vehicle_type("emergency", speed_factor={"max": 1.25})  # local checks pass
    with pytest.raises(ScenarioValidationError) as info:
        demo_builder.build()
    assert one_issue(info) == ("E007", "vehicle_types[2].speed_factor")


def test_editor_ops_are_not_part_of_p1() -> None:
    """``apply``/``OPS`` and the op-only methods arrive with scenario/ops.py (P14)."""
    assert not hasattr(ScenarioBuilder, "apply") and not hasattr(ScenarioBuilder, "OPS")


def test_remove_and_cascade(demo_builder: ScenarioBuilder) -> None:
    with pytest.raises(ScenarioValidationError) as info:
        demo_builder.remove("road", "N_in")
    message = info.value.issues[0].message
    assert 'flow "ns"' in message and "cascade=True" in message
    demo_builder.remove("road", "N_in", cascade=True)
    spec = demo_builder.to_spec()
    assert "N_in" not in {r.id for r in spec.network.roads}
    phases = spec.network.intersections[0].signal.phases  # type: ignore[union-attr]
    assert all(not m.startswith("N_in") for p in phases for m in p.green)
    with pytest.raises(ScenarioValidationError) as built:
        demo_builder.build()  # demand is never deleted: the dangling flow is reported
    assert ("E502", "demand.flows[0].route[0]") in {(i.code, i.path) for i in built.value.issues}
    demo_builder.remove("flow", "ns").remove("trip", "probe")
    demo_builder.remove("intersection", "E", cascade=True)
    assert {"E_in", "E_out"}.isdisjoint(r.id for r in demo_builder.to_spec().network.roads)


def test_move_and_update(demo_builder: ScenarioBuilder) -> None:
    demo_builder.move("N", (0, 300)).update("road", "N_in", name="Main Street")
    scenario = demo_builder.build()
    road = next(r for r in scenario.resolved.network.roads if r.id == "N_in")
    assert road.points == ((0.0, 300.0), (0.0, 0.0)) and road.name == "Main Street"
    demo_builder.update("road", "N_in", **{"from": "N"})
    assert demo_builder.to_spec().network.roads[0].from_ == "N"


def test_signal_and_kind(demo_builder: ScenarioBuilder) -> None:
    b = ScenarioBuilder("t")
    b.boundary("A", (-100, 0)).boundary("B", (100, 0)).intersection("K", (0, 0))
    b.road("A_K", "A", "K").road("K_B", "K", "B")
    b.signal("K", controller="actuated", gap=2.5, yellow=4)
    ix = b.to_spec().network.intersections[2]
    assert ix.kind is IntersectionKind.signalized and ix.signal is not None
    assert ix.signal.controller.type == "actuated" and dict(ix.signal.controller.params) == {
        "gap": 2.5
    }
    b.signal_all(green=20)  # K already has a signal
    assert b.to_spec().network.intersections[2].signal.phases is None  # type: ignore[union-attr]


def test_transit_line_stops(demo_builder: ScenarioBuilder) -> None:
    demo_builder.transit_line(
        "bus2",
        ["S_in", "N_out"],
        [("S_in", 50.0), {"road": "N_out", "position": 40}],
        headway=900,
        vehicle_type="city_bus",
        dwell=12,
    )
    line = demo_builder.to_spec().demand.transit[-1]
    assert [(s.road, s.dwell) for s in line.stops] == [("S_in", 12), ("N_out", 12)]
    assert demo_builder.build().issues == ()


def test_builder_arguments_use_schema_field_names() -> None:
    """Op payloads use the schema field names (B.2 #15): keep builder kwargs in sync."""
    pairs = [
        (ScenarioBuilder.flow, FlowSpec),
        (ScenarioBuilder.trip, TripSpec),
        (ScenarioBuilder.transit_line, TransitLineSpec),
    ]
    builder_only = {"self", "dwell"}
    for method, model in pairs:
        params = set(inspect.signature(method).parameters) - builder_only
        assert params <= set(model.model_fields), (
            method.__name__,
            params - set(model.model_fields),
        )
    assert "id" in inspect.signature(ScenarioBuilder.vehicle_type).parameters
    assert "id" in VehicleTypeSpec.model_fields
