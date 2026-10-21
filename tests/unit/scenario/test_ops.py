"""Editor operations on scenario drafts (plan P): defaults, atomicity, error paths."""

from __future__ import annotations

from typing import Any

import pytest

from urbanflow.core.errors import ScenarioValidationError
from urbanflow.scenario import Scenario
from urbanflow.scenario.ops import apply_ops
from urbanflow.scenario.validate import validate_data

BLANK: dict[str, Any] = {
    "format": "urbanflow.scenario",
    "version": "1.0",
    "meta": {"name": "drawn"},
    "network": {"intersections": [], "roads": []},
}


def _ids(data: dict[str, Any], kind: str) -> list[str]:
    return [x["id"] for x in data["network"][kind]]


def _draw() -> dict[str, Any]:
    """W -- J -- E, two lanes each way, traffic from the west edge to the east edge."""
    ops = [
        {"op": "add_intersection", "point": [-200, 0]},
        {"op": "add_intersection", "point": [0, 0]},
        {"op": "add_intersection", "point": [200, 0]},
        {"op": "add_road", "from": "J1", "to": "J2", "lanes": 2},
        {"op": "add_road", "from": "J2", "to": "J3", "lanes": 2},
        {"op": "add_flow", "origin": "J1_J2", "destination": "J2_J3", "rate": 300},
    ]
    return apply_ops(BLANK, ops).data


def test_a_drawn_map_becomes_a_valid_scenario() -> None:
    result = apply_ops(BLANK, [{"op": "add_intersection", "point": [0, 0]}])
    assert result.created == [{"op_index": 0, "kind": "intersection", "id": "J1"}]
    data = _draw()
    assert _ids(data, "intersections") == ["J1", "J2", "J3"]
    assert _ids(data, "roads") == ["J1_J2", "J2_J1", "J2_J3", "J3_J2"]
    assert data["demand"]["flows"][0]["id"] == "flow1"
    report = validate_data(data)
    assert report.ok, [(i.path, i.message) for i in report.errors]
    kinds = {ix.id: ix.kind.value for ix in report.resolved.network.intersections}
    assert kinds == {"J1": "boundary", "J2": "uncontrolled", "J3": "boundary"}
    movements = report.resolved.network.intersections[1].movements or ()
    assert {(m.from_road, m.to_road) for m in movements} == {("J1_J2", "J2_J3"), ("J3_J2", "J2_J1")}
    assert Scenario.from_dict(data).summary()["roads"] == 4


def test_a_blank_draft_is_editable_but_not_yet_valid() -> None:
    assert apply_ops(BLANK, []).data["network"] == {"intersections": [], "roads": []}
    paths = {i.path for i in validate_data(BLANK).errors}
    assert "network.roads" in paths


def test_road_ids_never_collide() -> None:
    data = apply_ops(
        _draw(), [{"op": "add_road", "from": "J1", "to": "J2", "bidirectional": False}]
    )
    assert _ids(data.data, "roads")[-1] == "J1_J2.2"


def test_signals_kinds_and_timing() -> None:
    data = apply_ops(
        _draw(),
        [
            {"op": "set_controller", "intersection": "J2", "controller": "max_pressure"},
            {"op": "set_signal_timing", "intersection": "J2", "green": 20, "yellow": 4},
        ],
    ).data
    ix = data["network"]["intersections"][1]
    assert ix["kind"] == "signalized" and ix["signal"]["controller"]["type"] == "max_pressure"
    assert ix["signal"]["yellow"] == 4
    assert {p["duration"] for p in ix["signal"]["phases"]} == {20}
    assert validate_data(data).ok
    priority = apply_ops(data, [{"op": "update_intersection", "id": "J2", "kind": "priority"}])
    ix = priority.data["network"]["intersections"][1]
    assert ix["kind"] == "priority" and "signal" not in ix
    auto = apply_ops(data, [{"op": "update_intersection", "id": "J2", "kind": "auto"}])
    assert "kind" not in auto.data["network"]["intersections"][1]


def test_edits_and_deletions() -> None:
    data = apply_ops(
        _draw(),
        [
            {"op": "move_intersection", "id": "J3", "point": [250, 30]},
            {"op": "update_road", "id": "J1_J2", "lanes": 3, "speed_limit": 20},
            {"op": "update_flow", "id": "flow1", "rate": 450},
            {"op": "set_metadata", "name": "renamed"},
            {"op": "set_simulation", "duration": 600},
        ],
    ).data
    assert data["network"]["intersections"][2]["point"] == [250, 30]
    road = data["network"]["roads"][0]
    assert len(road["lanes"]) == 3 and road["speed_limit"] == 20
    assert data["demand"]["flows"][0]["rate"] == 450
    assert data["meta"]["name"] == "renamed" and data["simulation"]["duration"] == 600
    gone = apply_ops(data, [{"op": "delete_intersection", "id": "J3"}]).data
    assert _ids(gone, "roads") == ["J1_J2", "J2_J1"]  # its roads went too
    codes = {(i.code, i.path) for i in validate_data(gone).errors}
    assert ("E502", "demand.flows[0].destination") in codes  # the flow is reported, not deleted
    cut = apply_ops(
        data, [{"op": "delete_road", "id": "J2_J3"}, {"op": "delete_flow", "id": "flow1"}]
    )
    assert "J2_J3" not in _ids(cut.data, "roads") and "demand" not in cut.data


@pytest.mark.parametrize(
    ("ops", "path"),
    [
        ([{"op": "teleport"}], "ops[0]"),
        ([{"op": "add_road", "from": "J1", "to": "J2", "lanes": 0}], "ops[0].lanes"),
        ([{"op": "add_intersection", "point": [0, 0]},
          {"op": "add_road", "from": "J9", "to": "J1"}], "ops[1].from"),
        ([{"op": "delete_road", "id": "nope"}], "ops[0].id"),
    ],
)  # fmt: skip
def test_errors_point_at_the_operation(ops: list[dict[str, Any]], path: str) -> None:
    before = _draw()
    with pytest.raises(ScenarioValidationError) as info:
        apply_ops(before, ops)
    assert info.value.issues[0].path.startswith(path), info.value.issues
    assert before == _draw()  # nothing changed: operations are all or nothing


def test_draft_entities_are_checked_with_document_paths() -> None:
    bad = {
        **BLANK,
        "network": {"intersections": [{"id": "A", "point": [0, 0]}], "roads": [{"id": "r"}]},
    }
    with pytest.raises(ScenarioValidationError) as info:
        apply_ops(bad, [])
    assert info.value.issues[0].path.startswith("network.roads[0]")
