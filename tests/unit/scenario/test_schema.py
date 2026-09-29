"""Scenario spec models and the exported JSON Schema (plan E.7 §1.6)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import BaseModel, ValidationError

from urbanflow.core import constants as C
from urbanflow.core.config import SimulationConfig
from urbanflow.scenario.io import bundled, bundled_names
from urbanflow.scenario.schema import (
    DEFAULT_VEHICLE_TYPES,
    EXCLUSIVE_MESSAGES,
    FlowSpec,
    LaneSpec,
    ScenarioSpec,
    SignalSpec,
    SimulationSpec,
    TransitLineSpec,
    TripSpec,
    VehicleTypeSpec,
    scenario_json_schema,
)

REPO = Path(__file__).resolve().parents[3]
SCHEMA_FILE = REPO / "docs" / "reference" / "scenario.schema.json"
ROUTE: dict[str, Any] = {"id": "f", "route": ["a"], "rate": 100}


def test_declared_form_round_trips(demo_data: dict[str, Any]) -> None:
    spec = ScenarioSpec.model_validate(demo_data)
    assert spec.model_dump(mode="json", exclude_unset=True) == demo_data
    assert ScenarioSpec.model_validate_json(spec.model_dump_json()) == spec


def test_models_are_frozen(demo_data: dict[str, Any]) -> None:
    spec = ScenarioSpec.model_validate(demo_data)
    with pytest.raises(ValidationError):
        spec.meta.name = "changed"  # type: ignore[misc]
    assert isinstance(spec.network.roads, tuple)


def test_road_from_alias_and_schema_hint(demo_data: dict[str, Any]) -> None:
    demo_data["$schema"] = "./scenario.schema.json"
    spec = ScenarioSpec.model_validate(demo_data)
    assert spec.network.roads[0].from_ == "N"
    dumped = spec.model_dump(mode="json")
    assert dumped["network"]["roads"][0]["from"] == "N" and dumped["$schema"]
    del demo_data["$schema"]
    assert "$schema" not in ScenarioSpec.model_validate(demo_data).model_dump(mode="json")


@pytest.mark.parametrize(
    ("model", "data", "rule"),
    [
        (FlowSpec, {"id": "f", "route": ["a"]}, "rate_period"),
        (FlowSpec, {**ROUTE, "period": 5}, "rate_period"),
        (FlowSpec, {"id": "f", "rate": 1}, "flow_route"),
        (FlowSpec, {**ROUTE, "origin": "a", "destination": "b"}, "flow_route"),
        (FlowSpec, {"id": "f", "rate": 1, "origin": "a"}, "flow_route"),
        (FlowSpec, {**ROUTE, "via": ["b"]}, "via"),
        (FlowSpec, {**ROUTE, "vehicle_type": "bus", "type_mix": {"car": 1}}, "type_mix"),
        (FlowSpec, {**ROUTE, "begin": 10, "end": 5}, "end_begin"),
        (TripSpec, {"id": "t", "depart": 0}, "trip_route"),
        (TripSpec, {"id": "t", "depart": 0, "route": ["a"], "via": ["b"]}, "via"),
        (
            TransitLineSpec,
            {"id": "t", "route": ["a"], "stops": [{"road": "a", "position": 20}]},
            "headway",
        ),
        (
            TransitLineSpec,
            {
                "id": "t",
                "route": ["a"],
                "stops": [{"road": "a", "position": 20}],
                "departures": [5, 1],
            },
            "departures",
        ),
        (SignalSpec, {"min_green": 10, "max_green": 10}, "green"),
        (VehicleTypeSpec, {"id": "v", "decel": 5, "emergency_decel": 4}, "decel"),
        (VehicleTypeSpec, {"id": "v", "speed_factor": {"mean": 2}}, "speed_factor"),
        (
            VehicleTypeSpec,
            {"id": "car", "speed_factor": {"mean": 2, "min": 0.5, "max": 1.5}},
            "speed_factor",
        ),
        (VehicleTypeSpec, {"id": "bus", "decel": 5, "emergency_decel": 4}, "decel"),
    ],
)
def test_cross_field_rules(model: type[BaseModel], data: dict[str, Any], rule: str) -> None:
    with pytest.raises(ValidationError) as info:
        model.model_validate(data)
    (err,) = info.value.errors()
    assert err["type"] == "uf_exclusive"
    assert err["msg"] == EXCLUSIVE_MESSAGES[rule].format(**err.get("ctx", {}))


def test_speed_factor_error_points_at_the_field() -> None:
    with pytest.raises(ValidationError) as info:
        VehicleTypeSpec.model_validate({"id": "v", "speed_factor": {"min": 1.5}})
    assert [e["loc"] for e in info.value.errors()] == [("speed_factor",)]


@pytest.mark.parametrize(
    "data",
    [
        {"id": "truck", "emergency_decel": 1.8},  # truck decel is 1.5 (car default 2.0)
        {"id": "emergency", "speed_factor": {"mean": 1.3}},  # emergency min = max = 1.3
        {"id": "emergency", "speed_factor": {"max": 1.25}},  # invalid once merged (P stage)
        {"id": "car", "decel": 9.5},  # car emergency_decel is 6.0: invalid once merged
    ],
)
def test_partial_built_in_overrides_defer_cross_field_rules(data: dict[str, Any]) -> None:
    """Only the fields you set override (E.7): the rules run on the merged type instead."""
    assert VehicleTypeSpec.model_validate(data).id == data["id"]


def test_non_finite_numbers_are_rejected() -> None:
    with pytest.raises(ValidationError) as info:
        LaneSpec(width=math.nan)
    assert info.value.errors()[0]["type"] == "finite_number"


def test_builtin_vehicle_types_follow_the_constants() -> None:
    assert set(DEFAULT_VEHICLE_TYPES) == {"car", "bus", "truck", "emergency"}
    assert DEFAULT_VEHICLE_TYPES["car"] == VehicleTypeSpec(id="car")
    bus = DEFAULT_VEHICLE_TYPES["bus"]
    assert (bus.length, bus.width, bus.max_speed, bus.speed_factor.std) == (12.0, 2.55, 25.0, 0.05)
    emergency = DEFAULT_VEHICLE_TYPES["emergency"].speed_factor
    assert (emergency.mean, emergency.std, emergency.min, emergency.max) == (1.3, 0.0, 1.3, 1.3)
    assert VehicleTypeSpec(id="x").length == C.VEHICLE_LENGTH


def test_simulation_block_mirrors_the_run_config() -> None:
    config = SimulationConfig.model_fields
    for name, info in SimulationSpec.model_fields.items():
        assert name in config, name
        assert info.default == config[name].default, name
        assert info.metadata == config[name].metadata, name
    for excluded in ("metrics", "record", "debug_checks", "accel"):
        assert excluded not in SimulationSpec.model_fields


def test_committed_schema_is_current() -> None:
    committed = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert committed == scenario_json_schema(), "run: uv run python scripts/gen_scenario_docs.py"


def test_schema_header() -> None:
    schema = scenario_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "urn:urbanflow:scenario:1.0"
    assert "title" in schema and schema["properties"]["network"]


def test_jsonschema_accepts_valid_scenarios(demo_data: dict[str, Any]) -> None:
    validator = jsonschema.Draft202012Validator(scenario_json_schema())
    validator.validate(demo_data)
    assert bundled_names()
    for name in bundled_names():
        validator.validate(json.loads(bundled(name).read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("where", "value"),
    [
        (("network", "roads", 0, "lanes", 0, "width"), 9),
        (("network", "roads", 0, "id"), "bad id"),
        (("network", "drive_side"), "middle"),
        (("meta", "surprise"), 1),
        (("simulation", "dt"), 5),
        (("network", "intersections", 0, "point"), [1, 2, 3]),
    ],
)
def test_jsonschema_and_pydantic_agree_on_rejections(
    demo_data: dict[str, Any], where: tuple[str | int, ...], value: Any
) -> None:
    node: Any = demo_data
    for key in where[:-1]:
        node = node[key]
    node[where[-1]] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(demo_data, scenario_json_schema())
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate(demo_data)
