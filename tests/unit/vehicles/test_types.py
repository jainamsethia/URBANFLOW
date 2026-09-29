"""Resolved vehicle types as parameter arrays (plan E.7 VehicleType, G.2, G.6, G.7)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from urbanflow.core import constants as C
from urbanflow.core.errors import ConfigError, NotFoundError
from urbanflow.core.types import VehicleClass
from urbanflow.scenario import Scenario
from urbanflow.vehicles import IDM, ParamArrays, VehicleTypes

Doc = Callable[..., dict[str, Any]]


@pytest.fixture
def make(corridor_doc: Doc) -> Callable[..., VehicleTypes]:
    def build(
        vehicle_types: list[dict[str, Any]], params: type[BaseModel] | None = IDM.Params
    ) -> VehicleTypes:
        data = corridor_doc()
        data["vehicle_types"] = vehicle_types
        specs = Scenario.from_dict(data).resolved.vehicle_types
        return VehicleTypes.from_specs(specs, params)

    return build


def test_builtins_resolved_and_overridden(make: Callable[..., VehicleTypes]) -> None:
    types = make([{"id": "car", "length": 4.5}, {"id": "van", "length": 6.0}])
    assert types.ids == ("bus", "car", "emergency", "truck", "van")  # sorted, built-ins merged
    assert types.index["van"] == 4
    car, bus = types.index["car"], types.index["bus"]
    assert types.length[car] == 4.5
    assert types.accel[car] == C.IDM_ACCEL  # untouched fields keep the built-in values
    assert (types.length[bus], types.max_speed[bus], types.decel[bus]) == (12.0, 25.0, 1.5)
    assert types.vclass[bus] == VehicleClass.bus.code
    assert types.vclass[types.index["van"]] == VehicleClass.car.code
    assert not types.length.flags.writeable


def test_b_hat_is_the_fleet_minimum(make: Callable[..., VehicleTypes]) -> None:
    types = make([{"id": "soft", "decel": 1.0, "emergency_decel": 4.0}])
    b = types.emergency_decel
    assert np.array_equal(types.b_hat, np.minimum(b, b.min()))
    assert set(types.b_hat.tolist()) == {4.0}  # G.2: b_hat <= b_L for every possible leader
    assert types.emergency_decel[types.index["car"]] == C.IDM_EMERGENCY_DECEL


def test_param_arrays_and_gather(make: Callable[..., VehicleTypes]) -> None:
    types = make([{"id": "car", "model_params": {"delta": 2.0}}])
    p = types.params
    assert isinstance(p, ParamArrays)
    assert {"a", "b", "T", "s0", "delta", "b_emerg", "b_hat"} <= set(p.names())
    car, truck = types.index["car"], types.index["truck"]
    assert p.delta[car] == 2.0
    assert p.delta[truck] == C.IDM_DELTA  # the Params default
    assert np.array_equal(p.a, types.accel)
    assert np.array_equal(p.T, types.headway)
    per_vehicle = p.gather(np.array([truck, car, car]))
    assert len(per_vehicle) == 3
    assert per_vehicle.a.tolist() == [0.8, C.IDM_ACCEL, C.IDM_ACCEL]
    assert per_vehicle.s0.tolist() == [2.5, C.IDM_MIN_GAP, C.IDM_MIN_GAP]
    with pytest.raises(AttributeError, match='"delta"'):
        _ = p.deltaa


def test_custom_model_params_become_columns(make: Callable[..., VehicleTypes]) -> None:
    class OVMParams(BaseModel):
        sensitivity: float = 0.8
        d_c: float = 20.0

    types = make([{"id": "car", "model_params": {"d_c": 15.0}}], OVMParams)
    car = types.index["car"]
    assert types.params.d_c[car] == 15.0
    assert types.params.sensitivity[car] == 0.8
    assert types.params.delta[car] == C.IDM_DELTA  # always present (IDM default)


def test_unknown_model_param_is_e903(make: Callable[..., VehicleTypes]) -> None:
    with pytest.raises(ConfigError) as err:
        make([{"id": "car", "model_params": {"delta": 4.0, "gamma": 1.0}}])
    (issue,) = err.value.issues
    assert issue.code == "E903"
    assert issue.path == "vehicle_types.car.model_params.gamma"
    assert '"gamma"' in issue.message
    assert '"idm"' in issue.message
    assert "delta" in issue.message


def test_invalid_model_param_value(make: Callable[..., VehicleTypes]) -> None:
    with pytest.raises(ConfigError) as err:
        make([{"id": "car", "model_params": {"delta": -1.0}}])
    (issue,) = err.value.issues
    assert issue.path == "vehicle_types.car.model_params.delta"
    assert issue.code == "E004"


def test_type_index_lookup(make: Callable[..., VehicleTypes]) -> None:
    types = make([])
    assert types.type_index("truck") == types.index["truck"]
    with pytest.raises(NotFoundError, match='did you mean "truck"'):
        types.type_index("truk")
