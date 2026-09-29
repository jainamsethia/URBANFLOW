"""Enum values and codes are part of the file/frame formats: pin them."""

from __future__ import annotations

import pytest

from urbanflow.core.types import (
    SIGNAL_CODE_UNSIGNALISED,
    DriveSide,
    IntersectionKind,
    LinkKind,
    SignalState,
    Stage,
    TurnKind,
    VehicleClass,
    VehicleStatus,
)

PINNED = [
    (TurnKind, ["straight", "left", "right", "uturn"]),
    (SignalState, ["r", "y", "g", "G"]),  # codes r=0, y=1, g=2, G=3 (K.10, H.2)
    (VehicleStatus, ["pending", "waiting_insert", "running", "arrived", "removed"]),
    (LinkKind, ["lane", "connector"]),
    (IntersectionKind, ["signalized", "priority", "uncontrolled", "boundary"]),
    (Stage, ["green", "yellow", "all_red"]),
    (DriveSide, ["right", "left"]),
    (VehicleClass, ["car", "bus", "truck", "emergency"]),
]


@pytest.mark.parametrize(("enum", "values"), PINNED, ids=lambda x: getattr(x, "__name__", ""))
def test_values_and_codes_are_stable(enum: type, values: list[str]) -> None:
    members = list(enum)
    assert [m.value for m in members] == values
    assert [m.code for m in members] == list(range(len(values)))
    assert all(enum.from_code(m.code) is m for m in members)
    assert all(enum(v) == v for v in values)  # StrEnum: compares equal to its value


def test_signal_codes_match_frame_format() -> None:
    codes = {s.value: s.code for s in SignalState}
    assert codes == {"r": 0, "y": 1, "g": 2, "G": 3}
    assert SIGNAL_CODE_UNSIGNALISED == 255
    assert SignalState.g != SignalState.G  # case matters


def test_link_kind_codes() -> None:
    assert (LinkKind.lane.code, LinkKind.connector.code) == (0, 1)


@pytest.mark.parametrize("code", [-1, 4, 255])
def test_from_code_rejects_unknown(code: int) -> None:
    with pytest.raises(ValueError, match="SignalState"):
        SignalState.from_code(code)
