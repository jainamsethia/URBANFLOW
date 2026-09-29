"""Array aliases and the enums shared by every layer.

Enum *values* are stable: they appear in scenario files, frames, replays and the
WebSocket protocol. Every enum also has a stable small-integer ``code`` (its definition
order), which is what the compiled network and the runtime arrays store (``u8`` columns).
Reordering members is therefore a format change; ``tests/unit/core/test_types.py`` pins them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SIGNAL_CODE_UNSIGNALISED",
    "BoolArray",
    "DriveSide",
    "FloatArray",
    "Handle",
    "IntArray",
    "IntersectionKind",
    "LinkKind",
    "SignalState",
    "Stage",
    "TurnKind",
    "UIntArray",
    "VehicleClass",
    "VehicleStatus",
]

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.signedinteger[Any]]
UIntArray = NDArray[np.unsignedinteger[Any]]
BoolArray = NDArray[np.bool_]
Handle = int
"""A reusable VehicleTable slot. Never leaves L4/L5 internals; external identity is the uid."""


class _CodedEnum(StrEnum):
    """A ``StrEnum`` whose members also carry a stable ``u8`` code (definition order)."""

    @property
    def code(self) -> int:
        """The integer stored in compiled/runtime arrays for this member."""
        return list(type(self)).index(self)

    @classmethod
    def from_code(cls, code: int) -> Self:
        """The member stored as ``code``; ``ValueError`` if no member has it."""
        members = list(cls)
        if not 0 <= code < len(members):
            raise ValueError(f"{code} is not a valid {cls.__name__} code (0-{len(members) - 1})")
        return members[code]


class TurnKind(_CodedEnum):
    """Turn of a movement (``mov_turn``)."""

    straight = "straight"
    left = "left"
    right = "right"
    uturn = "uturn"


class SignalState(_CodedEnum):
    """Signal state of a movement. Codes are the frame/replay codes: r 0, y 1, g 2, G 3.

    ``G`` is protected green, ``g`` permissive green (yield to higher rank).
    Unsignalised movements use :data:`SIGNAL_CODE_UNSIGNALISED` (255) in frames.
    """

    r = "r"
    y = "y"
    g = "g"
    G = "G"


SIGNAL_CODE_UNSIGNALISED: Final = 255
"""Frame/replay signal code of a movement at a non-signalised intersection (K.10.1)."""


class VehicleStatus(_CodedEnum):
    """Lifecycle status of a vehicle (``VehicleTable.status``)."""

    pending = "pending"
    waiting_insert = "waiting_insert"
    running = "running"
    arrived = "arrived"
    removed = "removed"


class LinkKind(_CodedEnum):
    """Kind of a compiled link (``link_kind``): 0 lane, 1 connector."""

    lane = "lane"
    connector = "connector"


class IntersectionKind(_CodedEnum):
    """Control kind of an intersection (``int_kind``)."""

    signalized = "signalized"
    priority = "priority"
    uncontrolled = "uncontrolled"
    boundary = "boundary"


class Stage(_CodedEnum):
    """Signal runtime stage (``SignalSnapshot.stage``): 0 green, 1 yellow, 2 all-red."""

    green = "green"
    yellow = "yellow"
    all_red = "all_red"


class DriveSide(_CodedEnum):
    """Side of the road vehicles drive on; lanes stack from the median toward this side."""

    right = "right"
    left = "left"


class VehicleClass(_CodedEnum):
    """Vehicle class (``VehicleTypeSpec.vclass``); selects energy and palette defaults."""

    car = "car"
    bus = "bus"
    truck = "truck"
    emergency = "emergency"
