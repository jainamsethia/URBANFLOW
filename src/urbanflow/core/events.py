"""Engine events: the canonical :class:`EventType` list and the columnar :class:`EventBuffer`.

The buffer holds the events of the *current* step (B.2 #12, F.8): the engine clears it at
step start and appends with vectorised :meth:`EventBuffer.append_many`; metrics, the replay
recorder and the server read the columns directly. Vehicle identity outside L4/L5 is the
``uid`` column (B.2 #1); ``handle`` is for engine internals only.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from urbanflow.core.types import _CodedEnum

__all__ = ["COLUMNS", "NO_VEHICLE", "EventBuffer", "EventType"]

NO_VEHICLE: Final = 0xFFFF_FFFF
"""``handle``/``uid`` of events that are not about a vehicle (u32 maximum)."""


class EventType(_CodedEnum):
    """Canonical event types (B.2 #12). The ``code`` stored in the ``type`` column is stable."""

    vehicle_departed = "vehicle_departed"
    vehicle_inserted = "vehicle_inserted"
    vehicle_entered_link = "vehicle_entered_link"
    vehicle_exited_link = "vehicle_exited_link"
    vehicle_stopped = "vehicle_stopped"
    vehicle_resumed = "vehicle_resumed"
    vehicle_changed_lane = "vehicle_changed_lane"
    vehicle_arrived = "vehicle_arrived"
    vehicle_removed = "vehicle_removed"
    vehicle_teleported = "vehicle_teleported"
    phase_changed = "phase_changed"
    simulation_started = "simulation_started"
    simulation_ended = "simulation_ended"


COLUMNS: Final[tuple[tuple[str, DTypeLike], ...]] = (
    ("type", np.uint8),
    ("step", np.int64),
    ("time", np.float64),
    ("handle", np.uint32),
    ("uid", np.uint32),
    ("link", np.int32),
    ("intersection", np.int32),
    ("aux", np.int32),
)
"""Column names and dtypes, in record order (B.2 #12)."""


class EventBuffer:
    """Columnar, amortised-growth event storage for one step (cleared by the engine).

    Columns: ``type u8, step i64, time f64, handle u32, uid u32, link i32, intersection
    i32, aux i32``. Properties return read-only views of the filled rows; they stay valid
    until the next append or :meth:`clear`.
    """

    def __init__(self, capacity: int = 64) -> None:
        self._n = 0
        self._cols: dict[str, NDArray[Any]] = {
            name: np.zeros(max(1, capacity), dtype=dtype) for name, dtype in COLUMNS
        }

    def __len__(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        return len(self._cols["type"])

    def clear(self) -> None:
        """Drop every event (storage is kept)."""
        self._n = 0

    def append(
        self,
        type: EventType,
        step: int,
        time: float,
        *,
        handle: int = NO_VEHICLE,
        uid: int = NO_VEHICLE,
        link: int = -1,
        intersection: int = -1,
        aux: int = 0,
    ) -> None:
        """Append one event."""
        self.append_many(
            type,
            step,
            time,
            handles=handle,
            uids=uid,
            links=link,
            intersections=intersection,
            aux=aux,
        )

    def append_many(
        self,
        type: EventType,
        step: int,
        time: ArrayLike,
        *,
        handles: ArrayLike = NO_VEHICLE,
        uids: ArrayLike = NO_VEHICLE,
        links: ArrayLike = -1,
        intersections: ArrayLike = -1,
        aux: ArrayLike = 0,
    ) -> None:
        """Append ``n`` events of one type; array arguments (length n) and scalars broadcast.

        ``n`` is the broadcast length of ``time`` and the keyword columns (all scalars: 1).
        """
        values = {
            "time": np.asarray(time),
            "handle": np.asarray(handles),
            "uid": np.asarray(uids),
            "link": np.asarray(links),
            "intersection": np.asarray(intersections),
            "aux": np.asarray(aux),
        }
        shape = np.broadcast_shapes(*(v.shape for v in values.values()))
        if len(shape) > 1:
            raise ValueError(f"event columns must be scalars or 1-D, got shape {shape}")
        n = shape[0] if shape else 1
        if n == 0:
            return
        start, stop = self._n, self._n + n
        if stop > self.capacity:
            self._grow(stop)
        cols = self._cols
        cols["type"][start:stop] = type.code
        cols["step"][start:stop] = step
        for name, value in values.items():
            cols[name][start:stop] = value
        self._n = stop

    def _grow(self, needed: int) -> None:
        new = max(needed, 2 * self.capacity)
        for name, old in self._cols.items():
            arr = np.zeros(new, dtype=old.dtype)
            arr[: self._n] = old[: self._n]
            self._cols[name] = arr

    # ------------------------------------------------------------------ read-only columns
    def _col(self, name: str) -> NDArray[Any]:
        view = self._cols[name][: self._n]
        view.flags.writeable = False
        return view

    def view(self) -> dict[str, NDArray[Any]]:
        """Every column (read-only views of the filled rows) in record order."""
        return {name: self._col(name) for name, _ in COLUMNS}

    @property
    def type(self) -> NDArray[np.uint8]:
        """``EventType`` codes."""
        return self._col("type")

    @property
    def step(self) -> NDArray[np.int64]:
        return self._col("step")

    @property
    def time(self) -> NDArray[np.float64]:
        """Event time, s (``vehicle_arrived`` is interpolated within the step)."""
        return self._col("time")

    @property
    def handle(self) -> NDArray[np.uint32]:
        """Engine slot (internal; ``NO_VEHICLE`` for non-vehicle events)."""
        return self._col("handle")

    @property
    def uid(self) -> NDArray[np.uint32]:
        """Vehicle uid (``NO_VEHICLE`` for non-vehicle events)."""
        return self._col("uid")

    @property
    def link(self) -> NDArray[np.int32]:
        return self._col("link")

    @property
    def intersection(self) -> NDArray[np.int32]:
        return self._col("intersection")

    @property
    def aux(self) -> NDArray[np.int32]:
        """Type-specific payload (e.g. the new phase of ``phase_changed``)."""
        return self._col("aux")
