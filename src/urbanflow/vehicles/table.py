"""The vehicle table: structure-of-arrays storage with reusable slots (plan E.4, B.2 #1).

A vehicle lives in a *handle* (row). Handles are internal to L4/L5; everything external
uses the ``uid``, a monotonic spawn serial that is never reused within a run. Freed handles
go to a deferred list and become reusable only after the next :meth:`VehicleTable.recycle`
(engine sub-step 0); handles freed *between* steps (commands) wait one recycle longer,
because the next step still reports their removal. So a handle never names two vehicles
within one step's events. The free list is LIFO, which keeps handle assignment
deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
from numpy.typing import DTypeLike, NDArray

from urbanflow.core.errors import CommandError, NotFoundError, suggest
from urbanflow.core.types import IntArray, VehicleStatus

__all__ = ["COLUMNS", "DEPART_LANE_BEST", "DEPART_LANE_FIRST", "VehicleTable"]

DEPART_LANE_BEST: Final = -1
"""``depart_lane`` code of ``"best"`` (most free space, then lowest index; G.8)."""
DEPART_LANE_FIRST: Final = -2
"""``depart_lane`` code of ``"first"`` (lowest valid lane). ``"random"`` is stored as the
lane index drawn at spawn, integers as themselves."""

_PENDING = VehicleStatus.pending.code

# name -> (dtype, fill of a fresh row). E.4 order; `depart_lane`/`depart_speed` hold the
# G.8 insertion request of a waiting vehicle (so it is part of the snapshot and digest).
COLUMNS: Final[dict[str, tuple[DTypeLike, Any]]] = {
    "status": (np.uint8, _PENDING),
    "active": (np.bool_, False),
    "uid": (np.uint32, 0),
    "type_idx": (np.uint16, 0),
    "source_idx": (np.int32, -1),
    "link": (np.int32, -1),
    "pos": (np.float64, 0.0),
    "speed": (np.float64, 0.0),
    "accel": (np.float64, 0.0),
    "length": (np.float32, 0.0),
    "width": (np.float32, 0.0),
    "speed_factor": (np.float32, 1.0),
    "v0": (np.float64, 0.0),
    "route_id": (np.int32, -1),
    "route_cursor": (np.int16, 0),
    "valid_mask": (np.uint16, 0),
    "next_conn": (np.int32, -1),
    "committed": (np.bool_, False),
    "granted": (np.bool_, False),
    "lock_conn": (np.int32, -1),
    "forced": (np.bool_, False),
    "commit_seq": (np.int64, -1),
    "held": (np.bool_, False),
    "lat_offset": (np.float32, 0.0),
    "lc_cooldown": (np.float32, 0.0),
    "lc_target": (np.int16, -1),
    "lc_target_until": (np.float64, 0.0),
    "speed_override": (np.float64, math.nan),
    "override_until": (np.float64, math.inf),
    "depart_time": (np.float64, math.nan),
    "insert_time": (np.float64, math.nan),
    "waiting_time": (np.float64, 0.0),
    "link_waiting_time": (np.float32, 0.0),
    "stops": (np.int32, 0),
    "halting": (np.bool_, False),
    "distance": (np.float64, 0.0),
    "ff_time": (np.float64, 0.0),
    "lane_changes": (np.int16, 0),
    "teleports": (np.int16, 0),
    "energy": (np.float64, 0.0),
    "stuck_time": (np.float32, 0.0),
    "stop_idx": (np.int16, -1),
    "dwell_left": (np.float32, 0.0),
    "depart_lane": (np.int16, DEPART_LANE_BEST),
    "depart_speed": (np.float64, math.nan),
}
"""Every per-vehicle column: dtype and the value of a freshly allocated row."""


class VehicleTable:
    """SoA vehicle store (E.4): one numpy column per field, rows are handles.

    Columns are plain attributes (``table.pos[h]``). They are replaced on growth, so do not
    keep references across :meth:`alloc` calls. Only rows below :attr:`top` (the
    high-water mark) have ever been used.
    """

    status: NDArray[np.uint8]
    """``VehicleStatus`` code."""
    active: NDArray[np.bool_]
    """``status == running``."""
    uid: NDArray[np.uint32]
    type_idx: NDArray[np.uint16]
    source_idx: NDArray[np.int32]
    """Flow, trip or transit source index (flows, then trips, then transit); -1 for API."""
    link: NDArray[np.int32]
    pos: NDArray[np.float64]
    """Front-bumper position along the link, m."""
    speed: NDArray[np.float64]
    accel: NDArray[np.float64]
    length: NDArray[np.float32]
    width: NDArray[np.float32]
    speed_factor: NDArray[np.float32]
    v0: NDArray[np.float64]
    """Desired speed on the current link, m/s."""
    route_id: NDArray[np.int32]
    route_cursor: NDArray[np.int16]
    valid_mask: NDArray[np.uint16]
    next_conn: NDArray[np.int32]
    committed: NDArray[np.bool_]
    granted: NDArray[np.bool_]
    lock_conn: NDArray[np.int32]
    forced: NDArray[np.bool_]
    commit_seq: NDArray[np.int64]
    held: NDArray[np.bool_]
    lat_offset: NDArray[np.float32]
    lc_cooldown: NDArray[np.float32]
    lc_target: NDArray[np.int16]
    lc_target_until: NDArray[np.float64]
    speed_override: NDArray[np.float64]
    """NaN if none."""
    override_until: NDArray[np.float64]
    depart_time: NDArray[np.float64]
    insert_time: NDArray[np.float64]
    waiting_time: NDArray[np.float64]
    link_waiting_time: NDArray[np.float32]
    stops: NDArray[np.int32]
    halting: NDArray[np.bool_]
    distance: NDArray[np.float64]
    ff_time: NDArray[np.float64]
    lane_changes: NDArray[np.int16]
    teleports: NDArray[np.int16]
    energy: NDArray[np.float64]
    stuck_time: NDArray[np.float32]
    stop_idx: NDArray[np.int16]
    dwell_left: NDArray[np.float32]
    depart_lane: NDArray[np.int16]
    """Requested depart lane: ``DEPART_LANE_BEST``, ``DEPART_LANE_FIRST`` or a lane index."""
    depart_speed: NDArray[np.float64]
    """Requested depart speed, m/s; NaN = ``"max"``."""

    def __init__(self, capacity: int = 64) -> None:
        self._capacity = max(1, capacity)
        for name, (dtype, fill) in COLUMNS.items():
            setattr(self, name, np.full(self._capacity, fill, dtype=dtype))
        self.ids: list[str | None] = [None] * self._capacity
        """Vehicle id by handle (None for never-used or recycled rows)."""
        self.id_to_handle: dict[str, int] = {}
        """Live (not yet freed) vehicle ids."""
        self.uid_to_id: list[str] = []
        """Vehicle id by uid, append-only within a run (frames and replays)."""
        self._top = 0
        self._free: list[int] = []
        self._deferred: list[int] = []
        self._held: list[int] = []  # freed between steps: deferred at the next recycle

    def __len__(self) -> int:
        """Number of allocated, not yet freed vehicles."""
        return len(self.id_to_handle)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def top(self) -> int:
        """High-water mark: every handle ever used is below it."""
        return self._top

    @property
    def next_uid(self) -> int:
        return len(self.uid_to_id)

    def reset(self) -> None:
        """Forget every vehicle; uids restart at 0 (capacity is kept)."""
        self._top = 0
        self._free.clear()
        self._deferred.clear()
        self._held.clear()
        self.ids = [None] * self._capacity
        self.id_to_handle.clear()
        self.uid_to_id.clear()
        self.active[:] = False

    def alloc(self, vehicle_id: str) -> int:
        """A fresh row for ``vehicle_id`` (status pending, next uid); returns its handle.

        Reuses the most recently recycled handle (LIFO), else the next unused row, growing
        the table (doubling) when full. ``CommandError`` if the id is already live.
        """
        if vehicle_id in self.id_to_handle:
            raise CommandError(f'vehicle id "{vehicle_id}" is already in use')
        if self._free:
            h = self._free.pop()
        else:
            if self._top == self._capacity:
                self._grow(2 * self._capacity)
            h = self._top
            self._top += 1
        for name, (_, fill) in COLUMNS.items():
            getattr(self, name)[h] = fill
        self.uid[h] = len(self.uid_to_id)
        self.uid_to_id.append(vehicle_id)
        self.ids[h] = vehicle_id
        self.id_to_handle[vehicle_id] = h
        return h

    def free_deferred(self, handles: IntArray | int, *, between_steps: bool = False) -> None:
        """Release rows whose vehicles left (status already arrived/removed).

        They stop being ``active`` now and leave ``id_to_handle``; ``ids`` keeps the id, and
        the handle is reusable only after the next :meth:`recycle`, or the one after it
        with ``between_steps=True`` (a command between steps: the next step's events still
        name the vehicle by this handle).
        """
        for h in np.atleast_1d(np.asarray(handles, dtype=np.intp)).tolist():
            vid = self.ids[h]
            if vid is None or self.id_to_handle.get(vid) != h:
                raise ValueError(f"handle {h} is not a live vehicle")
            self.active[h] = False
            del self.id_to_handle[vid]
            (self._held if between_steps else self._deferred).append(h)

    def recycle(self) -> None:
        """Make the handles freed since the last call reusable (engine sub-step 0); those
        freed between steps become reusable at the following call."""
        for h in self._deferred:
            self.ids[h] = None
        self._free.extend(self._deferred)
        self._deferred, self._held = self._held, []

    def running(self) -> IntArray:
        """Handles of running vehicles, ascending: ``flatnonzero(active)``."""
        return np.flatnonzero(self.active[: self._top])

    def state_dict(self) -> dict[str, Any]:
        """The table's state: ``"columns"`` (copies of every column sliced to :attr:`top`)
        plus the JSON-safe slot lists and id maps (``ids``, ``id_to_handle`` as ``[id,
        handle]`` pairs, ``uid_to_id``, ``free``, ``deferred``, ``held``)."""
        top = self._top
        return {
            "columns": {name: getattr(self, name)[:top].copy() for name in COLUMNS},
            "ids": self.ids[:top],
            "id_to_handle": [[vid, h] for vid, h in self.id_to_handle.items()],
            "uid_to_id": list(self.uid_to_id),
            "free": list(self._free),
            "deferred": list(self._deferred),
            "held": list(self._held),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Replace the table by :meth:`state_dict` output (the rows above the high-water
        mark are fresh). ``ValueError`` if a column is missing or has the wrong dtype or
        length."""
        ids: list[str | None] = list(state["ids"])
        top = len(ids)
        columns = state["columns"]
        for name, (dtype, _) in COLUMNS.items():
            col = columns.get(name)
            if col is None or col.shape != (top,) or col.dtype != np.dtype(dtype):
                raise ValueError(f"vehicle column {name!r} is missing or malformed")
        capacity = max(self._capacity, top)
        for name, (dtype, fill) in COLUMNS.items():
            arr = np.full(capacity, fill, dtype=dtype)
            arr[:top] = columns[name]
            setattr(self, name, arr)
        self._capacity, self._top = capacity, top
        self.ids = ids + [None] * (capacity - top)
        self.id_to_handle = {str(vid): int(h) for vid, h in state["id_to_handle"]}
        self.uid_to_id = [str(v) for v in state["uid_to_id"]]
        self._free = [int(h) for h in state["free"]]
        self._deferred = [int(h) for h in state["deferred"]]
        self._held = [int(h) for h in state["held"]]

    def handle_of(self, vehicle_id: str) -> int:
        """Handle of a live vehicle; ``NotFoundError`` with a hint if unknown."""
        try:
            return self.id_to_handle[vehicle_id]
        except KeyError:
            hint = suggest(vehicle_id, self.id_to_handle)
            raise NotFoundError(f'unknown vehicle "{vehicle_id}"{hint}') from None

    def _grow(self, capacity: int) -> None:
        for name, (dtype, fill) in COLUMNS.items():
            new = np.full(capacity, fill, dtype=dtype)
            new[: self._capacity] = getattr(self, name)
            setattr(self, name, new)
        self.ids.extend([None] * (capacity - self._capacity))
        self._capacity = capacity
