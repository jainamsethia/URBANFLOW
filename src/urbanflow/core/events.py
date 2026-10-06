"""Engine events: the canonical :class:`EventType` list, the columnar :class:`EventBuffer`
and the :class:`EventBus` of user callbacks.

The buffer holds the events of the *current* step (B.2 #12, F.8): the engine clears it at
step start and appends with vectorised :meth:`EventBuffer.append_many`; metrics, the replay
recorder and the server read the columns directly. Vehicle identity outside L4/L5 is the
``uid`` column (B.2 #1); ``handle`` is for engine internals only.

The bus (``sim.events.subscribe``) materialises frozen :class:`Event` objects, with ids
instead of indices, only for the types someone subscribed to: without subscribers a step
costs nothing extra. Kernels never call user code; the facade dispatches after the step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from urbanflow.core.errors import NotFoundError, suggest
from urbanflow.core.types import _CodedEnum

__all__ = [
    "COLUMNS",
    "NO_VEHICLE",
    "PHASE_FORCED_BIT",
    "Event",
    "EventBuffer",
    "EventBus",
    "EventCallback",
    "EventType",
    "Subscription",
    "decode_phase_aux",
    "phase_aux",
]

NO_VEHICLE: Final = 0xFFFF_FFFF
"""``handle``/``uid`` of events that are not about a vehicle (u32 maximum)."""

PHASE_FORCED_BIT: Final = 1 << 16
"""Set in ``aux`` of a ``phase_changed`` event caused by a forced ``set_phase`` jump (B.2
#25); the low bits hold the new phase index."""


def phase_aux(phase: int, *, forced: bool = False) -> int:
    """``aux`` of a ``phase_changed`` event: the new phase, plus :data:`PHASE_FORCED_BIT`
    when the change was forced."""
    return phase | PHASE_FORCED_BIT if forced else phase


def decode_phase_aux(aux: int) -> tuple[int, bool]:
    """Inverse of :func:`phase_aux`: ``(phase, forced)``."""
    return aux & (PHASE_FORCED_BIT - 1), bool(aux & PHASE_FORCED_BIT)


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
        """Type-specific payload (``phase_changed``: :func:`phase_aux`)."""
        return self._col("aux")


# ------------------------------------------------------------------------------- the bus
@dataclass(frozen=True, slots=True)
class Event:
    """One event as user callbacks see it (``sim.events.subscribe``; AA 5.4).

    Identity is by id, never by engine index or slot: ``vehicle`` is the vehicle id,
    ``link`` the lane or connector id and ``intersection`` the intersection id, each None
    when the event has none. ``data`` holds type-specific fields: ``phase_changed`` has
    ``{"phase": index, "forced": bool}`` (B.2 #25); the other types have none. ``data`` is
    a read-only mapping: the same event goes to every subscriber.
    """

    type: EventType
    time: float
    """s; ``vehicle_arrived`` is interpolated within the step."""
    step: int
    """The step whose frame the event belongs to (``step_count`` after it)."""
    vehicle: str | None
    link: str | None
    intersection: str | None
    data: Mapping[str, Any]


EventCallback = Callable[[Event], None]

_NO_DATA: Final[Mapping[str, Any]] = MappingProxyType({})


class Subscription:
    """A registered callback; :meth:`unsubscribe` removes it (idempotent)."""

    def __init__(self, bus: EventBus, types: frozenset[EventType] | None, callback: EventCallback):
        self._bus = bus
        self.types = types
        """The subscribed types; None means every type."""
        self.callback = callback

    @property
    def active(self) -> bool:
        """Still receiving events."""
        return self in self._bus._subs

    def unsubscribe(self) -> None:
        """Stop receiving events (a no-op when already unsubscribed)."""
        if self.active:
            self._bus._subs.remove(self)

    def __repr__(self) -> str:
        names = "all" if self.types is None else ", ".join(sorted(t.value for t in self.types))
        return f"Subscription({names}, active={self.active})"


def _event_types(
    types: EventType | str | Iterable[EventType | str] | None,
) -> frozenset[EventType] | None:
    if types is None:
        return None
    items = [types] if isinstance(types, str) else list(types)
    names = [t.value for t in EventType]
    out = set()
    for t in items:
        try:
            out.add(EventType(t))
        except ValueError:
            raise NotFoundError(
                f'unknown event type "{t}"{suggest(str(t), names)} (available: {", ".join(names)})'
            ) from None
    return frozenset(out)


class EventBus:
    """User callbacks per event type (F.8), called in subscription order.

    :meth:`dispatch` turns the rows of an :class:`EventBuffer` whose type someone
    subscribed to into :class:`Event` objects, in buffer order, and calls every matching
    callback. Callbacks may subscribe or unsubscribe: a new subscription receives events
    from the next dispatch, an unsubscribed one receives nothing more. A callback's
    exception propagates out of :meth:`dispatch` and skips the remaining calls.
    """

    def __init__(self) -> None:
        self._subs: list[Subscription] = []

    @property
    def has_subscribers(self) -> bool:
        return bool(self._subs)

    def subscribe(
        self,
        types: EventType | str | Iterable[EventType | str] | None,
        callback: EventCallback,
    ) -> Subscription:
        """Call ``callback(event)`` for every event of ``types`` (one type, several, or
        None for all). Unknown type names raise ``NotFoundError`` with a hint."""
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {callback!r}")
        sub = Subscription(self, _event_types(types), callback)
        self._subs.append(sub)
        return sub

    def dispatch(
        self,
        events: EventBuffer,
        *,
        vehicle_ids: Sequence[str],
        link_ids: Sequence[str],
        intersection_ids: Sequence[str],
    ) -> None:
        """Deliver the events of ``events`` to the subscribers.

        ``vehicle_ids`` maps uids to vehicle ids, ``link_ids`` and ``intersection_ids``
        indices to ids. Nothing is materialised for types nobody subscribed to.
        """
        subs = list(self._subs)
        if not subs or not len(events):
            return
        wanted: set[EventType] = set()
        for sub in subs:
            wanted |= set(EventType) if sub.types is None else sub.types
        kind = events.type
        rows = np.flatnonzero(np.isin(kind, [t.code for t in wanted]))
        if not rows.size:
            return
        cols = events.view()
        values = {name: cols[name][rows].tolist() for name in ("type", "step", "time", "uid")}
        links, ints, aux = (cols[c][rows].tolist() for c in ("link", "intersection", "aux"))
        for i, code in enumerate(values["type"]):
            kind_i = EventType.from_code(code)
            data: Mapping[str, Any] = _NO_DATA
            if kind_i is EventType.phase_changed:  # read-only: every subscriber sees it
                phase, forced = decode_phase_aux(aux[i])
                data = MappingProxyType({"phase": phase, "forced": forced})
            uid = values["uid"][i]
            event = Event(
                type=kind_i,
                time=values["time"][i],
                step=values["step"][i],
                vehicle=None if uid == NO_VEHICLE else vehicle_ids[uid],
                link=None if links[i] < 0 else link_ids[links[i]],
                intersection=None if ints[i] < 0 else intersection_ids[ints[i]],
                data=data,
            )
            for sub in subs:
                if (sub.types is None or kind_i in sub.types) and sub.active:
                    sub.callback(event)
