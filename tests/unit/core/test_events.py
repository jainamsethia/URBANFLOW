"""EventType codes, the columnar EventBuffer and the EventBus (plan B.2 #12, F.8)."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any

import numpy as np
import pytest

import urbanflow
from urbanflow.core import events as events_module
from urbanflow.core.config import RecordEvent
from urbanflow.core.errors import NotFoundError
from urbanflow.core.events import (
    COLUMNS,
    NO_VEHICLE,
    PHASE_FORCED_BIT,
    Event,
    EventBuffer,
    EventBus,
    EventType,
    decode_phase_aux,
    phase_aux,
)

CANONICAL = [
    "vehicle_departed",
    "vehicle_inserted",
    "vehicle_entered_link",
    "vehicle_exited_link",
    "vehicle_stopped",
    "vehicle_resumed",
    "vehicle_changed_lane",
    "vehicle_arrived",
    "vehicle_removed",
    "vehicle_teleported",
    "phase_changed",
    "simulation_started",
    "simulation_ended",
]


def test_event_types_and_codes_are_stable() -> None:
    assert [e.value for e in EventType] == CANONICAL
    assert [e.code for e in EventType] == list(range(len(CANONICAL)))
    assert all(EventType.from_code(e.code) is e for e in EventType)
    assert urbanflow.EventType is EventType


def test_record_config_literal_matches_the_enum() -> None:
    assert list(typing.get_args(RecordEvent)) == CANONICAL


def test_column_dtypes() -> None:
    buf = EventBuffer()
    expected = {
        "type": np.uint8,
        "step": np.int64,
        "time": np.float64,
        "handle": np.uint32,
        "uid": np.uint32,
        "link": np.int32,
        "intersection": np.int32,
        "aux": np.int32,
    }
    assert [name for name, _ in COLUMNS] == list(expected)
    view = buf.view()
    assert list(view) == list(expected)
    assert {k: v.dtype.type for k, v in view.items()} == expected


def test_append_one_with_defaults() -> None:
    buf = EventBuffer()
    buf.append(EventType.phase_changed, 3, 3.0, intersection=2, aux=1)
    buf.append(EventType.vehicle_inserted, 3, 3.0, handle=5, uid=17, link=4)
    assert len(buf) == 2
    assert buf.type.tolist() == [EventType.phase_changed.code, EventType.vehicle_inserted.code]
    assert buf.handle.tolist() == [NO_VEHICLE, 5]
    assert buf.uid.tolist() == [NO_VEHICLE, 17]
    assert buf.link.tolist() == [-1, 4]
    assert buf.intersection.tolist() == [2, -1]
    assert buf.aux.tolist() == [1, 0]
    assert buf.step.tolist() == [3, 3]


def test_append_many_is_vectorised_and_broadcasts() -> None:
    buf = EventBuffer(capacity=2)
    handles = np.array([4, 1, 9])
    buf.append_many(
        EventType.vehicle_arrived,
        12,
        np.array([12.25, 12.5, 12.75]),  # interpolated arrival times
        handles=handles,
        uids=handles + 100,
        links=7,
    )
    assert len(buf) == 3
    assert buf.time.tolist() == [12.25, 12.5, 12.75]
    assert buf.uid.tolist() == [104, 101, 109]
    assert buf.link.tolist() == [7, 7, 7]
    assert set(buf.type.tolist()) == {EventType.vehicle_arrived.code}
    buf.append_many(EventType.vehicle_departed, 12, 12.0, handles=np.array([], dtype=int))
    assert len(buf) == 3  # zero-length append is a no-op


def test_growth_keeps_rows() -> None:
    buf = EventBuffer(capacity=1)
    for i in range(1000):
        buf.append(EventType.vehicle_stopped, i, i * 0.5, handle=i, uid=i)
    assert len(buf) == 1000
    assert buf.capacity >= 1000
    assert np.array_equal(buf.uid, np.arange(1000))
    assert np.array_equal(buf.time, np.arange(1000) * 0.5)


def test_views_are_read_only_and_clear_keeps_storage() -> None:
    buf = EventBuffer()
    buf.append_many(EventType.vehicle_resumed, 1, 1.0, handles=np.arange(5), uids=np.arange(5))
    with pytest.raises(ValueError, match="read-only"):
        buf.uid[0] = 3
    capacity = buf.capacity
    buf.clear()
    assert len(buf) == 0
    assert buf.uid.size == 0
    assert buf.capacity == capacity
    buf.append(EventType.simulation_ended, 2, 2.0)
    assert buf.step.tolist() == [2]


def test_bad_shapes_are_rejected() -> None:
    buf = EventBuffer()
    with pytest.raises(ValueError, match="1-D"):
        buf.append_many(EventType.vehicle_departed, 0, 0.0, handles=np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError):
        buf.append_many(EventType.vehicle_departed, 0, 0.0, handles=[1, 2], uids=[1, 2, 3])
    assert len(buf) == 0


def test_phase_changed_aux_encoding() -> None:
    """B.2 #25: aux = new phase, plus bit 16 when the change was forced."""
    assert PHASE_FORCED_BIT == 1 << 16
    assert phase_aux(3) == 3 and phase_aux(3, forced=True) == 3 + 65536
    for phase in (0, 7, 65535):
        for forced in (False, True):
            aux = phase_aux(phase, forced=forced)
            assert decode_phase_aux(aux) == (phase, forced)
    buf = EventBuffer()  # fits the i32 aux column
    buf.append(EventType.phase_changed, 1, 0.0, intersection=0, aux=phase_aux(2, forced=True))
    assert decode_phase_aux(int(buf.aux[0])) == (2, True) and buf.link[0] == -1


# ------------------------------------------------------------------------------- EventBus
VEHICLES = ["a", "b", "c"]
LINKS = ["L0", "L1", "C0"]
INTERSECTIONS = ["J"]


def _buffer() -> EventBuffer:
    buf = EventBuffer()
    buf.append(EventType.vehicle_inserted, 3, 2.0, handle=5, uid=1, link=0)
    buf.append(EventType.phase_changed, 3, 2.0, intersection=0, aux=phase_aux(1, forced=True))
    buf.append(EventType.vehicle_exited_link, 3, 2.25, handle=5, uid=1, link=0, intersection=-1)
    buf.append(EventType.vehicle_arrived, 3, 2.5, handle=0, uid=2, link=1)
    return buf


def _dispatch(bus: EventBus, buf: EventBuffer) -> None:
    bus.dispatch(buf, vehicle_ids=VEHICLES, link_ids=LINKS, intersection_ids=INTERSECTIONS)


def test_bus_materialises_events_with_ids() -> None:
    bus, seen = EventBus(), []
    bus.subscribe(None, seen.append)
    _dispatch(bus, _buffer())
    assert [e.type for e in seen] == [
        EventType.vehicle_inserted,
        EventType.phase_changed,
        EventType.vehicle_exited_link,
        EventType.vehicle_arrived,
    ]
    inserted, phase, _, arrived = seen
    assert inserted == Event(EventType.vehicle_inserted, 2.0, 3, "b", "L0", None, {})
    assert phase == Event(
        EventType.phase_changed, 2.0, 3, None, None, "J", {"phase": 1, "forced": True}
    )
    assert (arrived.vehicle, arrived.link, arrived.time) == ("c", "L1", 2.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        arrived.time = 0.0  # type: ignore[misc]


def test_event_data_is_read_only_for_every_subscriber() -> None:
    """P4 review: one Event goes to every subscriber, so a callback mutating ``data``
    changed what later subscribers saw for phase_changed."""
    bus, seen = EventBus(), []

    def vandal(e: Event) -> None:
        with pytest.raises(TypeError):
            e.data["phase"] = 0  # type: ignore[index]

    bus.subscribe("phase_changed", vandal)
    bus.subscribe(None, lambda e: seen.append(dict(e.data)))
    _dispatch(bus, _buffer())
    assert seen == [{}, {"phase": 1, "forced": True}, {}, {}]


def test_bus_filters_by_type_and_keeps_subscription_order() -> None:
    bus, calls = EventBus(), []
    bus.subscribe(EventType.vehicle_arrived, lambda e: calls.append(("arrived", e.vehicle)))
    bus.subscribe(
        ["vehicle_inserted", EventType.vehicle_arrived], lambda e: calls.append(("both", e.vehicle))
    )
    bus.subscribe("phase_changed", lambda e: calls.append(("phase", e.data["phase"])))
    _dispatch(bus, _buffer())
    assert calls == [("both", "b"), ("phase", 1), ("arrived", "c"), ("both", "c")]


def test_unsubscribe_is_idempotent_and_works_during_dispatch() -> None:
    bus, calls = EventBus(), []

    def once(e: Event) -> None:
        calls.append(e.type)
        sub.unsubscribe()

    sub = bus.subscribe(None, once)
    later = bus.subscribe(EventType.vehicle_arrived, lambda _: bus.subscribe(None, calls.append))
    assert sub.active and bus.has_subscribers
    _dispatch(bus, _buffer())
    assert calls == [EventType.vehicle_inserted]  # the one added mid-dispatch waits
    assert not sub.active
    sub.unsubscribe()
    later.unsubscribe()
    _dispatch(bus, _buffer())
    assert len(calls) == 1 + 4
    assert "active=False" in repr(later) and "vehicle_arrived" in repr(later)


def test_no_events_are_built_for_unsubscribed_types(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []
    real = events_module.Event

    def counting(*args: Any, **kwargs: Any) -> Event:
        built.append(args or kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(events_module, "Event", counting)
    bus = EventBus()
    _dispatch(bus, _buffer())  # no subscribers at all
    sub = bus.subscribe(EventType.vehicle_teleported, built.append)
    _dispatch(bus, _buffer())  # subscribers, but not for these types
    assert built == []
    sub.unsubscribe()
    bus.subscribe(EventType.vehicle_arrived, lambda _: None)
    _dispatch(bus, _buffer())
    assert len(built) == 1  # only the arrival


def test_callback_errors_propagate() -> None:
    bus, calls = EventBus(), []

    def boom(e: Event) -> None:
        raise RuntimeError("user bug")

    bus.subscribe(EventType.vehicle_inserted, boom)
    bus.subscribe(None, calls.append)
    with pytest.raises(RuntimeError, match="user bug"):
        _dispatch(bus, _buffer())
    assert calls == []  # the rest of that dispatch is skipped


def test_subscribe_validation() -> None:
    bus = EventBus()
    with pytest.raises(NotFoundError, match=r'unknown event type "vehicle_arived" \(did you mean'):
        bus.subscribe("vehicle_arived", print)
    with pytest.raises(TypeError, match="callable"):
        bus.subscribe(None, "print")  # type: ignore[arg-type]
    assert not bus.has_subscribers
