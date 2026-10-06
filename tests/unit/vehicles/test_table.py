"""The SoA vehicle table: columns, growth, deferred handle reuse, uids, id maps (E.4, B.2 #1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urbanflow.core.errors import CommandError, NotFoundError
from urbanflow.core.types import VehicleStatus
from urbanflow.vehicles.table import COLUMNS, DEPART_LANE_BEST, VehicleTable

# Plan E.4, verbatim: field -> dtype.
E4 = {
    "status": np.uint8,
    "active": np.bool_,
    "uid": np.uint32,
    "type_idx": np.uint16,
    "source_idx": np.int32,
    "link": np.int32,
    "pos": np.float64,
    "speed": np.float64,
    "accel": np.float64,
    "length": np.float32,
    "width": np.float32,
    "speed_factor": np.float32,
    "v0": np.float64,
    "route_id": np.int32,
    "route_cursor": np.int16,
    "valid_mask": np.uint16,
    "next_conn": np.int32,
    "committed": np.bool_,
    "granted": np.bool_,
    "lock_conn": np.int32,
    "forced": np.bool_,
    "commit_seq": np.int64,
    "held": np.bool_,
    "lat_offset": np.float32,
    "lc_cooldown": np.float32,
    "lc_target": np.int16,
    "lc_target_until": np.float64,
    "speed_override": np.float64,
    "override_until": np.float64,
    "depart_time": np.float64,
    "insert_time": np.float64,
    "waiting_time": np.float64,
    "link_waiting_time": np.float32,
    "stops": np.int32,
    "halting": np.bool_,
    "distance": np.float64,
    "ff_time": np.float64,
    "lane_changes": np.int16,
    "teleports": np.int16,
    "energy": np.float64,
    "stuck_time": np.float32,
    "stop_idx": np.int16,
    "dwell_left": np.float32,
}


def test_every_e4_column_with_its_dtype() -> None:
    table = VehicleTable(4)
    for name, dtype in E4.items():
        col = getattr(table, name)
        assert col.dtype == dtype, name
        assert col.shape == (4,)
    # the insertion request of waiting vehicles is part of the record too (G.8)
    assert set(COLUMNS) == set(E4) | {"depart_lane", "depart_speed"}


def test_fresh_rows_have_neutral_values() -> None:
    table = VehicleTable(2)
    h = table.alloc("a")
    assert table.status[h] == VehicleStatus.pending.code
    assert not table.active[h]
    assert (table.link[h], table.route_id[h], table.next_conn[h], table.lock_conn[h]) == (-1,) * 4
    assert (table.commit_seq[h], table.lc_target[h], table.stop_idx[h]) == (-1, -1, -1)
    assert math.isnan(table.speed_override[h])
    assert math.isinf(table.override_until[h])
    assert math.isnan(table.insert_time[h])
    assert table.depart_lane[h] == DEPART_LANE_BEST
    assert math.isnan(table.depart_speed[h])
    # a reused row is reset, not inherited
    table.pos[h], table.committed[h] = 42.0, True
    table.free_deferred(h)
    table.recycle()
    assert table.alloc("b") == h
    assert table.pos[h] == 0.0
    assert not table.committed[h]


def test_growth_doubles_and_keeps_data() -> None:
    table = VehicleTable(2)
    handles = [table.alloc(f"v{i}") for i in range(5)]
    assert handles == [0, 1, 2, 3, 4]
    assert table.capacity == 8
    for h in handles:
        table.pos[h] = 10.0 * h
        table.active[h] = True
    table.alloc("v5")
    table.alloc("v6")
    table.alloc("v7")
    table.alloc("v8")  # grows again
    assert table.capacity == 16
    assert table.pos[:5].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert table.ids[:9] == [f"v{i}" for i in range(9)]
    assert len(table.ids) == 16
    assert table.running().tolist() == handles


def test_freed_handles_are_reused_only_after_recycle() -> None:
    table = VehicleTable(8)
    a, b, c = (table.alloc(x) for x in "abc")
    table.status[[a, b]] = VehicleStatus.arrived.code
    table.free_deferred(np.array([a, b]))
    # same step: a new vehicle gets a fresh row, never a handle freed this step (B.2 #1)
    d = table.alloc("d")
    assert d not in (a, b)
    assert table.ids[a] == "a"  # still readable for this step's events
    assert "a" not in table.id_to_handle
    table.recycle()
    assert table.ids[a] is None
    assert table.alloc("e") == b  # LIFO: last freed, first reused
    assert table.alloc("f") == a
    assert table.alloc("g") == table.top - 1  # free list exhausted: next unused row
    assert c == 2


def test_handles_freed_between_steps_wait_one_more_recycle() -> None:
    """A command frees a handle between steps; the next step still reports the removal
    with it, so it must not be reused by that step's spawns (B.2 #1)."""
    table = VehicleTable(8)
    a = table.alloc("a")
    table.status[a] = VehicleStatus.removed.code
    table.free_deferred(a, between_steps=True)
    assert "a" not in table.id_to_handle and not table.active[a]
    table.recycle()  # sub-step 0 of the step that emits vehicle_removed
    assert table.ids[a] == "a" and table.alloc("b") != a
    table.recycle()
    assert table.ids[a] is None and table.alloc("c") == a


def test_uids_are_monotonic_and_never_reused() -> None:
    table = VehicleTable(2)
    h0 = table.alloc("x")
    table.free_deferred(h0)
    table.recycle()
    h1 = table.alloc("y")
    assert h1 == h0
    assert table.uid[h1] == 1
    assert table.uid_to_id == ["x", "y"]
    assert table.next_uid == 2
    table.reset()
    assert table.next_uid == 0
    assert table.alloc("z") == 0
    assert table.uid[0] == 0
    assert table.uid_to_id == ["z"]
    assert table.id_to_handle == {"z": 0}


def test_id_maps_stay_consistent() -> None:
    table = VehicleTable(4)
    for name in ("f.0", "f.1", "trip"):
        table.alloc(name)
    assert len(table) == 3
    assert table.handle_of("f.1") == 1
    assert all(table.ids[h] == vid for vid, h in table.id_to_handle.items())
    with pytest.raises(CommandError, match="already in use"):
        table.alloc("f.0")
    with pytest.raises(NotFoundError, match='did you mean "trip"'):
        table.handle_of("trp")
    table.free_deferred(1)
    assert len(table) == 2
    with pytest.raises(ValueError, match="not a live vehicle"):
        table.free_deferred(1)
    table.alloc("f.1")  # the id is free again (a new uid)
    assert table.uid_to_id.count("f.1") == 2


def test_running_is_flatnonzero_of_active() -> None:
    table = VehicleTable(4)
    for i in range(6):
        table.alloc(str(i))
    table.active[[4, 1, 5]] = True
    assert table.running().tolist() == [1, 4, 5]
    table.free_deferred(4)
    assert table.running().tolist() == [1, 5]


def test_state_dict_round_trip() -> None:
    import json

    t = VehicleTable(2)
    for i in range(5):
        h = t.alloc(f"v{i}")
        t.pos[h] = 10.0 * i
    t.free_deferred(1)
    t.recycle()  # handle 1 is free
    t.free_deferred(3)  # deferred
    t.free_deferred(4, between_steps=True)  # held
    state = t.state_dict()
    cols = state.pop("columns")
    assert set(cols) == set(COLUMNS) and all(c.shape == (5,) for c in cols.values())
    state = json.loads(json.dumps(state))  # the rest is JSON-safe
    assert state["free"] == [1] and state["deferred"] == [3] and state["held"] == [4]
    assert state["id_to_handle"] == [["v0", 0], ["v2", 2]]
    assert state["ids"] == ["v0", None, "v2", "v3", "v4"]

    other = VehicleTable(64)
    for i in range(9):
        other.alloc(f"x{i}")
        other.active[i] = True
    other.load_state_dict({**state, "columns": cols})
    assert (other.top, other.capacity, other.next_uid) == (5, 64, 5)
    assert not other.active[5:].any() and other.ids[5:] == [None] * 59  # fresh rows
    assert other.pos[:5].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert other.handle_of("v2") == 2 and "v3" not in other.id_to_handle
    assert other.alloc("new") == 1  # the free list is restored (LIFO)
    other.recycle()
    assert other.alloc("again") == 3  # the deferred handle, then the held one
    cols["pos"] = cols["pos"].astype(np.float32)
    with pytest.raises(ValueError, match="'pos' is missing or malformed"):
        VehicleTable().load_state_dict({**state, "columns": cols})
