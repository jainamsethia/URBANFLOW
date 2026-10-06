"""Engine snapshot, restore, digest and state files (plan F.5): the parts of EngineState."""

from __future__ import annotations

import io
import json
import struct
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from urbanflow import generate
from urbanflow.core.errors import NotFoundError, SimulationError
from urbanflow.core.npz import write_npz
from urbanflow.core.types import VehicleStatus
from urbanflow.engine import Engine
from urbanflow.engine.state import (
    COUNTERS,
    STATE_JSON,
    STATE_NPZ,
    EngineState,
    config_hash,
    load_state,
    save_state,
    snapshot,
    state_digest,
)
from urbanflow.scenario import Scenario
from urbanflow.signals import ControllerBase, ControllerContext, External, FixedTime
from urbanflow.vehicles.table import COLUMNS

MakeEngine = Callable[..., Engine]
J = 0


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    """The signalised 2-lane junction, busy enough for queues, commits and zone locks."""
    return generate("single_intersection", demand_rate=700, duration=600)


@pytest.fixture
def busy(make_engine: MakeEngine, scenario: Scenario) -> Engine:
    e = make_engine(scenario, seed=4)
    for _ in range(90):
        e.step()
    e.commands.add_vehicle(route=["W_in", "E_out"])  # waiting, with a pending event
    return e


def _steps(e: Engine, n: int) -> list[str]:
    out = []
    for _ in range(n):
        e.step()
        out.append(e.digest())
    return out


def test_snapshot_is_a_deep_read_only_copy(busy: Engine) -> None:
    e = busy
    s = e.snapshot()
    assert isinstance(s, EngineState) and s.scenario_hash == e.network.scenario_hash
    assert s.config == e.config.model_dump(mode="json") and s.step_count == 90
    top = e.vehicles.top
    for name in COLUMNS:
        arr = s.arrays[f"veh.{name}"]
        assert arr.shape == (top,) and not arr.flags.writeable
    assert s.arrays["sneakers"].shape == (e.network.n_lanes,)
    json.dumps(s.meta)  # the JSON half is JSON
    counters = s.meta["counters"]
    assert list(counters) == list(COUNTERS)
    assert (counters["generated"], counters["next_uid"]) == (e.generated, e.vehicles.next_uid)
    assert s.meta["controllers"] == {
        "J": {"type": "fixed_time", "params": {"offset": 0.0}, "state": {}}
    }
    assert s.meta["commands"]["pending"][0][0] == "vehicle_departed"
    assert s.meta["command_log"][0][:2] == [90, "add_vehicle"]
    pos, digest = s.arrays["veh.pos"].copy(), state_digest(s)
    _steps(e, 5)
    assert np.array_equal(s.arrays["veh.pos"], pos) and state_digest(s) == digest
    assert snapshot(e).step_count == 95


def test_restore_continues_exactly(busy: Engine) -> None:
    e = busy
    s = e.snapshot()
    before = e.digest()
    ahead = _steps(e, 40)
    e.commands.set_controller("J", "external")  # undone by the restore
    e.commands.add_vehicle(route=["N_in", "S_out"])
    e.restore(s)
    assert e.digest() == before and e.step_count == 90 and e.holds is None
    assert len(e.events) == 0
    assert isinstance(e.controllers[J], FixedTime) and len(e.commands.command_log) == 1
    assert _steps(e, 40) == ahead
    e.restore(s)  # the same snapshot twice
    assert _steps(e, 40) == ahead


def _changed(value: Any) -> Any:
    if isinstance(value, bool | np.bool_):
        return not value
    if isinstance(value, float | np.floating):
        return value + 0.5 if np.isfinite(value) else 1.0
    return value + 1


def _digest_with(e: Engine, name: str, h: int, value: Any) -> str:
    col = getattr(e.vehicles, name)
    old = col[h].copy()
    col[h] = value
    try:
        return e.digest()
    finally:
        col[h] = old


@pytest.mark.parametrize("name", list(COLUMNS))
def test_digest_changes_with_any_column_of_a_live_vehicle(busy: Engine, name: str) -> None:
    e = busy
    before = e.digest()
    running = int(e.vehicles.running()[0])
    waiting = e.vehicles.handle_of("api.0")
    for h in (running, waiting):
        assert _digest_with(e, name, h, _changed(getattr(e.vehicles, name)[h])) != before
    assert e.digest() == before


def test_digest_ignores_dead_rows_and_nan_payloads(busy: Engine) -> None:
    e = busy
    veh = e.vehicles
    victim = str(veh.ids[int(veh.running()[-1])])
    dead = veh.handle_of(victim)
    e.commands.remove_vehicle(victim)
    before = e.digest()
    assert _digest_with(e, "pos", dead, 123.0) == before
    h = int(veh.running()[0])
    assert np.isnan(veh.speed_override[h])
    other_nan = np.frombuffer(np.uint64(0x7FF8_0000_0000_0001).tobytes(), dtype=np.float64)[0]
    assert _digest_with(e, "speed_override", h, other_nan) == before


def test_digest_rows_follow_uid_order_not_slots(busy: Engine) -> None:
    e = busy
    e.step()  # no pending command events (they name slots)
    s = e.snapshot()
    a, b = (int(h) for h in e.vehicles.running()[:2])
    arrays = {k: v.copy() for k, v in s.arrays.items()}
    for name in COLUMNS:  # the two vehicles trade slots
        col = arrays[f"veh.{name}"]
        col[[a, b]] = col[[b, a]]
    meta = json.loads(json.dumps(s.meta))
    swap = {a: b, b: a}
    pairs = meta["vehicles"]["id_to_handle"]
    meta["vehicles"]["id_to_handle"] = [[vid, swap.get(h, h)] for vid, h in pairs]
    assert state_digest(EngineState(s.scenario_hash, s.config, arrays, meta)) == state_digest(s)
    arrays["veh.uid"][[a, b]] = arrays["veh.uid"][[b, a]]  # same slots, other order
    assert state_digest(EngineState(s.scenario_hash, s.config, arrays, meta)) != state_digest(s)


def _mutations(e: Engine) -> dict[str, Callable[[], object]]:
    q = e.queues
    road = q.roads()[0]
    return {
        "counter": lambda: setattr(e, "red_runs", e.red_runs + 1),
        "commit counter": lambda: setattr(e, "_next_seq", e._next_seq + 1),
        "route": lambda: e.routes.intern([0]),
        "queue order": lambda: q[road].rotate(1) if len(q[road]) > 1 else q.push(road, 0),
        "signal timer": lambda: e.signals.stage_elapsed.__setitem__(J, 0.25),
        "sneakers": lambda: e.sneakers.__setitem__(0, 1),
        "detector": lambda: e.detector_seen.__setitem__(0, 1.0),
        "held flag": lambda: e.held.__setitem__(J, True),
        "controller": lambda: e.commands.set_controller("J", "external"),
        "spawner": lambda: setattr(e.spawners[0], "count", e.spawners[0].count + 7),
        "rng": lambda: e.rng.stream("flow:N_E").random(),
        "fresh rng stream": lambda: e.rng.stream("generator:unused"),
        "api counter": lambda: setattr(e.commands, "_api_count", 5),
        "forced jump": lambda: e.commands.set_phase("J", 1),
    }


def test_digest_covers_every_part_of_the_state(busy: Engine) -> None:
    e = busy
    s = e.snapshot()
    for name, mutate in _mutations(e).items():
        before = e.digest()
        mutate()
        assert e.digest() != before, name
        e.restore(s)
        assert e.digest() == state_digest(s), name
    e.commands.command_log.append((0, "x", {}))  # the log is restored but not digested
    assert e.digest() == state_digest(s)


def test_digest_never_reads_the_command_log(busy: Engine) -> None:
    """P4 review: a digest copied the whole log (0.5 ms per digest, 21 ms after 20k
    commands, so per-step digests were O(n^2)). It is not digested, so it is not read."""

    class Untouchable(list[Any]):
        def __iter__(self) -> Any:
            raise AssertionError("the digest read the command log")

    e = busy
    s = e.snapshot()
    assert s.meta["command_log"] and not snapshot(e, with_log=False).meta["command_log"]
    e.commands.command_log = Untouchable(e.commands.command_log)
    assert e.digest() == state_digest(s)


def test_restore_rejects_another_scenario_or_config(
    make_engine: MakeEngine, scenario: Scenario, busy: Engine
) -> None:
    s = busy.snapshot()
    other = make_engine(generate("single_intersection", kind="priority"))
    with pytest.raises(SimulationError, match="the snapshot is of another scenario"):
        other.restore(s)
    with pytest.raises(
        SimulationError, match=r"another simulation config \(it differs in: dt, seed\)"
    ):
        make_engine(scenario, dt=0.5, seed=1).restore(s)
    assert config_hash(busy.config) == config_hash(json.loads(json.dumps(s.config)))


def test_restore_rejects_a_malformed_state(busy: Engine) -> None:
    s = busy.snapshot()
    arrays = {k: v for k, v in s.arrays.items() if k != "veh.pos"}
    with pytest.raises(SimulationError, match=r"malformed engine state: KeyError: 'veh\.pos'"):
        busy.restore(EngineState(s.scenario_hash, s.config, arrays, s.meta))
    meta = {**s.meta, "spawners": s.meta["spawners"][:-1]}
    with pytest.raises(SimulationError, match="spawner states"):
        busy.restore(EngineState(s.scenario_hash, s.config, s.arrays, meta))


@dataclass
class Parts:
    """A deep copy of a state's ``meta`` and ``arrays`` to break, and a running handle."""

    m: dict[str, Any]
    a: dict[str, np.ndarray]
    h: int


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda p: p.m.__setitem__("queues", [[0, [10**9]]]), "insertion queues"),
        (lambda p: p.m["vehicles"]["id_to_handle"][0].__setitem__(1, 10**9), "slots"),
        (lambda p: p.m["vehicles"]["free"].append(p.h), "listed twice"),
        (lambda p: p.m["vehicles"]["uid_to_id"].__setitem__(p.a["veh.uid"][p.h], "x"), "maps"),
        (lambda p: p.m.__setitem__("queues", []), "insertion queues"),
        (lambda p: p.m["queues"][0].__setitem__(0, 999), "road out of range"),
        (lambda p: p.m["routes"][0].__setitem__(0, 999), "road out of range"),
        (lambda p: p.a["veh.route_id"].__setitem__(p.h, 999), "route or type"),
        (lambda p: p.a["veh.type_idx"].__setitem__(p.h, 99), "route or type"),
        (lambda p: p.a["veh.link"].__setitem__(p.h, 10**6), "link, connector"),
        (lambda p: p.a["veh.lock_conn"].__setitem__(p.h, 0), "link, connector"),
        (lambda p: p.a["veh.route_cursor"].__setitem__(p.h, 99), "route cursor"),
        (lambda p: p.a["veh.active"].__setitem__(p.h, False), "active flags"),
        (lambda p: p.m["commands"]["pending"].append(["vehicle_removed", 0, 10**9, 0, -1]), "pend"),
    ],
)
def test_restore_rejects_out_of_range_values(
    busy: Engine, edit: Callable[[Parts], None], message: str
) -> None:
    """P4 review repro: ``meta.queues = [[0, [10**9]]]`` restored fine and the next digest
    raised IndexError. Every index is now range-checked: ``SimulationError``."""
    s = busy.snapshot()
    arrays = {k: v.copy() for k, v in s.arrays.items()}
    parts = Parts(
        json.loads(json.dumps(s.meta)), arrays, int(np.flatnonzero(arrays["veh.active"])[0])
    )
    edit(parts)
    with pytest.raises(SimulationError, match=f"malformed engine state: ValueError: .*{message}"):
        busy.restore(EngineState(s.scenario_hash, s.config, parts.a, parts.m))
    busy.restore(s)  # a good state repairs it
    assert busy.digest() == state_digest(s)


# ------------------------------------------------------------------------------- controllers
class Counting(ControllerBase):
    """An unregistered controller with state: counts its decisions."""

    def __init__(self) -> None:
        self.n = 0

    def decide(self, ctx: ControllerContext) -> int | None:  # noqa: ARG002
        self.n += 1
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"n": self.n}

    def load_state_dict(self, d: Any) -> None:
        self.n = int(d["n"])


def test_controllers_holds_and_parked_round_trip(busy: Engine) -> None:
    e = busy
    configured = e.controllers[J]
    e.commands.hold_phase("J", 1)
    held_external = e.controllers[J]
    assert isinstance(held_external, External) and held_external.pending == 1
    s = e.snapshot()
    assert s.meta["held"] == [True] + [False] * (e.network.n_intersections - 1)
    assert s.meta["parked"] == {"J": {"type": "fixed_time", "params": {"offset": 0.0}, "state": {}}}
    ahead = _steps(e, 20)
    e.commands.release("J")  # the configured instance is active again
    e.restore(s)
    assert e.held[J] and e.parked[J] is configured  # the matching instance is reused
    assert isinstance(e.controllers[J], External) and e.controllers[J].pending == 1
    assert (e.specs[J]["type"], e.parked_specs[J]["type"]) == ("external", "fixed_time")
    assert _steps(e, 20) == ahead


def test_unregistered_controllers_are_reused_or_reported(
    make_engine: MakeEngine, scenario: Scenario
) -> None:
    e = make_engine(scenario, controllers={J: Counting})
    _steps(e, 10)
    counting = e.controllers[J]
    assert isinstance(counting, Counting) and counting.n == 10
    s = e.snapshot()
    assert s.meta["controllers"]["J"] == {"type": "Counting", "params": {}, "state": {"n": 10}}
    _steps(e, 5)
    e.restore(s)
    assert e.controllers[J] is counting and counting.n == 10
    e.commands.set_controller("J", "external")  # the Counting instance is gone
    with pytest.raises(SimulationError, match='cannot restore the signal controller "Counting"'):
        e.restore(s)


# ------------------------------------------------------------------------------- files
def test_file_round_trip(busy: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e = busy
    s = e.snapshot()
    path = save_state(tmp_path / "state.ufs", s)
    assert path == tmp_path / "state.ufs" and not (tmp_path / "state.ufs.tmp").exists()
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == [STATE_JSON, STATE_NPZ]
        doc = json.loads(zf.read(STATE_JSON))
    assert (doc["format"], doc["version"]) == ("urbanflow.state", "1.0")

    def banned(*args: object, **kwargs: object) -> None:
        raise AssertionError("np.load must not be used")

    monkeypatch.setattr(np, "load", banned)
    loaded = load_state(path)
    assert loaded.meta == s.meta and loaded.config == s.config
    assert set(loaded.arrays) == set(s.arrays)
    for name, arr in s.arrays.items():
        assert loaded.arrays[name].dtype == arr.dtype and not loaded.arrays[name].flags.writeable
        np.testing.assert_array_equal(loaded.arrays[name], arr)
    assert state_digest(loaded) == state_digest(s)
    ahead = _steps(e, 30)
    e.restore(loaded)
    assert _steps(e, 30) == ahead


def _rewrite(
    path: Path, target: Path, *, npz: bytes = b"", extra: bool = False, **doc_changes: Any
) -> Path:
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    doc = {**json.loads(members[STATE_JSON]), **doc_changes}
    members[STATE_JSON] = json.dumps(doc).encode()
    if npz:
        members[STATE_NPZ] = npz
    if extra:
        members["evil.py"] = b""
    with zipfile.ZipFile(target, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return target


def test_load_state_rejects_bad_files(busy: Engine, tmp_path: Path) -> None:
    good = save_state(tmp_path / "good.ufs", busy.snapshot())
    with pytest.raises(NotFoundError, match="no saved state"):
        load_state(tmp_path / "missing.ufs")
    (tmp_path / "text.ufs").write_text("hello", encoding="utf-8")
    with pytest.raises(SimulationError, match="is not a valid UrbanFlow state file"):
        load_state(tmp_path / "text.ufs")
    with zipfile.ZipFile(good) as zf, zipfile.ZipFile(io.BytesIO(zf.read(STATE_NPZ))) as inner:
        npz_names = inner.namelist()
    pickled = io.BytesIO()
    np.lib.format.write_array(pickled, np.array([None], dtype=object), allow_pickle=True)
    objects = io.BytesIO()
    with zipfile.ZipFile(objects, "w") as zf:
        for name in npz_names:
            zf.writestr(name, pickled.getvalue())
    partial = io.BytesIO()
    write_npz(partial, {"veh.pos": np.zeros(1)}, 1)
    with zipfile.ZipFile(good) as zf:  # P4 review: a corrupt deflate stream (zlib.error)
        npz = zf.read(STATE_NPZ)
    start = 30 + sum(struct.unpack("<HH", npz[26:30])) + 20
    flipped = npz[:start] + bytes(b ^ 0xFF for b in npz[start : start + 8]) + npz[start + 8 :]
    cases: dict[str, tuple[dict[str, Any], str]] = {
        "corrupt": ({"npz": flipped}, r"veh\.status\.npy: unreadable"),
        "extra": ({"extra": True}, "expected the members state.json and state.npz"),
        "format": ({"format": "other"}, '"format" is not "urbanflow.state"'),
        "version": ({"version": "2.0"}, "version '2.0' is not supported"),
        "object": ({"npz": objects.getvalue()}, "dtype object is not allowed"),
        "missing": ({"npz": partial.getvalue()}, "missing npz members"),
    }
    for name, (changes, message) in cases.items():
        bad = _rewrite(good, tmp_path / f"{name}.ufs", **changes)
        with pytest.raises(SimulationError, match=message):
            load_state(bad)


def test_waiting_and_removed_vehicles_round_trip(busy: Engine) -> None:
    e = busy
    victim = next(iter(e.vehicles.id_to_handle))
    e.commands.remove_vehicle(victim)  # freed between steps, event pending
    s = e.snapshot()
    ahead = _steps(e, 10)
    e.restore(s)
    h = e.vehicles.handle_of("api.0")
    assert e.vehicles.status[h] == VehicleStatus.waiting_insert.code
    assert victim not in e.vehicles.id_to_handle
    assert _steps(e, 10) == ahead
