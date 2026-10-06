"""Snapshot, restore and digest of the engine's runtime state (plan F.5, F.6).

:class:`EngineState` holds everything a step reads that is not static: the vehicle table
(columns sliced to the high-water mark, the free and deferred slot lists and the id maps),
the route table, the signal runtime arrays and every controller's ``{type, params,
state_dict}`` (active and parked by a manual hold), the held flags, the spawners (with
their ``bit_generator.state``), the router, every RNG stream, the insertion queues, the
per-lane end-of-green eligibility and detector times, the counters and the command log. Its
two halves are the file form: ``arrays`` (numpy, ``state.npz``) and ``meta`` (JSON-safe,
``state.json``). :func:`snapshot` deep-copies; :func:`restore` copies back after checking
the scenario and config hashes and the configured controllers, then range-checks every
index (a malformed state raises ``SimulationError``).

:func:`digest` is sha256 over a canonical byte stream, in this order (F.5):

1. ``step_count`` and the counters (:data:`COUNTERS`) as little-endian int64;
2. every live or waiting vehicle in uid order, each with every column of
   :data:`~urbanflow.vehicles.table.COLUMNS` packed little-endian (NaN canonicalised);
3. the routes in id order, each as its length (int64) and its roads (int32);
4. the insertion queues in road order: road and length (int64), then the uids (uint32);
5. the signal runtime arrays (:data:`~urbanflow.signals.state.STATE_ARRAYS`), then the
   per-lane end-of-green eligibility (uids) and detector times;
6. ``json.dumps(sort_keys=True)`` of the controllers, the parked controllers, the held
   flags, the spawners, the router, the RNG streams, the unreported forced signal jumps
   and the command bookkeeping (API id counter and unreported command events).

Equal digests on the same platform mean equal futures (F.6); the command log itself is
restored but not digested.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import DTypeLike, NDArray

from urbanflow._version import __version__
from urbanflow.core import constants as C
from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import ConfigError, NotFoundError, SimulationError
from urbanflow.core.npz import ZIP_ERRORS, read_npz, write_npz
from urbanflow.core.types import VehicleStatus
from urbanflow.signals import SignalController
from urbanflow.signals.state import STATE_ARRAYS
from urbanflow.vehicles.table import COLUMNS

if TYPE_CHECKING:
    import os

    from urbanflow.engine.engine import ControllerSpecDict, Engine

__all__ = [
    "COUNTERS",
    "EngineState",
    "check_compatible",
    "config_hash",
    "digest",
    "load_state",
    "restore",
    "save_state",
    "snapshot",
    "state_digest",
]

FORMAT: Final = "urbanflow.state"
VERSION: Final = "1.0"
STATE_JSON: Final = "state.json"
STATE_NPZ: Final = "state.npz"

COUNTERS: Final = (
    "step_count",
    "next_uid",
    "next_seq",
    "generated",
    "arrived",
    "removed",
    "teleported",
    "forced_commits",
    "red_runs",
    "zone_conflicts",
    "safety_cap_violations",
)
"""Digest item 1, in order (``next_seq`` is the commit counter)."""

# name -> dtype of every array of the state (veh.* columns, sig.* runtime, per-lane state)
_ARRAYS: Final[dict[str, DTypeLike]] = {
    **{f"veh.{name}": dtype for name, (dtype, _) in COLUMNS.items()},
    **{f"sig.{name}": dtype for name, dtype in STATE_ARRAYS.items()},
    "sneakers": np.int64,
    "detector_seen": np.float64,
}
_LANE_ARRAYS: Final = ("sneakers", "detector_seen")
# meta keys digested as JSON (item 6); the others are items 1-4 or not digested (the log)
_JSON_PART: Final = (
    "controllers",
    "parked",
    "held",
    "spawners",
    "router",
    "rng",
    "signal_events",
    "commands",
)
_WAITING: Final = VehicleStatus.waiting_insert.code
_RUNNING: Final = VehicleStatus.running.code
_RECORD: Final = np.dtype(
    [(name, np.dtype(dtype).newbyteorder("<")) for name, (dtype, _) in COLUMNS.items()]
)


@dataclass(frozen=True, slots=True, eq=False)
class EngineState:
    """A deep copy of an engine's runtime state (F.5); see the module docstring."""

    scenario_hash: str
    """``Scenario.content_hash`` of the engine's scenario."""
    config: dict[str, Any]
    """The resolved :class:`SimulationConfig` in JSON form."""
    arrays: dict[str, NDArray[Any]]
    """Read-only arrays: ``veh.<column>`` (rows below the high-water mark),
    ``sig.<name>`` (signal runtime), ``sneakers`` and ``detector_seen`` (per lane)."""
    meta: dict[str, Any]
    """JSON-safe rest: counters, slot lists and id maps, routes, queues, the configured
    controllers (checked on restore, not digested), the active and parked controllers,
    held flags, spawners, router, RNG streams, command log."""

    @property
    def step_count(self) -> int:
        """Steps completed when the state was taken."""
        return int(self.meta["counters"]["step_count"])


def config_hash(config: SimulationConfig | Mapping[str, Any]) -> str:
    """sha256 hex of the canonical JSON of a resolved config."""
    data = config.model_dump(mode="json") if isinstance(config, SimulationConfig) else config
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _router_state(router: object) -> Any:
    state_dict = getattr(router, "state_dict", None)
    return state_dict() if callable(state_dict) else {}


def _controller_entry(controller: SignalController, spec: ControllerSpecDict) -> dict[str, Any]:
    return {"type": spec["type"], "params": spec["params"], "state": controller.state_dict()}


# ------------------------------------------------------------------------------- snapshot
def snapshot(engine: Engine, *, with_log: bool = True) -> EngineState:
    """A deep copy of ``engine``'s runtime state; ``with_log=False`` leaves the command log
    out (empty), as :func:`digest` does: it is not digested and grows with the run."""
    vehicles = engine.vehicles.state_dict()
    columns = vehicles.pop("columns")
    signals = engine.signals.state_dict()
    arrays: dict[str, NDArray[Any]] = {f"veh.{n}": a for n, a in columns.items()}
    arrays.update({f"sig.{n}": a for n, a in signals["arrays"].items()})
    arrays["sneakers"] = engine.sneakers.copy()
    arrays["detector_seen"] = engine.detector_seen.copy()
    for arr in arrays.values():
        arr.setflags(write=False)
    ids = engine.network.int_ids
    commands = engine.commands.state_dict(with_log=with_log)
    counters = {
        "step_count": engine.step_count,
        "next_uid": engine.vehicles.next_uid,
        "next_seq": engine._next_seq,
        **{name: int(getattr(engine, name)) for name in COUNTERS[3:]},
    }
    meta = {
        "counters": counters,
        "vehicles": vehicles,
        "routes": engine.routes.state_dict()["routes"],
        "queues": engine.queues.state_dict()["queues"],
        "configured": _configured(engine),
        "controllers": {
            ids[j]: _controller_entry(c, engine.specs[j])
            for j, c in sorted(engine.controllers.items())
        },
        "parked": {
            ids[j]: _controller_entry(c, engine.parked_specs[j])
            for j, c in sorted(engine.parked.items())
        },
        "held": engine.held.tolist(),
        "spawners": [s.state_dict() for s in engine.spawners],
        "router": _router_state(engine.router),
        "rng": engine.rng.state_dict(),
        "signal_events": signals["forced_events"],
        "commands": {"api_count": commands["api_count"], "pending": commands["pending"]},
        "command_log": commands["log"],
    }
    # the JSON round trip deep-copies and gives the in-memory form the file form exactly
    meta = json.loads(json.dumps(meta))
    return EngineState(
        scenario_hash=engine.network.scenario_hash,
        config=engine.config.model_dump(mode="json"),
        arrays=arrays,
        meta=meta,
    )


def _configured(engine: Engine) -> dict[str, Any]:
    """The configured controller ``{type, params}`` of each signalised intersection, by id."""
    ids = engine.network.int_ids
    data = {ids[j]: spec for j, spec in sorted(engine.configured.items())}
    out: dict[str, Any] = json.loads(json.dumps(data))
    return out


# ------------------------------------------------------------------------------- restore
def check_compatible(engine: Engine, state: EngineState) -> None:
    """``SimulationError`` unless ``state`` was taken from the same scenario and config, and
    with the same configured controllers (the ``controllers=`` override: ``reset()`` and
    the results' provenance use the engine's own, so a mismatch is rejected)."""
    short = C.SHORT_HASH_LENGTH
    if state.scenario_hash != engine.network.scenario_hash:
        raise SimulationError(
            f"the snapshot is of another scenario (hash {state.scenario_hash[:short]}); this "
            f"simulation runs {engine.network.scenario_hash[:short]}"
        )
    current = engine.config.model_dump(mode="json")
    if config_hash(state.config) != config_hash(current):
        keys = sorted(set(current) | set(state.config))
        differ = [k for k in keys if current.get(k) != state.config.get(k)]
        raise SimulationError(
            "the snapshot was taken with another simulation config (it differs in: "
            f"{', '.join(differ)}); restore it into a simulation built with the same config"
        )
    mine, theirs = _configured(engine), state.meta.get("configured")
    if theirs != mine:
        saved = theirs if isinstance(theirs, dict) else {}

        def kind(entry: Any) -> str:
            return str(entry.get("type")) if isinstance(entry, dict) else "none"

        where = ", ".join(
            f'"{j}" ({kind(saved.get(j))} in the snapshot, {kind(spec)} here)'
            for j, spec in mine.items()
            if saved.get(j) != spec
        )
        where = where or "(none recorded in the snapshot)"
        raise SimulationError(
            f"the snapshot was taken with other signal controllers configured at {where}; "
            "restore it into a simulation built with the same controllers="
        )


def restore(engine: Engine, state: EngineState) -> None:
    """Copy ``state`` back into ``engine`` (it continues exactly as the snapshotted one).

    ``SimulationError`` if the scenario or config hash differs, or if the state is
    malformed; after a failure past the hash checks the engine is inconsistent (reset or
    restore again).
    """
    check_compatible(engine, state)
    try:
        _restore(engine, state)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SimulationError(f"malformed engine state: {type(exc).__name__}: {exc}") from None


def _restore(engine: Engine, state: EngineState) -> None:
    meta, arrays = state.meta, state.arrays
    counters = meta["counters"]
    engine._step_count = int(counters["step_count"])
    engine._next_seq = int(counters["next_seq"])
    for name in COUNTERS[3:]:
        setattr(engine, name, int(counters[name]))
    columns = {n: arrays[f"veh.{n}"] for n in COLUMNS}
    engine.vehicles.load_state_dict({**meta["vehicles"], "columns": columns})
    engine.routes.load_state_dict({"routes": meta["routes"]})
    engine.queues.load_state_dict({"queues": meta["queues"]})
    engine.signals.load_state_dict(
        {
            "arrays": {n: arrays[f"sig.{n}"] for n in STATE_ARRAYS},
            "forced_events": meta["signal_events"],
        }
    )
    for name in _LANE_ARRAYS:
        target, saved = getattr(engine, name), arrays[name]
        if saved.shape != target.shape:
            raise ValueError(f"{name} has shape {saved.shape}, expected {target.shape}")
        target[:] = saved
    engine.rng.load_state_dict(meta["rng"])  # in place: spawners keep their generators
    if len(meta["spawners"]) != len(engine.spawners):
        raise ValueError(f"{len(meta['spawners'])} spawner states for {len(engine.spawners)}")
    for spawner, spawner_state in zip(engine.spawners, meta["spawners"], strict=True):
        spawner.load_state_dict(spawner_state)
    load = getattr(engine.router, "load_state_dict", None)
    if callable(load):
        load(meta["router"])
    held = np.asarray(meta["held"], dtype=bool)
    if held.shape != engine.held.shape:
        raise ValueError(f"held flags for {held.size} intersections, expected {engine.held.size}")
    engine.held[:] = held
    _restore_controllers(engine, meta["controllers"], meta["parked"])
    engine.commands.load_state_dict({**meta["commands"], "log": meta["command_log"]})
    engine.events.clear()
    engine.holds = None
    _check_values(engine, meta)


def _check_values(engine: Engine, meta: Mapping[str, Any]) -> None:
    """``ValueError`` unless every index of the restored state is in range (shapes and
    dtypes were checked while loading): slot lists, id maps, queues, routes, pending
    command events, and the links, connectors, routes and types of the live rows."""
    net, veh, routes = engine.network, engine.vehicles, engine.routes
    top = veh.top
    live = np.fromiter(veh.id_to_handle.values(), dtype=np.intp, count=len(veh.id_to_handle))
    slots = meta["vehicles"]
    every = [*live.tolist(), *(int(h) for k in ("free", "deferred", "held") for h in slots[k])]
    if any(not 0 <= h < top for h in every) or len(set(every)) != len(every):
        raise ValueError(f"vehicle slots out of range [0, {top}) or listed twice")
    for vid, h in veh.id_to_handle.items():
        u = int(veh.uid[h])
        if veh.ids[h] != vid or u >= len(veh.uid_to_id) or veh.uid_to_id[u] != vid:
            raise ValueError(f'vehicle "{vid}": its handle, id and uid maps disagree')
    status = veh.status[live]
    queued = sorted(int(h) for _, hs in meta["queues"] for h in hs)
    if queued != sorted(live[status == _WAITING].tolist()):
        raise ValueError("the insertion queues do not hold exactly the waiting vehicles")
    run = np.flatnonzero(veh.active[:top])
    if not np.array_equal(run, np.sort(live[status == _RUNNING])):
        raise ValueError("the active flags disagree with the running vehicles")
    roads = [int(r) for r, _ in meta["queues"]] + [r for x in routes.routes for r in x.tolist()]
    if any(not 0 <= r < net.n_roads for r in roads):
        raise ValueError("an insertion queue or a route names a road out of range")
    rid = veh.route_id[live].astype(np.intp)
    if ((rid < 0) | (rid >= len(routes))).any() or (veh.type_idx[live] >= len(engine.types)).any():
        raise ValueError("a vehicle's route or type index is out of range")
    link = veh.link[run].astype(np.intp)
    cursor = veh.route_cursor[run].astype(np.intp)
    conn = np.r_[veh.next_conn[run], veh.lock_conn[run]].astype(np.intp)
    if (
        ((link < 0) | (link >= net.n_links) | (cursor < 0)).any()
        or (cursor >= routes.lengths[veh.route_id[run]]).any()
        or ((conn != -1) & ((conn < net.n_lanes) | (conn >= net.n_links))).any()
    ):
        raise ValueError("a running vehicle's link, connector or route cursor is out of range")
    for _, _, h, _, lk in meta["commands"]["pending"]:
        if not (0 <= int(h) < top and -1 <= int(lk) < net.n_links):
            raise ValueError("a pending command event names a slot or link out of range")


def _restore_controllers(
    engine: Engine, active: Mapping[str, Any], parked: Mapping[str, Any]
) -> None:
    """Reuse the engine's controller instances whose ``{type, params}`` match the state (so
    unregistered factories work), re-create the others from the registry, then load each
    ``state_dict``."""
    index, ids = engine.network.int_index, engine.network.int_ids
    pool: dict[int, list[tuple[SignalController, ControllerSpecDict]]] = {
        j: [(c, engine.specs[j])] for j, c in engine.controllers.items()
    }
    for j, c in engine.parked.items():
        pool[j].append((c, engine.parked_specs[j]))
    if sorted(index[k] for k in active) != sorted(engine.controllers):
        raise ValueError("the state's signalised intersections differ from the engine's")

    def take(j: int, entry: Mapping[str, Any]) -> tuple[SignalController, ControllerSpecDict]:
        spec: ControllerSpecDict = {"type": entry["type"], "params": entry["params"]}
        match = next((k for k, (_, have) in enumerate(pool[j]) if have == spec), None)
        if match is not None:
            controller, spec = pool[j].pop(match)
        else:
            try:
                controller, spec = engine.make_controller(j, dict(spec))
            except (NotFoundError, ConfigError) as exc:
                raise SimulationError(
                    f'cannot restore the signal controller "{spec["type"]}" of intersection '
                    f'"{ids[j]}": {exc}'
                ) from None
        controller.load_state_dict(entry["state"])
        return controller, spec

    engine.parked, engine.parked_specs = {}, {}
    for key in sorted(active, key=index.__getitem__):
        j = index[key]
        engine.controllers[j], engine.specs[j] = take(j, active[key])
    for key in sorted(parked, key=index.__getitem__):
        j = index[key]
        engine.parked[j], engine.parked_specs[j] = take(j, parked[key])


# ------------------------------------------------------------------------------- digest
def _canonical(values: NDArray[Any]) -> NDArray[Any]:
    """Little-endian, with every NaN replaced by the one canonical NaN."""
    if values.dtype.kind == "f":
        values = np.where(np.isnan(values), np.nan, values).astype(values.dtype)
    return values.astype(values.dtype.newbyteorder("<"), copy=False)


def state_digest(state: EngineState) -> str:
    """sha256 hex of ``state`` in the canonical byte order of the module docstring."""
    meta, arrays = state.meta, state.arrays
    h = hashlib.sha256()
    counters = meta["counters"]
    h.update(np.array([counters[k] for k in COUNTERS], dtype="<i8").tobytes())
    uid = arrays["veh.uid"]
    live = np.array([hd for _, hd in meta["vehicles"]["id_to_handle"]], dtype=np.intp)
    live = live[np.argsort(uid[live], kind="stable")]
    rows = np.zeros(live.size, dtype=_RECORD)
    for name in COLUMNS:
        rows[name] = _canonical(arrays[f"veh.{name}"][live])
    h.update(rows.tobytes())
    for roads in meta["routes"]:
        h.update(np.array([len(roads)], dtype="<i8").tobytes())
        h.update(np.array(roads, dtype="<i4").tobytes())
    for road, handles in meta["queues"]:
        h.update(np.array([road, len(handles)], dtype="<i8").tobytes())
        h.update(uid[np.array(handles, dtype=np.intp)].astype("<u4").tobytes())
    for name in (*(f"sig.{n}" for n in STATE_ARRAYS), *_LANE_ARRAYS):
        h.update(_canonical(arrays[name]).tobytes())
    doc = {key: meta[key] for key in _JSON_PART}
    h.update(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def digest(engine: Engine) -> str:
    """:func:`state_digest` of the engine's current state."""
    # ponytail: hashes a snapshot without the command log (copies included, ~1 ms at 50
    # vehicles); hash the live table in place if per-step digests of very large runs matter
    return state_digest(snapshot(engine, with_log=False))


# ------------------------------------------------------------------------------- files
def save_state(path: str | os.PathLike[str], state: EngineState) -> Path:
    """Write ``state`` as a zip of ``state.npz`` (safe npz, no pickle) and ``state.json``
    (format, version, scenario hash, config and the JSON half); atomic (tmp + replace)."""
    target = Path(path)
    npz = io.BytesIO()
    write_npz(npz, state.arrays, C.STATE_COMPRESSLEVEL)
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "urbanflow": __version__,
        "scenario_hash": state.scenario_hash,
        "config": state.config,
        "meta": state.meta,
    }
    tmp = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(STATE_JSON, json.dumps(doc))
        zf.writestr(STATE_NPZ, npz.getvalue())
    tmp.replace(target)
    return target


def load_state(path: str | os.PathLike[str]) -> EngineState:
    """Read a :func:`save_state` file. ``NotFoundError`` if it does not exist,
    ``SimulationError`` if it is not a valid engine state (nothing is unpickled: the arrays
    go through :func:`~urbanflow.core.npz.read_npz`)."""
    source = Path(path)
    if not source.is_file():
        raise NotFoundError(f"no saved state at {source}")
    try:
        with zipfile.ZipFile(source) as zf:
            names = sorted(zf.namelist())
            if names != sorted((STATE_JSON, STATE_NPZ)):
                raise ValueError(f"expected the members {STATE_JSON} and {STATE_NPZ}, got {names}")
            declared = sum(info.file_size for info in zf.infolist())
            if declared > C.NPZ_MAX_BYTES:
                raise ValueError(f"members declare {declared} bytes (limit {C.NPZ_MAX_BYTES})")
            doc = json.loads(zf.read(STATE_JSON))
            npz = zf.read(STATE_NPZ)
        if not isinstance(doc, dict) or doc.get("format") != FORMAT:
            raise ValueError(f'"format" is not "{FORMAT}"')
        major = str(doc.get("version", "")).partition(".")[0]
        if major != VERSION.partition(".")[0]:
            raise ValueError(f"version {doc.get('version')!r} is not supported (1.x)")
        arrays = read_npz(npz, _ARRAYS, C.STATE_MAX_ELEMS)
        state = EngineState(
            scenario_hash=str(doc["scenario_hash"]),
            config=dict(doc["config"]),
            arrays=arrays,
            meta=dict(doc["meta"]),
        )
        _ = state.step_count  # the counters exist
    except (*ZIP_ERRORS, KeyError, TypeError, ValueError) as exc:
        why = f"{type(exc).__name__}: {exc}"
        raise SimulationError(f"{source} is not a valid UrbanFlow state file ({why})") from None
    for arr in arrays.values():
        arr.setflags(write=False)
    return state
