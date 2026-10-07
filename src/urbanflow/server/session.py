"""Live simulation sessions for the web workbench (plan N, simplified).

A :class:`Session` owns one :class:`~urbanflow.Simulation` and an asyncio task that paces
it: every display tick it advances the due number of steps in a worker thread (so the event
loop stays responsive) and sends each subscriber the ordered text messages (vehicle registry
deltas, status, metrics, selection) followed by one binary UFB1 frame.

ponytail: sessions run in threads of the server process with one lock per session (the plan
uses one spawned process per session); upgrade to process workers if many heavy sessions
must run concurrently.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from urbanflow.core.errors import CommandError, UrbanFlowError
from urbanflow.replay import ReplayReader
from urbanflow.simulation import Simulation
from urbanflow.visualization import LaneMetric, RegistryTracker, decode_frame, encode_frame

__all__ = ["ClientMessage", "ReplaySession", "Session", "SessionInfo"]

DISPLAY_FPS: Final = 20.0
METRICS_HZ: Final = 2.0
STATUS_HZ: Final = 1.0
MAX_STEPS_PER_TICK: Final = 2000
MAX_SPS: Final = 10_000.0
SEND_TIMEOUT_S: Final = 5.0
State = Literal["paused", "playing", "ended", "error"]


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Play(_Msg):
    type: Literal["play"]


class Pause(_Msg):
    type: Literal["pause"]


class Step(_Msg):
    type: Literal["step"]
    n: int = Field(default=1, ge=-10_000, le=10_000)
    """Negative steps go back (replay sessions only)."""


class Seek(_Msg):
    type: Literal["seek"]
    step: int = Field(ge=0)


class SetSpeed(_Msg):
    type: Literal["set_speed"]
    steps_per_second: float | None = Field(default=10.0, ge=-MAX_SPS, le=MAX_SPS)
    """``None`` = as fast as possible; negative rewinds (replay sessions only)."""


class Reset(_Msg):
    type: Literal["reset"]
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class SetController(_Msg):
    type: Literal["set_controller"]
    controller: str = Field(min_length=1, max_length=64)
    intersection: str | None = Field(default=None, max_length=128)
    """``None`` = every signalised intersection."""


class HoldPhase(_Msg):
    type: Literal["hold_phase"]
    intersection: str = Field(max_length=128)
    phase: int = Field(ge=0, le=64)


class ReleasePhase(_Msg):
    type: Literal["release_phase"]
    intersection: str = Field(max_length=128)


class Subscribe(_Msg):
    type: Literal["subscribe"]
    lane_metric: int = Field(default=0, ge=0, le=max(LaneMetric))


class Select(_Msg):
    type: Literal["select"]
    kind: Literal["vehicle", "intersection", "none"]
    id: str = Field(default="", max_length=128)


ClientMessage = (
    Play | Pause | Step | Seek | SetSpeed | Reset | SetController | HoldPhase | ReleasePhase
    | Subscribe | Select
)  # fmt: skip
_ADAPTER: Final[TypeAdapter[ClientMessage]] = TypeAdapter(ClientMessage)


class SessionInfo(BaseModel):
    """REST view of a session."""

    id: str
    label: str
    scenario: str
    state: State
    step: int
    time: float
    end_time: float | None
    dt: float
    steps_per_second: float | None
    vehicles: int
    controllers: dict[str, str]
    subscribers: int


@dataclass(eq=False)
class _Client:
    ws: WebSocket
    tracker: RegistryTracker = field(default_factory=RegistryTracker)
    fresh: bool = True


class Session:
    """One live simulation plus its subscribers."""

    kind: str = "live"
    controllers: dict[str, str]

    def __init__(self, sim: Simulation, label: str, scenario: str) -> None:
        self.sim = sim
        self._init_common(label, scenario)
        self.controllers = {j: str(sim.signals[j].controller) for j in sim.signals.ids}

    def _init_common(self, label: str, scenario: str) -> None:
        self.id = secrets.token_hex(8)
        self.label = label
        self.scenario = scenario
        self.state: State = "paused"
        self.error = ""
        self.sps: float | None = 10.0
        self.lane_metric = LaneMetric.none
        self.selection: tuple[str, str] = ("none", "")
        self.clients: list[_Client] = []
        self._lock = asyncio.Lock()
        self._seq = 0
        self._sent_step = -1
        self._next_metrics = 0.0
        self._next_status = 0.0
        self._closed = False
        self._task: asyncio.Task[None] | None = None
        self.last_activity = time.monotonic()
        self.controllers = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the pacing task (in the running event loop)."""
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def close(self) -> None:
        """Stop the task and disconnect every subscriber."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        for c in list(self.clients):
            with contextlib.suppress(Exception):
                await c.ws.close(code=4410)
        self.clients.clear()
        self._release()

    def _release(self) -> None:
        self.sim.close()

    def info(self) -> SessionInfo:
        """The REST view."""
        sim = self.sim
        return SessionInfo(
            id=self.id,
            label=self.label,
            scenario=self.scenario,
            state=self.state,
            step=sim.step_count,
            time=sim.time,
            end_time=sim.end_time,
            dt=sim.dt,
            steps_per_second=self.sps,
            vehicles=len(sim.vehicles),
            controllers=dict(self.controllers),
            subscribers=len(self.clients),
        )

    # ------------------------------------------------------------------ pacing
    async def _loop(self) -> None:
        budget, last = 0.0, time.perf_counter()
        while not self._closed:
            await asyncio.sleep(1.0 / DISPLAY_FPS)
            now = time.perf_counter()
            if self.state == "playing":
                if self.sps is None:
                    n = MAX_STEPS_PER_TICK
                else:
                    rate = abs(self.sps)
                    budget = min(budget + (now - last) * rate, rate)  # no burst
                    n = int(budget)
                    budget -= n
                if n:
                    await self._advance(min(n, MAX_STEPS_PER_TICK), deadline=1.0 / DISPLAY_FPS)
            else:
                budget = 0.0
            last = now
            await self._flush()

    async def _advance(self, n: int, *, deadline: float | None = None) -> None:
        async with self._lock:
            await asyncio.to_thread(self._step_n, n, deadline)

    def _step_n(self, n: int, deadline: float | None) -> None:
        sim, start = self.sim, time.perf_counter()
        try:
            for _ in range(n):
                if sim.done:
                    break
                sim.step()
                if deadline is not None and time.perf_counter() - start > deadline:
                    break
        except UrbanFlowError as exc:
            self.state, self.error = "error", str(exc)
        if sim.done and self.state != "error":
            self.state = "ended"

    # ------------------------------------------------------------------ sending
    def _frame_bytes(self) -> tuple[bytes, np.ndarray]:
        frame = self.sim.frame(lane_metric=int(self.lane_metric))
        self._seq += 1
        crc = self.sim.network.compiled.geometry_crc
        return encode_frame(frame, self._seq, crc), frame.uid

    async def _flush(self, *, force: bool = False) -> None:
        if not self.clients:
            return
        now = time.monotonic()
        async with self._lock:
            changed = (
                self._current_step() != self._sent_step
                or force
                or any(c.fresh for c in self.clients)
            )
            texts: list[dict[str, Any]] = []
            if now >= self._next_status or force:
                texts.append({"type": "status", **self._status()})
                self._next_status = now + 1.0 / STATUS_HZ
            if now >= self._next_metrics and changed:
                texts.append({"type": "metrics", "values": _finite(self._latest_metrics())})
                sel = self._selection()
                if sel is not None:
                    texts.append(sel)
                self._next_metrics = now + 1.0 / METRICS_HZ
            if not changed and not texts:
                return
            frame, uids = await asyncio.to_thread(self._frame_bytes) if changed else (b"", None)
            deltas: list[tuple[np.ndarray, np.ndarray, bool] | None] = []
            for c in self.clients:
                if uids is None:
                    deltas.append(None)
                    continue
                added, removed = c.tracker.delta(uids, reset=c.fresh)
                deltas.append((added, removed, c.fresh))
            wanted = sorted({int(u) for d in deltas if d is not None for u in d[0].tolist()})
            meta = self._meta(wanted)
            self._sent_step = self._current_step()
        for c, d in zip(list(self.clients), deltas, strict=True):
            msgs = list(texts)
            if d is not None:
                added, removed, reset = d
                if removed.size:
                    msgs.append({"type": "vehicles_removed", "uids": removed.tolist()})
                if added.size or reset:
                    rows = [meta[int(u)] for u in added.tolist()]
                    msgs.append(
                        {
                            "type": "vehicles_added",
                            "reset": reset,
                            "uids": added.tolist(),
                            "ids": [r[0] for r in rows],
                            "types": [r[1] for r in rows],
                            "dests": [r[2] for r in rows],
                        }
                    )
                c.fresh = False
            try:
                for m in msgs:
                    await asyncio.wait_for(c.ws.send_json(m), SEND_TIMEOUT_S)
                if frame:
                    await asyncio.wait_for(c.ws.send_bytes(frame), SEND_TIMEOUT_S)
            except Exception:
                self.detach(c.ws)

    def _current_step(self) -> int:
        return self.sim.step_count

    def _latest_metrics(self) -> dict[str, float]:
        return self.sim.metrics.latest()

    def _meta(self, uids: list[int]) -> dict[int, tuple[str, int, int]]:
        if not uids:
            return {}
        rows = zip(*self.sim.vehicle_meta(uids), strict=True)
        return dict(zip(uids, rows, strict=True))

    def _status(self) -> dict[str, Any]:
        sim = self.sim
        return {
            "kind": self.kind,
            "state": self.state,
            "step": sim.step_count,
            "time": sim.time,
            "end_time": sim.end_time,
            "steps_per_second": self.sps,
            "vehicles": len(sim.vehicles),
            "lane_metric": int(self.lane_metric),
            "controllers": dict(self.controllers),
            "error": self.error,
        }

    def _selection(self) -> dict[str, Any] | None:
        kind, ident = self.selection
        sim = self.sim
        if kind == "vehicle":
            v = sim.vehicles.get(ident)
            if v is None:
                return {"type": "selection", "kind": kind, "id": ident, "available": False}
            detail = {
                k: getattr(v, k)
                for k in (
                    "type", "status", "road", "lane", "speed", "desired_speed",
                    "acceleration", "waiting_time", "stops", "distance", "travel_time",
                    "destination", "route", "route_index",
                )
            }  # fmt: skip
            return {
                "type": "selection",
                "kind": kind,
                "id": ident,
                "available": True,
                "detail": _jsonable(detail),
            }
        if kind == "intersection" and ident in sim.signals.ids:
            s = sim.signals[ident]
            phases = [
                {"index": p.index, "id": p.id, "duration": p.duration}
                for p in sim.signals.phases(ident)
            ]
            return {
                "type": "selection",
                "kind": kind,
                "id": ident,
                "available": True,
                "detail": _jsonable(
                    {
                        "phase_index": s.phase_index,
                        "phase_id": s.phase_id,
                        "stage": s.stage,
                        "remaining": s.remaining,
                        "cycle": s.cycle,
                        "held": s.held,
                        "controller": s.controller,
                        "state_string": s.state_string,
                        "movement_states": s.movement_states,
                        "phases": phases,
                    }
                ),
            }
        return None

    # ------------------------------------------------------------------ clients
    async def attach(self, ws: WebSocket) -> None:
        """Register a subscriber and send it ``hello`` (it gets a full registry next)."""
        self.clients.append(_Client(ws))
        self.last_activity = time.monotonic()
        await ws.send_json({"type": "hello", "protocol": 1, "session": self.info().model_dump()})
        await self._flush(force=True)

    def detach(self, ws: WebSocket) -> None:
        """Forget a subscriber."""
        self.clients = [c for c in self.clients if c.ws is not ws]
        self.last_activity = time.monotonic()

    async def handle(self, ws: WebSocket, raw: str | bytes) -> None:
        """Apply one client command; answers ``error`` on invalid input."""
        self.last_activity = time.monotonic()
        try:
            msg = _ADAPTER.validate_json(raw)
        except ValidationError as exc:
            await ws.send_json({"type": "error", "code": "invalid_message", "message": _first(exc)})
            return
        try:
            await self._apply(msg)
        except UrbanFlowError as exc:
            await ws.send_json({"type": "error", "code": "command_failed", "message": str(exc)})
            return
        await self._flush(force=True)

    async def _apply(self, msg: ClientMessage) -> None:
        sim = self.sim
        match msg:
            case Play():
                if self.state == "paused":
                    self.state = "playing"
            case Pause():
                if self.state == "playing":
                    self.state = "paused"
            case Step(n=n) if n < 0:
                raise CommandError("stepping back needs a replay session")
            case Seek():
                raise CommandError("seeking needs a replay session")
            case SetSpeed(steps_per_second=sps) if sps is not None and sps <= 0:
                raise CommandError("rewinding needs a replay session; speed must be > 0")
            case Step(n=n):
                if self.state in ("paused", "playing"):
                    self.state = "paused"
                    await self._advance(n)
            case SetSpeed(steps_per_second=sps):
                self.sps = sps
            case Reset(seed=seed):
                async with self._lock:
                    await asyncio.to_thread(sim.reset, seed)
                    self.state, self.error = "paused", ""
                    for c in self.clients:
                        c.fresh = True
                    self.controllers = {j: str(sim.signals[j].controller) for j in sim.signals.ids}
            case SetController(controller=name, intersection=where):
                targets = sim.signals.ids if where is None else (where,)
                async with self._lock:
                    for j in targets:
                        sim.signals.set_controller(j, name)
                        self.controllers[j] = name
            case HoldPhase(intersection=j, phase=p):
                async with self._lock:
                    sim.signals.hold_phase(j, p)
            case ReleasePhase(intersection=j):
                async with self._lock:
                    sim.signals.release(j)
            case Subscribe(lane_metric=code):
                self.lane_metric = LaneMetric(code)
            case Select(kind=kind, id=ident):
                self.selection = (kind, ident)
                self._next_metrics = 0.0


class ReplaySession(Session):
    """Plays a recorded ``.ufr`` with the live protocol plus seek, step back and rewind."""

    kind = "replay"

    def __init__(self, reader: ReplayReader, label: str, name: str) -> None:
        self.reader = reader
        self._init_common(label, name)
        recorded = reader.manifest.get("controllers", {})
        self.controllers = {str(k): str(v) for k, v in recorded.items()}
        self._index = 0
        self._dt = float(reader.manifest.get("dt", 1.0))
        self._vehicles = reader.vehicles()
        self._series = reader.timeseries()
        geo = json.loads(reader.geometry_json())
        self._road_ids: list[str] = list(geo["roads"]["id"])
        self._mov_ids: list[str] = list(geo["movements"]["id"])
        self._mov_int: list[str] = list(geo["movements"]["intersection"])
        self._last: bytes = b""

    def _release(self) -> None:
        self.reader.close()

    def info(self) -> SessionInfo:
        """The REST view."""
        steps = self.reader.steps
        return SessionInfo(
            id=self.id,
            label=self.label,
            scenario=self.scenario,
            state=self.state,
            step=self._current_step(),
            time=self._current_step() * self._dt,
            end_time=steps[-1] * self._dt if steps else None,
            dt=self._dt,
            steps_per_second=self.sps,
            vehicles=self._count(),
            controllers=dict(self.controllers),
            subscribers=len(self.clients),
        )

    def _count(self) -> int:
        return int.from_bytes(self._last[32:36], "little") if self._last else 0

    def _current_step(self) -> int:
        return self.reader.steps[self._index] if self.reader.steps else 0

    def _latest_metrics(self) -> dict[str, float]:
        table = self._series
        if not table or not table["time"].size:
            return {}
        t = self._current_step() * self._dt
        k = int(np.searchsorted(table["time"], t + 1e-9, side="right")) - 1
        if k < 0:
            return {}
        return {name: float(col[k]) for name, col in table.items()}

    def _meta(self, uids: list[int]) -> dict[int, tuple[str, int, int]]:
        return {u: self._vehicles.get(u, ("", -1, -1)) for u in uids}

    def _step_n(self, n: int, deadline: float | None) -> None:
        del deadline
        direction = -1 if self.sps is not None and self.sps < 0 else 1
        self._move(direction * n)

    def _move(self, frames: int) -> None:
        last = len(self.reader.steps) - 1
        target = min(max(self._index + frames, 0), last)
        if target != self._index + frames and self.state == "playing":
            self.state = "paused"  # reached an end
        self._index = target

    def _frame_bytes(self) -> tuple[bytes, np.ndarray]:
        data = self.reader.frame_bytes(self._index)
        self._last = data
        n = int.from_bytes(data[32:36], "little")
        header = int.from_bytes(data[6:8], "little")
        return data, np.frombuffer(data, dtype="<u4", count=n, offset=header)

    def _status(self) -> dict[str, Any]:
        steps = self.reader.steps
        return {
            "kind": self.kind,
            "state": self.state,
            "step": self._current_step(),
            "time": self._current_step() * self._dt,
            "end_time": steps[-1] * self._dt if steps else None,
            "first_step": steps[0] if steps else 0,
            "last_step": steps[-1] if steps else 0,
            "steps_per_second": self.sps,
            "vehicles": self._count(),
            "lane_metric": 0,
            "controllers": dict(self.controllers),
            "error": self.error,
        }

    def _selection(self) -> dict[str, Any] | None:
        kind, ident = self.selection
        if kind == "none" or not self._last:
            return None
        frame = decode_frame(self._last)[0]
        if kind == "vehicle":
            uid = next((u for u, m in self._vehicles.items() if m[0] == ident), None)
            hit = np.flatnonzero(frame.uid == uid) if uid is not None else np.zeros(0)
            if uid is None or not hit.size:
                return {"type": "selection", "kind": kind, "id": ident, "available": False}
            i = int(hit[0])
            _vid, vtype, dest = self._vehicles[uid]
            road = self._road_ids[dest] if 0 <= dest < len(self._road_ids) else None
            detail = {"type": vtype, "speed": float(frame.speed[i]), "destination": road}
            return {
                "type": "selection",
                "kind": kind,
                "id": ident,
                "available": True,
                "detail": _jsonable(detail),
            }
        codes = "rygG"
        signals = frame.signals
        states = {
            m: codes[int(signals[k])]
            for k, m in enumerate(self._mov_ids)
            if signals is not None and self._mov_int[k] == ident and signals[k] < len(codes)
        }
        return {
            "type": "selection",
            "kind": kind,
            "id": ident,
            "available": bool(states),
            "detail": {
                "controller": self.controllers.get(ident, "unsignalised"),
                "movement_states": states,
                "phases": [],
            },
        }

    async def _apply(self, msg: ClientMessage) -> None:
        match msg:
            case Play():
                at_end = self._index >= len(self.reader.steps) - 1
                if at_end and (self.sps is None or self.sps > 0):
                    self._index = 0
                self.state = "playing"
            case Pause():
                self.state = "paused"
            case Step(n=n):
                self.state = "paused"
                self._move(n)
            case Seek(step=step):
                self._index = self.reader.index_at(step)
            case SetSpeed(steps_per_second=sps):
                self.sps = sps
            case Reset():
                self.state, self._index = "paused", 0
            case Subscribe():
                self.lane_metric = LaneMetric.none  # replays carry no lane values
            case Select(kind=kind, id=ident):
                self.selection = (kind, ident)
                self._next_metrics = 0.0
            case _:
                raise CommandError("signal control is not available while replaying a recording")


def _first(exc: ValidationError) -> str:
    err = exc.errors()[0]
    where = ".".join(str(p) for p in err["loc"])
    return f"{where}: {err['msg']}" if where else err["msg"]


def _finite(values: dict[str, float]) -> dict[str, float | None]:
    return {k: (v if math.isfinite(v) else None) for k, v in values.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if hasattr(value, "value") and isinstance(value.value, str):  # StrEnum
        return value.value
    return value
