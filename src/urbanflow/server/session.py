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
import math
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from urbanflow.core.errors import UrbanFlowError
from urbanflow.simulation import Simulation
from urbanflow.visualization import LaneMetric, RegistryTracker, encode_frame

__all__ = ["ClientMessage", "Session", "SessionInfo"]

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
    n: int = Field(default=1, ge=1, le=10_000)


class SetSpeed(_Msg):
    type: Literal["set_speed"]
    steps_per_second: float | None = Field(default=10.0, gt=0, le=MAX_SPS)
    """``None`` = as fast as possible."""


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
    Play | Pause | Step | SetSpeed | Reset | SetController | HoldPhase | ReleasePhase
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

    def __init__(self, sim: Simulation, label: str, scenario: str) -> None:
        self.id = secrets.token_hex(8)
        self.sim = sim
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
        self.controllers = {j: str(sim.signals[j].controller) for j in sim.signals.ids}

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
                    budget = min(budget + (now - last) * self.sps, self.sps)  # no burst
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
            sim = self.sim
            changed = (
                sim.step_count != self._sent_step or force or any(c.fresh for c in self.clients)
            )
            texts: list[dict[str, Any]] = []
            if now >= self._next_status or force:
                texts.append({"type": "status", **self._status()})
                self._next_status = now + 1.0 / STATUS_HZ
            if now >= self._next_metrics and changed:
                texts.append({"type": "metrics", "values": _finite(sim.metrics.latest())})
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
            meta = dict(zip(wanted, zip(*sim.vehicle_meta(wanted), strict=True), strict=True))
            self._sent_step = sim.step_count
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

    def _status(self) -> dict[str, Any]:
        sim = self.sim
        return {
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
                    "destination", "route",
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
