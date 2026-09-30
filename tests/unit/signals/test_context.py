"""ControllerContext: lazy fields computed once, values equal to hand-built references (H.3)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from urbanflow import bundled
from urbanflow.core import constants as C
from urbanflow.core.types import IntArray, Stage, VehicleClass
from urbanflow.engine import Engine
from urbanflow.engine.signalling import LaneStats, detector_occupancy
from urbanflow.network import CompiledNetwork
from urbanflow.scenario import Scenario, ScenarioBuilder
from urbanflow.signals import ControllerContext, FixedTime, SignalProgram

FIELDS = (
    "in_lanes",
    "lane_count",
    "lane_halting",
    "lane_queue",
    "lane_waiting",
    "detector_occupied",
    "detector_last_seen",
    "approaching",
    "connectors",
    "phase_movements",
    "phase_conn",
    "movement_count",
    "movement_halting",
    "movement_out_count",
    "conn_in_count",
    "conn_out_count",
    "emergency_approach",
)


class CountingLanes:
    """A LaneData that counts how often each field is read."""

    def __init__(self, net: CompiledNetwork) -> None:
        self.net = net
        self.reads: Counter[str] = Counter()

    def _read(self, name: str, n: int, dtype: Any = np.int64) -> Any:
        self.reads[name] += 1
        return np.arange(n).astype(dtype)

    count = property(lambda self: self._read("count", self.net.n_links))
    halting = property(lambda self: self._read("halting", self.net.n_links))
    queue = property(lambda self: self._read("queue", self.net.n_lanes))
    waiting = property(lambda self: self._read("waiting", self.net.n_lanes, float))
    detector_occupied = property(lambda self: self._read("occupied", self.net.n_lanes, bool))
    detector_last_seen = property(lambda self: self._read("last_seen", self.net.n_lanes, float))
    approaching = property(lambda self: self._read("approaching", self.net.n_lanes))
    planned = property(lambda self: self._read("planned", self.net.n_conn))
    planned_halting = property(lambda self: self._read("planned_halting", self.net.n_conn))

    @property
    def emergency(self) -> tuple[IntArray, IntArray, np.ndarray]:
        self.reads["emergency"] += 1
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, np.zeros(0)


def _context(prog: SignalProgram, lanes: Any, **timing: Any) -> ControllerContext:
    args: dict[str, Any] = {
        "time": 12.0,
        "dt": 1.0,
        "program": prog,
        "phase": 1,
        "stage": Stage.green,
        "green_elapsed": 7.0,
        "stage_elapsed": 7.0,
        "lanes": lanes,
        "rng": np.random.default_rng(0),
    }
    return ControllerContext(**(args | timing))


def test_timing_fields_and_lazy_heavy_fields(
    make_engine: Callable[..., Engine],
) -> None:
    e = make_engine(Scenario.load(bundled("single_intersection")))
    net, prog = e.network, e.signals.programs[0]
    lanes = CountingLanes(net)
    ctx = _context(prog, lanes)
    assert (ctx.time, ctx.dt, ctx.intersection, ctx.phase, ctx.stage) == (
        12.0,
        1.0,
        prog.intersection,
        1,
        Stage.green,
    )
    assert (ctx.green_elapsed, ctx.stage_elapsed, ctx.n_phases) == (7.0, 7.0, 2)
    assert (ctx.min_green, ctx.max_green) == (5.0, 60.0)
    assert not lanes.reads  # nothing heavy is computed up front
    first = ctx.lane_count
    assert ctx.lane_count is first and lanes.reads == {"count": 1}
    ctx.conn_in_count, ctx.conn_out_count, ctx.movement_out_count  # noqa: B018
    assert lanes.reads == {"count": 4}  # once per field that needs it
    for name in FIELDS:
        getattr(ctx, name)
        getattr(ctx, name)
    assert max(lanes.reads.values()) <= 4 and all(
        v == 1 for k, v in lanes.reads.items() if k != "count"
    )


def test_lane_stats_fields_are_computed_once(make_engine: Callable[..., Engine]) -> None:
    e = make_engine(Scenario.load(bundled("single_intersection")))
    for _ in range(60):
        e.step()
    run = e.vehicles.running()
    occupied = detector_occupancy(e.network, e.vehicles, run)
    stats = LaneStats(e.network, e.vehicles, e.types, run, occupied, np.zeros(e.network.n_lanes))
    for name in ("count", "halting", "queue", "waiting", "approaching", "planned"):
        assert getattr(stats, name) is getattr(stats, name), name
    assert stats.detector_occupied is occupied


class Recorder(FixedTime):
    """fixed_time that keeps a copy of every context field it is shown."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[int, dict[str, Any]] = {}

    def decide(self, ctx: ControllerContext) -> int | None:
        step = round(ctx.time / ctx.dt)
        self.seen[step] = {name: getattr(ctx, name) for name in FIELDS}
        self.seen[step]["rng"] = ctx.rng
        return super().decide(ctx)


def _reference(e: Engine, prog: SignalProgram, last_seen: dict[int, float]) -> dict[str, Any]:
    """Every context field from plain loops over the vehicles (state at a step's start)."""
    net, veh, types = e.network, e.vehicles, e.types
    j, t = prog.intersection, e.time
    run = veh.running().tolist()
    in_lanes = net.int_in_lanes[net.int_in_ptr[j] : net.int_in_ptr[j + 1]].tolist()
    movs = prog.movements.tolist()
    conns = [c for m in movs for c in net.mov_conn[net.mov_conn_ptr[m] : net.mov_conn_ptr[m + 1]]]
    conn_mov = {c: k for k, m in enumerate(movs) for c in conns if net.link_movement[c] == m}
    on = {lk: [h for h in run if veh.link[h] == lk] for lk in range(net.n_links)}

    def queue(lane: int) -> int:
        hs = sorted(on[lane], key=lambda h: -veh.pos[h])
        if not hs or net.link_length[lane] - veh.pos[hs[0]] > C.QUEUE_FRONT_TOLERANCE_M:
            return 0
        n = 0
        for h in hs:
            if not veh.halting[h]:
                break
            n += 1
        return n

    def occupied(lane: int) -> bool:
        start = max(0.0, net.link_length[lane] - C.DETECTOR_LENGTH)
        hang = [
            h
            for c in net.lane_out_conn[net.lane_out_ptr[lane] : net.lane_out_ptr[lane + 1]]
            for h in on[c]
            if veh.pos[h] < veh.length[h]
        ]
        return any(veh.pos[h] >= start for h in on[lane]) or bool(hang)

    for lane in range(net.n_lanes):
        if occupied(lane):
            last_seen[lane] = t
    planning = {
        c: [h for h in run if veh.link[h] < net.n_lanes and veh.next_conn[h] == c] for c in conns
    }
    emergency = []
    for h in run:
        if types.vclass[veh.type_idx[h]] != VehicleClass.emergency.code:
            continue
        lk = int(veh.link[h])
        if lk in in_lanes:
            c = int(veh.next_conn[h])
            emergency.append((lk, conn_mov.get(c, -1), float(net.link_length[lk] - veh.pos[h])))
        elif lk in conn_mov:
            emergency.append(
                (int(net.conn_from_lane[lk - net.n_lanes]), conn_mov[lk], -float(veh.pos[h]))
            )
    emergency.sort(key=lambda x: (x[2], x[0]))
    green = prog.phase_state >= 2
    return {
        "in_lanes": in_lanes,
        "lane_count": [len(on[lane]) for lane in in_lanes],
        "lane_halting": [sum(bool(veh.halting[h]) for h in on[lane]) for lane in in_lanes],
        "lane_queue": [queue(lane) for lane in in_lanes],
        "lane_waiting": [
            sum(float(veh.link_waiting_time[h]) for h in on[lane]) for lane in in_lanes
        ],
        "detector_occupied": [occupied(lane) for lane in in_lanes],
        "detector_last_seen": [t - last_seen.get(lane, -np.inf) for lane in in_lanes],
        "approaching": [
            sum(
                not veh.halting[h] and net.link_length[lane] - veh.pos[h] <= C.APPROACH_DISTANCE_M
                for h in on[lane]
            )
            for lane in in_lanes
        ],
        "connectors": conns,
        "phase_movements": green.tolist(),
        "phase_conn": green[:, [conn_mov[c] for c in conns]].tolist(),
        "movement_count": [
            sum(len(planning[c]) for c in conns if conn_mov[c] == k) for k in range(len(movs))
        ],
        "movement_halting": [
            sum(sum(bool(veh.halting[h]) for h in planning[c]) for c in conns if conn_mov[c] == k)
            for k in range(len(movs))
        ],
        "movement_out_count": [
            sum(
                len(on[lane])
                for lane in {
                    int(net.conn_to_lane[c - net.n_lanes]) for c in conns if conn_mov[c] == k
                }
            )
            for k in range(len(movs))
        ],
        "conn_in_count": [len(on[int(net.conn_from_lane[c - net.n_lanes])]) for c in conns],
        "conn_out_count": [len(on[int(net.conn_to_lane[c - net.n_lanes])]) for c in conns],
        "emergency_approach": emergency,
    }


def test_context_values_equal_hand_built_references(make_engine: Callable[..., Engine]) -> None:
    b = ScenarioBuilder.from_scenario(Scenario.load(bundled("single_intersection")))
    b.trip("amb1", 20.0, route=["W_in", "E_out"], vehicle_type="emergency")
    b.trip("amb2", 45.0, route=["N_in", "W_out"], vehicle_type="emergency")
    recorder = Recorder()
    e = make_engine(b.build(), controllers={0: recorder}, seed=4)
    prog = e.signals.programs[0]
    assert prog.intersection == 0
    last_seen: dict[int, float] = {}
    references: dict[int, dict[str, Any]] = {}
    for _ in range(150):
        references[e.step_count] = _reference(e, prog, last_seen)
        e.step()
    assert len(recorder.seen) > 100  # decide runs in every green step
    checked: Counter[str] = Counter()
    for step, seen in recorder.seen.items():
        ref = references[step]
        assert seen["rng"] is e.rng.stream("controller:J")
        for name in FIELDS:
            got = seen[name]
            if name == "emergency_approach":
                assert [(a, b, pytest.approx(c)) for a, b, c in got] == ref[name], step
            else:
                np.testing.assert_allclose(
                    np.asarray(got, dtype=float),
                    np.asarray(ref[name], dtype=float),
                    err_msg=f"{name} at step {step}",
                )
            if np.asarray(got, dtype=float).any():
                checked[name] += 1
    # the run exercised every field with non-trivial values
    assert set(checked) == set(FIELDS), set(FIELDS) - set(checked)
