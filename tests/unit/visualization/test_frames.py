"""Frames, lane values, the registry tracker and the UFB1 encoding (plan K.10)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from urbanflow import Simulation, bundled
from urbanflow.core.errors import ReplayFormatError
from urbanflow.visualization import (
    LaneMetric,
    RegistryTracker,
    build_frame,
    decode_frame,
    encode_frame,
)
from urbanflow.visualization.encoding import HEADER


@pytest.fixture(scope="module")
def sim() -> Simulation:
    s = Simulation.from_scenario(bundled("single_intersection"), seed=4, duration=600)
    s.step(150)
    return s


def test_frame_matches_state_arrays(sim: Simulation) -> None:
    frame = build_frame(sim._engine)
    state = sim.state
    order = np.argsort(state.uids)
    assert frame.n == state.uids.size > 0
    assert np.array_equal(frame.uid, state.uids[order])
    local = state.xy[order] - sim.network.compiled.origin
    assert np.allclose(frame.xy, local, atol=1e-3)
    assert np.allclose(frame.speed, state.speeds[order], atol=1e-5)
    assert frame.signals is not None and frame.signals.size == len(sim.network.movement_ids)
    assert np.all(np.diff(frame.uid.astype(np.int64)) > 0)


@pytest.mark.parametrize("metric", list(LaneMetric))
def test_round_trip(sim: Simulation, metric: LaneMetric) -> None:
    frame = build_frame(sim._engine, lane_metric=metric)
    buf = encode_frame(frame, seq=7, geometry_crc=0xDEADBEEF)
    assert struct.unpack_from("<I", buf, 48)[0] == len(buf)
    back, seq, crc = decode_frame(buf)
    assert (seq, crc, back.step, back.lane_metric) == (7, 0xDEADBEEF, frame.step, metric)
    for name in ("uid", "xy", "heading", "speed", "link", "flags"):
        assert np.array_equal(getattr(back, name), getattr(frame, name))
    assert np.array_equal(back.signals, frame.signals)  # type: ignore[arg-type]
    if metric is LaneMetric.none:
        assert back.lane_values is None
    else:
        assert back.lane_values is not None
        assert np.array_equal(back.lane_values, frame.lane_values, equal_nan=True)  # type: ignore[arg-type]


def test_layout_offsets_are_aligned(sim: Simulation) -> None:
    frame = build_frame(sim._engine, lane_metric=LaneMetric.density)
    buf = encode_frame(frame)
    n, m = frame.n, frame.signals.size  # type: ignore[union-attr]
    signals_at = (HEADER.size + 25 * n + 3) & ~3
    lanes_at = (signals_at + m + 3) & ~3
    assert np.array_equal(np.frombuffer(buf, "u1", m, signals_at), frame.signals)  # type: ignore[arg-type]
    lanes = np.frombuffer(buf, "<f4", sim.network.compiled.n_links, lanes_at)
    assert lanes_at % 4 == 0 and np.array_equal(lanes, frame.lane_values, equal_nan=True)  # type: ignore[arg-type]


def test_lane_values_density_and_queue(sim: Simulation) -> None:
    eng = sim._engine
    net = eng.network
    dens = build_frame(eng, lane_metric=LaneMetric.density).lane_values
    counts = np.bincount(eng.vehicles.link[eng.vehicles.active], minlength=net.n_links)
    assert dens is not None
    assert np.allclose(dens, counts / (net.link_length / 1000.0), rtol=1e-5)
    queue = build_frame(eng, lane_metric=LaneMetric.queue).lane_values
    assert queue is not None and np.all(np.isnan(queue[net.n_lanes :]))


def test_bad_frames_are_rejected(sim: Simulation) -> None:
    buf = encode_frame(build_frame(sim._engine))
    with pytest.raises(ReplayFormatError):
        decode_frame(b"XXXX" + buf[4:])
    with pytest.raises(ReplayFormatError):
        decode_frame(buf[:-1])


def test_registry_tracker() -> None:
    t = RegistryTracker()
    added, removed = t.delta(np.array([1, 2, 3]))
    assert added.tolist() == [1, 2, 3] and removed.tolist() == []
    added, removed = t.delta(np.array([2, 3, 5]))
    assert added.tolist() == [5] and removed.tolist() == [1]
    added, removed = t.delta(np.array([5]), reset=True)
    assert added.tolist() == [5] and removed.tolist() == []
