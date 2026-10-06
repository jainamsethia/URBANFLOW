"""Replay recording and safe reading (plan K, AT-24/AT-25 core)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from urbanflow import Simulation, bundled
from urbanflow.core.errors import ReplayFormatError, SimulationError
from urbanflow.replay import ReplayReader
from urbanflow.visualization import encode_frame


def test_recorded_frames_equal_the_live_frames(tmp_path: Path) -> None:
    sim = Simulation.from_scenario(bundled("single_intersection"), seed=5, duration=300)
    sim.start_recording(tmp_path / "a.ufr")
    live = {0: sim.frame()}
    for _ in range(300):
        sim.step()
        if sim.step_count % 37 == 0:
            live[sim.step_count] = sim.frame()
    path = sim.stop_recording()
    r = ReplayReader(path)
    assert r.manifest["complete"] and r.n_frames == 301 and r.steps[-1] == 300
    for step, frame in live.items():
        got = r.frame(step)
        assert got.step == step
        for name in ("uid", "xy", "heading", "speed", "link", "flags", "signals"):
            assert np.array_equal(getattr(got, name), getattr(frame, name)), name
    meta = r.vehicles()
    assert set(live[max(live)].uid.tolist()) <= set(meta)
    assert r.summary() is not None and json.loads(r.scenario_json())["meta"]["name"]
    assert json.loads(r.geometry_json())["geometry_crc"] == r.manifest["geometry_crc"]


def test_record_every_and_seek(tmp_path: Path) -> None:
    sim = Simulation.from_scenario(
        bundled("single_intersection"), duration=120, record=tmp_path / "b.ufr", **{}
    )
    sim.run()
    r = ReplayReader(sim.stop_recording())
    assert r.index_at(-5) == 0 and r.index_at(10_000) == r.n_frames - 1
    assert r.frame(57).step == 57


def test_unpacked_directory_is_readable(tmp_path: Path) -> None:
    sim = Simulation.from_scenario(bundled("single_intersection"), duration=900)
    sim.start_recording(tmp_path / "c.ufr")
    sim.step(650)  # two full chunks flushed, the rest still buffered
    r = ReplayReader(tmp_path / "c.ufr.d")
    assert r.manifest["complete"] is False and r.n_frames == 600
    assert r.frame(599).step == 599


def test_reset_restarts_and_double_start_fails(tmp_path: Path) -> None:
    sim = Simulation.from_scenario(bundled("single_intersection"), duration=100)
    sim.start_recording(tmp_path / "d.ufr")
    with pytest.raises(SimulationError):
        sim.start_recording(tmp_path / "e.ufr")
    sim.step(10)
    sim.reset()
    sim.step(5)
    r = ReplayReader(sim.stop_recording())
    assert r.steps == [0, 1, 2, 3, 4, 5]


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_malicious_archives_are_rejected(tmp_path: Path) -> None:
    good = {"format": "urbanflow.replay", "version": "1.0", "chunks": []}
    with pytest.raises(ReplayFormatError, match="unsafe"):
        ReplayReader(_zip(tmp_path / "1.ufr", {"../evil.txt": b"x", "manifest.json": b"{}"}))
    with pytest.raises(ReplayFormatError, match="not an UrbanFlow replay"):
        ReplayReader(_zip(tmp_path / "2.ufr", {"manifest.json": b'{"format": "x"}'}))
    with pytest.raises(ReplayFormatError, match="not supported"):
        ReplayReader(
            _zip(
                tmp_path / "3.ufr",
                {"manifest.json": json.dumps({**good, "version": "2.0"}).encode()},
            )
        )
    bad_chunks = {**good, "chunks": [{"name": "chunks/0.bin", "steps": [0], "offsets": [0, 99]}]}
    with pytest.raises(ReplayFormatError, match="offsets"):
        ReplayReader(
            _zip(
                tmp_path / "4.ufr",
                {"manifest.json": json.dumps(bad_chunks).encode(), "chunks/0.bin": b"x"},
            )
        )
    (tmp_path / "5.ufr").write_bytes(b"not a zip")
    with pytest.raises(ReplayFormatError):
        ReplayReader(tmp_path / "5.ufr")
    with pytest.raises(ReplayFormatError, match="JSON"):
        ReplayReader(_zip(tmp_path / "6.ufr", {"manifest.json": b"{nope"}))


def test_frame_bytes_are_ufb1(tmp_path: Path) -> None:
    sim = Simulation.from_scenario(bundled("single_intersection"), duration=30)
    sim.start_recording(tmp_path / "f.ufr")
    sim.run()
    frame = sim.frame()
    r = ReplayReader(sim.stop_recording())
    raw = r.frame_bytes(r.n_frames - 1)
    assert raw[:4] == b"UFB1" and raw[9] & 4  # flags: source_replay
    expect = encode_frame(frame, 0, 0, replay=True)
    assert raw[64:] == expect[64:]
