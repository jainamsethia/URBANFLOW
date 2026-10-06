"""Replay sessions over the WebSocket protocol, and the replay CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urbanflow import Simulation, bundled
from urbanflow.cli import main as cli
from urbanflow.core.settings import load_settings
from urbanflow.server.app import create_app
from urbanflow.visualization import decode_frame

API = "/api/v1"


@pytest.fixture
def recorded(workspace: Path) -> Path:
    sim = Simulation.from_scenario(bundled("single_intersection"), seed=2, duration=200)
    sim.start_recording(workspace / "replays" / "demo.ufr")
    sim.run()
    return sim.stop_recording()


@pytest.fixture
def client(recorded: Path) -> Iterator[TestClient]:
    with TestClient(create_app(load_settings())) as c:
        yield c


def _next(ws: Any, pred: Any, limit: int = 300) -> Any:
    for _ in range(limit):
        msg = ws.receive()
        item = decode_frame(msg["bytes"])[0] if msg.get("bytes") else json.loads(msg["text"])
        if pred(item):
            return item
    raise AssertionError("condition not reached")


def test_replay_session_seek_and_step_back(client: TestClient, recorded: Path) -> None:
    assert [r["name"] for r in client.get(f"{API}/replays").json()] == ["demo"]
    r = client.post(f"{API}/sessions", json={"replay": "demo"})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    geo = client.get(f"{API}/sessions/{sid}/geometry").json()
    assert geo["geometry_crc"] > 0
    with client.websocket_connect(f"{API}/ws/sessions/{sid}") as ws:
        assert ws.receive_json()["type"] == "hello"
        status = _next(ws, lambda m: isinstance(m, dict) and m.get("type") == "status")
        assert status["kind"] == "replay" and status["last_step"] == 200
        ws.send_json({"type": "seek", "step": 150})
        f = _next(ws, lambda x: not isinstance(x, dict) and x.step == 150)
        assert f.n > 0
        ws.send_json({"type": "step", "n": -10})
        _next(ws, lambda x: not isinstance(x, dict) and x.step == 140)
        ws.send_json({"type": "hold_phase", "intersection": "J", "phase": 0})
        err = _next(ws, lambda m: isinstance(m, dict) and m.get("type") == "error")
        assert "replay" in err["message"]
        ws.send_json({"type": "select", "kind": "intersection", "id": "J"})
        sel = _next(ws, lambda m: isinstance(m, dict) and m.get("type") == "selection")
        assert sel["available"] and sel["detail"]["movement_states"]
    missing = client.post(f"{API}/sessions", json={"replay": "nope"})
    assert missing.status_code == 404


def test_live_session_rejects_replay_commands(client: TestClient) -> None:
    sid = client.post(f"{API}/sessions", json={"scenario": "single_intersection"}).json()["id"]
    with client.websocket_connect(f"{API}/ws/sessions/{sid}") as ws:
        ws.receive_json()
        ws.send_json({"type": "seek", "step": 5})
        err = _next(ws, lambda m: isinstance(m, dict) and m.get("type") == "error")
        assert "replay session" in err["message"]


def test_replay_cli_info(recorded: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["replay", str(recorded), "--json"])
    assert info.value.code in (0, None)
    doc = json.loads(capsys.readouterr().out)
    assert doc["complete"] is True and doc["n_frames"] == 201
