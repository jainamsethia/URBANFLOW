"""Workbench server: REST, the session WebSocket protocol and controller comparison."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from urbanflow.core.settings import load_settings
from urbanflow.server.app import create_app
from urbanflow.visualization import decode_frame

API = "/api/v1"


@pytest.fixture
def client(workspace: Path) -> Iterator[TestClient]:
    app = create_app(load_settings(max_workers=1))
    with TestClient(app) as c:
        yield c


def _session(client: TestClient, **body: Any) -> str:
    r = client.post(f"{API}/sessions", json={"scenario": "single_intersection", **body})
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


def _until(ws: Any, pred: Any, limit: int = 400) -> Any:
    """Receive messages (text as dicts, binary as decoded frames) until ``pred`` holds."""
    for _ in range(limit):
        msg = ws.receive()
        if msg.get("bytes") is not None:
            item: Any = decode_frame(msg["bytes"])[0]
        else:
            import json

            item = json.loads(msg["text"])
        if pred(item):
            return item
    raise AssertionError("condition not reached")


def _is_frame(x: Any) -> bool:
    return not isinstance(x, dict)


def test_meta_and_scenarios(client: TestClient) -> None:
    assert client.get(f"{API}/health").json()["status"] == "ok"
    meta = client.get(f"{API}/meta").json()
    assert {"fixed_time", "max_pressure", "actuated"} <= set(meta["controllers"])
    assert {g["name"] for g in meta["generators"]} >= {"grid", "single_intersection"}
    assert any(m["name"] == "density" for m in meta["lane_metrics"])
    names = {s["name"] for s in client.get(f"{API}/scenarios").json()}
    assert {"single_intersection", "grid_3x3"} <= names


def test_session_lifecycle_and_errors(client: TestClient) -> None:
    sid = _session(client, controller="max_pressure", seed=3)
    info = client.get(f"{API}/sessions/{sid}").json()
    assert info["state"] == "paused" and info["controllers"] == {"J": "max_pressure"}
    geo = client.get(f"{API}/sessions/{sid}/geometry").json()
    assert len(geo["links"]["id"]) == 32 and geo["vehicle_types"]
    bad = client.post(f"{API}/sessions", json={"scenario": "../etc"})
    assert bad.status_code == 422
    missing = client.post(f"{API}/sessions", json={"scenario": "nope"})
    assert missing.status_code == 404
    both = client.post(f"{API}/sessions", json={"scenario": "x", "generator": "grid"})
    assert both.status_code == 422
    gen = client.post(f"{API}/sessions", json={"generator": "grid", "params": {"rows": 2}})
    assert gen.status_code == 201
    assert client.delete(f"{API}/sessions/{sid}").status_code == 204
    assert client.get(f"{API}/sessions/{sid}").status_code == 404


def test_websocket_protocol(client: TestClient) -> None:
    sid = _session(client, seed=1)
    with client.websocket_connect(f"{API}/ws/sessions/{sid}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["session"]["id"] == sid
        reg = _until(ws, lambda m: isinstance(m, dict) and m["type"] == "vehicles_added")
        assert reg["reset"] is True
        first = _until(ws, _is_frame)
        assert first.step == 0

        ws.send_json({"type": "step", "n": 30})
        frame = _until(ws, lambda f: _is_frame(f) and f.step == 30)
        assert frame.n > 0 and frame.signals is not None
        known: set[int] = set()
        names: dict[int, str] = {}
        # a vehicles_added for every uid in a frame always arrives before that frame
        ws.send_json({"type": "set_speed", "steps_per_second": 200})
        ws.send_json({"type": "play"})
        for _ in range(200):
            msg = ws.receive()
            if msg.get("text"):
                import json

                m = json.loads(msg["text"])
                if m["type"] == "vehicles_added":
                    known |= set(m["uids"])
                    names.update(zip(m["uids"], m["ids"], strict=True))
                elif m["type"] == "vehicles_removed":
                    known -= set(m["uids"])
            elif msg.get("bytes"):
                f = decode_frame(msg["bytes"])[0]
                if f.step > 30:
                    assert set(f.uid.tolist()) <= known | set(frame.uid.tolist())
                    if f.step > 120:
                        break
        ws.send_json({"type": "pause"})
        _until(ws, lambda m: isinstance(m, dict) and m.get("state") == "paused")
        vid = next(names[u] for u in f.uid.tolist() if u in names)
        ws.send_json({"type": "select", "kind": "vehicle", "id": vid})
        sel = _until(ws, lambda m: isinstance(m, dict) and m["type"] == "selection")
        d = sel["detail"]  # the workbench highlights route[route_index:]
        assert 0 <= d["route_index"] < len(d["route"]) and d["route"][-1] == d["destination"]

        ws.send_json({"type": "set_controller", "controller": "actuated"})
        status = _until(
            ws,
            lambda m: (
                isinstance(m, dict)
                and m["type"] == "status"
                and m["controllers"] == {"J": "actuated"}
            ),
        )
        assert status["state"] == "paused"
        ws.send_json({"type": "select", "kind": "intersection", "id": "J"})
        sel = _until(ws, lambda m: isinstance(m, dict) and m["type"] == "selection")
        assert sel["detail"]["controller"] == "actuated" and sel["detail"]["phases"]
        ws.send_json({"type": "subscribe", "lane_metric": 1})
        ws.send_json({"type": "step", "n": 1})
        dens = _until(ws, lambda f: _is_frame(f) and f.lane_values is not None)
        assert dens.lane_metric == 1
        ws.send_json({"type": "warp", "n": 1})
        err = _until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert err["code"] == "invalid_message"
        ws.send_json({"type": "hold_phase", "intersection": "J", "phase": 9})
        err = _until(ws, lambda m: isinstance(m, dict) and m["type"] == "error")
        assert err["code"] == "command_failed"
        ws.send_json({"type": "reset", "seed": 1})
        again = _until(ws, lambda m: isinstance(m, dict) and m["type"] == "vehicles_added")
        assert again["reset"] is True


def test_cross_origin_websocket_is_refused(client: TestClient) -> None:
    sid = _session(client)
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(
            f"{API}/ws/sessions/{sid}", headers={"origin": "http://evil.example"}
        ) as ws,
    ):
        ws.receive_json()


def test_unknown_session_websocket_closes(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"{API}/ws/sessions/doesnotexist") as ws,
    ):
        ws.receive_json()


def test_compare_endpoint(client: TestClient) -> None:
    body = {
        "scenario": "single_intersection",
        "controllers": ["fixed_time", "max_pressure"],
        "seeds": 2,
        "duration": 300,
    }
    r = client.post(f"{API}/compare", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["baseline"] == "fixed_time" and len(out["runs"]) == 4
    travel = next(m for m in out["metrics"] if m["metric"] == "travel_time.mean")
    assert set(travel["by_controller"]) == {"fixed_time", "max_pressure"}
    assert travel["vs_baseline"]["max_pressure"]["n_pairs"] == 2
    arrived = next(m for m in out["metrics"] if m["metric"] == "vehicles.arrived")
    assert all(v["mean"] > 0 for v in arrived["by_controller"].values())


def test_token_required(workspace: Path) -> None:
    app = create_app(load_settings(api_token="s3cret"))
    with TestClient(app) as c:
        assert c.get(f"{API}/health").status_code == 200
        assert c.get(f"{API}/meta").status_code == 401
        ok = c.get(f"{API}/meta", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
