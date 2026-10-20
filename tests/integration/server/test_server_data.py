"""Workbench server: importing scenarios, traffic scaling, recordings and result downloads."""

from __future__ import annotations

import csv
import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urbanflow import generate
from urbanflow.core.settings import load_settings
from urbanflow.server.app import create_app

API = "/api/v1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "unit" / "scenario"))
from test_importer_cityflow import cityflow_files  # noqa: E402


@pytest.fixture
def client(workspace: Path) -> Iterator[TestClient]:
    with TestClient(create_app(load_settings(max_workers=1))) as c:
        yield c


def _doc() -> dict[str, Any]:
    return generate("single_intersection", duration=600).to_dict()


def test_import_download_delete(client: TestClient) -> None:
    r = client.post(f"{API}/scenarios/import", json={"name": "my_junction", "scenario": _doc()})
    assert r.status_code == 201, r.text
    assert r.json()["summary"]["intersections"] == 5
    listed = {(s["source"], s["name"]) for s in client.get(f"{API}/scenarios").json()}
    assert ("workspace", "my_junction") in listed
    again = client.post(f"{API}/scenarios/import", json={"name": "my_junction", "scenario": _doc()})
    assert again.status_code == 409
    over = {"name": "my_junction", "scenario": _doc(), "overwrite": True}
    assert client.post(f"{API}/scenarios/import", json=over).status_code == 201
    down = client.get(f"{API}/scenarios/workspace/my_junction")
    assert down.status_code == 200 and json.loads(down.content)["format"] == "urbanflow.scenario"
    assert client.get(f"{API}/scenarios/bundled/grid_3x3").status_code == 200
    s = client.post(f"{API}/sessions", json={"scenario": "my_junction", "source": "workspace"})
    assert s.status_code == 201
    assert client.delete(f"{API}/scenarios/my_junction").status_code == 204
    assert client.get(f"{API}/scenarios/workspace/my_junction").status_code == 404


def test_import_errors(client: TestClient) -> None:
    bad = _doc()
    bad["network"]["roads"][0]["lanes"] = []
    r = client.post(f"{API}/scenarios/import", json={"name": "broken", "scenario": bad})
    assert r.status_code == 422 and r.json()["issues"][0]["path"].startswith("network.roads[0]")
    assert (
        client.post(
            f"{API}/scenarios/import", json={"name": "../x", "scenario": _doc()}
        ).status_code
        == 422
    )
    assert (
        client.post(f"{API}/scenarios/import", json={"name": "x", "format": "cityflow"}).status_code
        == 422
    )
    assert client.delete(f"{API}/scenarios/single_intersection").status_code == 404  # bundled


def test_import_cityflow(client: TestClient) -> None:
    roadnet, flows, config = cityflow_files()
    body = {"name": "cf", "format": "cityflow", "roadnet": roadnet, "flow": flows, "config": config}
    r = client.post(f"{API}/scenarios/import", json=body)
    assert r.status_code == 201, r.text
    assert any("fan-out" in w["message"] for w in r.json()["warnings"])


def test_body_limit(workspace: Path) -> None:
    with TestClient(create_app(load_settings(max_workers=1, max_body_mb=0.001))) as c:
        r = c.post(f"{API}/scenarios/import", json={"name": "big", "scenario": _doc()})
        assert r.status_code == 413


def test_demand_scale_recording_and_exports(client: TestClient, workspace: Path) -> None:
    base = client.post(f"{API}/sessions", json={"scenario": "single_intersection"}).json()
    doubled = client.post(
        f"{API}/sessions",
        json={"scenario": "single_intersection", "demand_scale": 2, "record": True},
    ).json()

    def first_flow(sid: str) -> dict[str, Any]:
        doc = json.loads(client.get(f"{API}/sessions/{sid}/scenario").content)
        return dict(doc["demand"]["flows"][0])

    first, second = first_flow(base["id"]), first_flow(doubled["id"])
    assert second.get("rate", 0) == pytest.approx(2 * first.get("rate", 0)) or second.get(
        "period", 0
    ) == pytest.approx(first.get("period", 0) / 2)
    sid = doubled["id"]
    with client.websocket_connect(f"{API}/ws/sessions/{sid}") as ws:
        ws.send_json({"type": "step", "n": 300})
        for _ in range(200):
            msg = ws.receive()
            if msg.get("text") and json.loads(msg["text"]).get("step", 0) >= 300:
                break
    trips = client.get(f"{API}/sessions/{sid}/export/trips.csv")
    assert trips.status_code == 200 and "attachment" in trips.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(trips.text)))
    assert rows and {"vehicle_id", "travel_time"} <= set(rows[0])
    series = client.get(f"{API}/sessions/{sid}/export/timeseries.csv")
    assert series.status_code == 200 and series.text.startswith("time,")
    assert client.get(f"{API}/sessions/{sid}/export/secrets.csv").status_code == 422
    assert client.delete(f"{API}/sessions/{sid}").status_code == 204  # finishes the recording
    replays = client.get(f"{API}/replays").json()
    assert replays, "the recording is listed"
    file = client.get(f"{API}/replays/{replays[0]['name']}/file")
    assert file.status_code == 200 and file.content[:2] == b"PK"


def test_record_command(client: TestClient) -> None:
    sid = client.post(f"{API}/sessions", json={"scenario": "single_intersection"}).json()["id"]
    with client.websocket_connect(f"{API}/ws/sessions/{sid}") as ws:
        ws.send_json({"type": "record", "on": True})
        ws.send_json({"type": "step", "n": 30})
        ws.send_json({"type": "record", "on": False})
        saved = None
        for _ in range(300):
            msg = ws.receive()
            if msg.get("text"):
                m = json.loads(msg["text"])
                if m.get("type") == "status" and m.get("last_recording"):
                    saved = m["last_recording"]
                    break
    assert saved and saved in {r["name"] for r in client.get(f"{API}/replays").json()}


def test_import_osm(client: TestClient) -> None:
    from osm_fixture import town

    r = client.post(
        f"{API}/scenarios/import", json={"name": "pune", "format": "osm", "osm": town()}
    )
    assert r.status_code == 201, r.text
    assert r.json()["summary"]["signals"] == 1
    bad = {"name": "x", "format": "osm", "osm": "<!DOCTYPE x><osm/>"}
    assert client.post(f"{API}/scenarios/import", json=bad).status_code == 422


def test_session_router(client: TestClient) -> None:
    r = client.post(f"{API}/sessions", json={"scenario": "grid_3x3", "router": "dynamic"})
    assert r.status_code == 201
    bad = client.post(f"{API}/sessions", json={"scenario": "grid_3x3", "router": "teleport"})
    assert bad.status_code == 422
