"""The map editor's endpoints: operations on drafts, validation, generate, save (plan P)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from urbanflow.core.settings import load_settings
from urbanflow.server.app import create_app

API = "/api/v1"
BLANK: dict[str, Any] = {
    "format": "urbanflow.scenario",
    "version": "1.0",
    "meta": {"name": "drawn"},
    "network": {"intersections": [], "roads": []},
}


@pytest.fixture
def client(workspace: Path) -> Iterator[TestClient]:
    with TestClient(create_app(load_settings())) as c:
        yield c


def _ops(client: TestClient, scenario: dict[str, Any], *ops: dict[str, Any]) -> Any:
    return client.post(f"{API}/scenarios/ops", json={"scenario": scenario, "ops": list(ops)})


def test_draw_save_and_run_a_map(client: TestClient) -> None:
    blank = _ops(client, BLANK).json()
    assert blank["valid"] is False and "network.roads" in {i["path"] for i in blank["issues"]}
    r = _ops(
        client,
        BLANK,
        {"op": "add_intersection", "point": [-200, 0]},
        {"op": "add_intersection", "point": [0, 0], "kind": "signalized"},
        {"op": "add_intersection", "point": [200, 0]},
        {"op": "add_intersection", "point": [0, 200]},
        {"op": "add_road", "from": "J1", "to": "J2", "lanes": 2},
        {"op": "add_road", "from": "J2", "to": "J3", "lanes": 2},
        {"op": "add_road", "from": "J4", "to": "J2"},
        {"op": "add_flow", "origin": "J1_J2", "destination": "J2_J3", "rate": 400},
    )
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["valid"] is True and [c["id"] for c in doc["created"]][:2] == ["J1", "J2"]
    saved = client.post(
        f"{API}/scenarios/import",
        json={"name": "drawn", "format": "urbanflow", "scenario": doc["scenario"]},
    )
    assert saved.status_code == 201, saved.text
    assert {"name": "drawn", "source": "workspace"} in client.get(f"{API}/scenarios").json()
    loaded = client.get(f"{API}/scenarios/workspace/drawn").json()
    assert [ix["id"] for ix in loaded["network"]["intersections"]] == ["J1", "J2", "J3", "J4"]
    session = client.post(f"{API}/sessions", json={"scenario": "drawn", "source": "workspace"})
    assert session.status_code == 201, session.text


def test_errors_name_the_operation_or_the_field(client: TestClient) -> None:
    r = _ops(client, BLANK, {"op": "add_road", "from": "J1", "to": "J2"})
    assert r.status_code == 422
    assert r.json()["issues"][0]["path"] == "ops[0].from"
    bad_lane = {
        **BLANK,
        "network": {
            "intersections": [{"id": "A", "point": [0, 0]}, {"id": "B", "point": [100, 0]}],
            "roads": [{"id": "A_B", "from": "A", "to": "B", "lanes": [{"width": -1}]}],
        },
    }
    r = _ops(client, bad_lane)
    assert r.status_code == 422
    assert r.json()["issues"][0]["path"] == "network.roads[0].lanes[0].width"
    too_many = [{"op": "add_intersection", "point": [i, 0]} for i in range(1001)]
    assert _ops(client, BLANK, *too_many).status_code == 422


def test_generate_a_starting_point(client: TestClient) -> None:
    r = client.post(f"{API}/scenarios/generate", json={"generator": "grid", "params": {"rows": 2}})
    assert r.status_code == 200
    doc = r.json()["scenario"]
    assert doc["format"] == "urbanflow.scenario" and len(doc["network"]["intersections"]) > 4
    assert _ops(client, doc).json()["valid"] is True
    assert client.post(f"{API}/scenarios/generate", json={"generator": "nope"}).status_code == 404
