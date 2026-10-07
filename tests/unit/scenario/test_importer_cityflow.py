"""CityFlow import: a hand-written two-junction network in CityFlow's file format."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from urbanflow import Simulation
from urbanflow.core.errors import ScenarioValidationError
from urbanflow.scenario.importers.cityflow import CityFlowOptions, convert, load

NODES = {  # name -> (x, y, virtual)
    "J0": (0, 0, False),
    "J1": (300, 0, False),
    "W": (-300, 0, True),
    "E": (600, 0, True),
    "N0": (0, 300, True),
    "S0": (0, -300, True),
    "N1": (300, 300, True),
    "S1": (300, -300, True),
}
ARMS = {"J0": ["W", "J1", "N0", "S0"], "J1": ["J0", "E", "N1", "S1"]}
LANES, LANE_W, WIDTH = 3, 3.0, 20.0


def _turn(a: str, j: str, c: str) -> str:
    (ax, ay, _), (jx, jy, _), (cx, cy, _) = NODES[a], NODES[j], NODES[c]
    cross = (jx - ax) * (cy - jy) - (jy - ay) * (cx - jx)
    return "go_straight" if cross == 0 else ("turn_left" if cross > 0 else "turn_right")


def _lane_point(a: str, b: str, k: int, at_start: bool) -> dict[str, float]:
    (ax, ay, av), (bx, by, bv) = NODES[a], NODES[b]
    length = math.hypot(bx - ax, by - ay)
    dx, dy = (bx - ax) / length, (by - ay) / length
    off = (k + 0.5) * LANE_W
    trim = (0.0 if av else WIDTH) if at_start else (length - (0.0 if bv else WIDTH))
    return {"x": ax + dx * trim + dy * off, "y": ay + dy * trim - dx * off}


def cityflow_files() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    roads, seen = [], set()
    for j, arms in ARMS.items():
        for b in arms:
            for a, z in ((b, j), (j, b)):
                if (a, z) not in seen:
                    seen.add((a, z))
                    roads.append(
                        {
                            "id": f"road_{a}_{z}",
                            "points": [
                                {"x": NODES[a][0], "y": NODES[a][1]},
                                {"x": NODES[z][0], "y": NODES[z][1]},
                            ],
                            "lanes": [{"width": LANE_W, "maxSpeed": 11.111}] * LANES,
                            "startIntersection": a,
                            "endIntersection": z,
                        }
                    )
    lane_of = {"turn_left": 0, "go_straight": 1, "turn_right": 2}
    intersections = []
    for name, (x, y, virtual) in NODES.items():
        ix: dict[str, Any] = {
            "id": name,
            "point": {"x": x, "y": y},
            "virtual": virtual,
            "width": 0 if virtual else WIDTH,
            "roads": [],
            "roadLinks": [],
        }
        if not virtual:
            links = []
            for a in ARMS[name]:
                for c in ARMS[name]:
                    if c == a:
                        continue
                    kind = _turn(a, name, c)
                    k = lane_of[kind]
                    links.append(
                        {
                            "type": kind,
                            "startRoad": f"road_{a}_{name}",
                            "endRoad": f"road_{name}_{c}",
                            "direction": 0,  # ignored, as in CityFlow
                            "laneLinks": [
                                {
                                    "startLaneIndex": k,
                                    "endLaneIndex": e,
                                    "points": [
                                        _lane_point(a, name, k, False),
                                        _lane_point(name, c, e, True),
                                    ],
                                }
                                for e in range(LANES)  # CityFlow's full fan-out
                            ],
                        }
                    )
            ix["roadLinks"] = links

            def ids(
                kind: str, vertical: bool, _links: list[Any] = links, _j: str = name
            ) -> list[int]:
                return [
                    i
                    for i, link in enumerate(_links)
                    if link["type"] == kind
                    and (NODES[link["startRoad"].split("_")[1]][0] == NODES[_j][0]) == vertical
                ]

            rights = [i for i, link in enumerate(links) if link["type"] == "turn_right"]
            ix["trafficLight"] = {
                "roadLinkIndices": list(range(len(links))),
                "lightphases": [
                    {"time": 5, "availableRoadLinks": rights},  # CityFlow-style intergreen
                    {"time": 30, "availableRoadLinks": ids("go_straight", True) + rights},
                    {"time": 15, "availableRoadLinks": ids("turn_left", True) + rights},
                    {"time": 30, "availableRoadLinks": ids("go_straight", False) + rights},
                    {"time": 15, "availableRoadLinks": ids("turn_left", False) + rights},
                ],
            }
        intersections.append(ix)
    vehicle = {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": 11.111,
        "headwayTime": 2,
    }
    flows = [
        {
            "vehicle": vehicle,
            "route": ["road_W_J0", "road_J0_J1", "road_J1_E"],
            "interval": 15,
            "startTime": 0,
            "endTime": -1,
        },
        {
            "vehicle": vehicle,
            "route": ["road_E_J1", "road_J0_W"],
            "interval": 10,  # anchors
            "startTime": 0,
            "endTime": 599,
        },
        {
            "vehicle": {**vehicle, "maxSpeed": 8.0},
            "route": ["road_N0_J0", "road_J0_S0"],
            "interval": 20,
            "startTime": 10,
            "endTime": -1,
        },
    ]
    config = {
        "interval": 1.0,
        "seed": 3,
        "dir": "",
        "roadnetFile": "roadnet.json",
        "flowFile": "flow.json",
        "rlTrafficLight": False,
        "laneChange": False,
        "saveReplay": True,
        "roadnetLogFile": "frontend/roadnet.json",
    }
    return {"intersections": intersections, "roads": roads}, flows, config


def test_mapping() -> None:
    roadnet, flows, config = cityflow_files()
    result = convert(roadnet, flows, config, CityFlowOptions(name="two_junctions"))
    spec = result.scenario.resolved
    ixs = {ix.id: ix for ix in spec.network.intersections}
    assert ixs["W"].kind == "boundary" and ixs["J0"].kind == "signalized"
    assert ixs["J0"].radius == WIDTH
    sig = ixs["J0"].signal
    assert sig is not None
    assert [p.duration for p in sig.phases] == [30, 15, 30, 15]  # intergreen dropped
    assert (sig.yellow, sig.all_red) == (5.0, 0.0)
    ns_left = dict(sig.phases[1].green)
    assert ns_left["road_N0_J0->road_J0_J1"] == "G"  # protected: no opposing straight
    ns_straight = dict(sig.phases[0].green)
    assert ns_straight["road_N0_J0->road_J0_S0"] == "G"
    assert ns_straight["road_S0_J0->road_J0_J1"] == "g"  # right turn yields to the straight
    movs = {m.id: m for m in ixs["J0"].movements or ()}
    assert len(movs["road_W_J0->road_J0_J1"].connections or ()) == 3  # fan-out kept
    flow = {f.id: f for f in spec.demand.flows}
    assert flow["flow_0"].route == ("road_W_J0", "road_J0_J1", "road_J1_E")
    assert (flow["flow_1"].origin, flow["flow_1"].destination) == ("road_E_J1", "road_J0_W")
    assert flow["flow_1"].end == 599.5 and flow["flow_0"].end is None
    assert flow["flow_0"].period == 15 and flow["flow_2"].vehicle_type == "cf_type_1"
    types = {t.id: t for t in spec.vehicle_types}
    assert (types["cf_type_0"].headway, types["cf_type_0"].accel) == (2.0, 2.0)
    assert (spec.simulation.seed, spec.simulation.lane_changing) == (3, False)
    messages = " | ".join(w.message for w in result.warnings)
    assert "fan-out" in messages and "replays" in messages


def test_options() -> None:
    roadnet, flows, config = cityflow_files()
    derived = convert(roadnet, flows, config, CityFlowOptions(derive_connections=True))
    j0 = next(ix for ix in derived.scenario.spec.network.intersections if ix.id == "J0")
    assert all(m.connections is None for m in j0.movements or ())
    kept = convert(
        roadnet, flows, {**config, "rlTrafficLight": True}, CityFlowOptions(keep_all_phases=True)
    )
    sig = next(ix.signal for ix in kept.scenario.resolved.network.intersections if ix.id == "J1")
    assert sig is not None and len(sig.phases) == 5 and sig.controller.type == "external"


def test_imported_network_runs() -> None:
    roadnet, flows, config = cityflow_files()
    scenario = convert(roadnet, flows, config, CityFlowOptions(duration=600)).scenario
    result = Simulation(scenario, debug_checks=True).run()
    assert result.summary["vehicles.arrived"] > 50
    assert result.summary["vehicles.teleported"] == 0


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (
            lambda n, _f: n["intersections"][0]["roadLinks"][0].update(type="turn_u"),
            "roadnet.intersections[0].roadLinks[0].type",
        ),
        (
            lambda n, _f: n["intersections"][0]["trafficLight"]["lightphases"][1][
                "availableRoadLinks"
            ].append(99),
            "roadnet.intersections[0].trafficLight.lightphases[1].availableRoadLinks",
        ),
        (lambda _n, f: f[0]["route"].append("road_nowhere"), "flow[0].route[3]"),
        (
            lambda n, _f: n["roads"][0]["lanes"][0].update(width=-1),
            "roadnet.roads[0].lanes[0].width",
        ),
    ],
)
def test_errors_point_into_the_input(mutate: Any, path: str) -> None:
    roadnet, flows, config = cityflow_files()
    mutate(roadnet, flows)
    with pytest.raises(ScenarioValidationError) as info:
        convert(roadnet, flows, config)
    assert path in str(info.value)


def test_load_files_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from urbanflow.cli import main as cli

    def run(argv: list[str]) -> int:
        with pytest.raises(SystemExit) as info:
            cli.main(argv)
        return int(info.value.code or 0)

    roadnet, flows, config = cityflow_files()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "roadnet.json").write_text(json.dumps(roadnet), encoding="utf-8")
    (tmp_path / "data" / "flow.json").write_text(json.dumps(flows), encoding="utf-8")
    cfg = tmp_path / "data" / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    assert load(config=cfg).scenario.summary()["flows"] == 3  # files next to the config
    out = tmp_path / "imported.json"
    assert run(["import", "cityflow", "--config", str(cfg), "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert "Wrote" in captured.out and "warning" in captured.err and out.is_file()
    assert run(["import", "cityflow", "--config", str(cfg), "-o", str(out)]) == 4  # exists
    assert run(["import", "sumo", "--config", str(cfg)]) == 2  # unknown format
