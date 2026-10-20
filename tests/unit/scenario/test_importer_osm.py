"""OpenStreetMap import on a hand-written extract (see ``osm_fixture``)."""

from __future__ import annotations

import math

import pytest
from osm_fixture import town

from urbanflow import Simulation
from urbanflow.core.errors import ScenarioValidationError
from urbanflow.scenario.importers.osm import OsmOptions, convert


@pytest.fixture(scope="module")
def imported():  # type: ignore[no-untyped-def]
    return convert(town(), OsmOptions(duration=900))


def test_network_structure(imported) -> None:  # type: ignore[no-untyped-def]
    spec = imported.scenario.resolved
    ixs = spec.network.intersections
    kinds = [ix.kind for ix in ixs]
    assert kinds.count("signalized") == 1  # only the crossing with the signal node
    assert kinds.count("boundary") == 10 and kinds.count("priority") == 5
    # the two nodes 4 m apart at x = -200 / -196 are one junction; no two junctions closer
    pts = [ix.point for ix in ixs]
    assert min(math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :]) > 25
    roads = {r.id: r for r in spec.network.roads}
    ways = {rid.split("_")[0] for rid in roads}
    assert {"w508", "w509", "w510"}.isdisjoint(ways)  # private, footway, service
    oneway = [r for rid, r in roads.items() if rid.startswith("w507_")]
    assert len(oneway) == 4 and all(r.points[0][0] < r.points[-1][0] for r in oneway)  # eastbound
    main = [r for rid, r in roads.items() if rid.startswith(("w500_", "w501_"))]
    assert len(main) == 8 and all(len(r.lanes) == 2 for r in main)  # lanes=4: 2 each way
    # the tag split at x = 300 is not a junction: one road from x = 200 to the east edge
    assert any(rid.startswith("w501_") and len(r.points) >= 3 for rid, r in roads.items())
    assert all(
        r.speed_limit == pytest.approx(30 * 1.609344 / 3.6, abs=0.01)
        for rid, r in roads.items()
        if rid.startswith("w506_")
    )
    assert all(r.speed_limit == pytest.approx(50 / 3.6, abs=0.01) for r in main)


def test_traffic_flows_and_runs(imported) -> None:  # type: ignore[no-untyped-def]
    flows = imported.scenario.resolved.demand.flows
    assert flows and all(f.origin and f.destination and f.arrival == "poisson" for f in flows)
    result = Simulation(imported.scenario, seed=0, debug_checks=True).run().summary
    assert result["vehicles.arrived"] > 200
    assert result["vehicles.teleported"] == 0 and result["vehicles.backlog"] == 0


def test_options() -> None:
    base = convert(town(), OsmOptions(duration=900))
    half = convert(town(), OsmOptions(duration=900, demand=0.5))
    total = lambda r: sum(f.rate or 0 for f in r.scenario.resolved.demand.flows)  # noqa: E731
    assert total(half) == pytest.approx(total(base) / 2, rel=0.02)
    service = convert(town(), OsmOptions(include_service=True))
    assert any(r.id.startswith("w510_") for r in service.scenario.resolved.network.roads)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('<?xml version="1.0"?><!DOCTYPE osm [<!ENTITY x "y">]><osm/>', "DTD"),
        ("<osm><node id='1' lat='0' lon='0'/></osm>", "no drivable roads"),
        ("<osm><way", "not valid OSM XML"),
    ],
)
def test_rejects(text: str, message: str) -> None:
    with pytest.raises(ScenarioValidationError, match=message):
        convert(text)


def test_cli(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    from urbanflow.cli import main as cli

    src = tmp_path / "map.osm"
    src.write_text(town(), encoding="utf-8")
    out = tmp_path / "city.json"
    with pytest.raises(SystemExit) as info:
        cli.main(["import", "osm", "--osm", str(src), "-o", str(out), "--demand", "0.5"])
    assert info.value.code in (0, None) and out.is_file()
    assert "Wrote" in capsys.readouterr().out
