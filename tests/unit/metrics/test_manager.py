"""Metrics manager: analytic cases, conservation, sampling, warm-up and export (plan J)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from urbanflow import ScenarioBuilder, Simulation, bundled, generate
from urbanflow.metrics import read_table


def _straight(length: float = 500.0) -> Simulation:
    b = ScenarioBuilder("straight", duration=120)
    b.boundary("A", (0.0, 0.0)).boundary("B", (length, 0.0))
    b.road("AB", "A", "B", lanes=1)
    b.trip("probe", 0.0, route=["AB"])
    return Simulation(b.build(), debug_checks=True)


def test_single_free_vehicle_has_no_delay() -> None:
    """AT-21 (single vehicle): free road -> delay below 0.5 s, travel ~ L / v0."""
    sim = _straight()
    r = sim.run()
    trips = r.tables["trips"]
    assert trips["vehicle_id"].tolist() == ["probe"]
    assert abs(float(trips["delay"][0])) < 0.5
    v0 = float(trips["distance_m"][0] / (trips["travel_time"][0] - trips["delay"][0]))
    assert v0 == pytest.approx(13.89, rel=0.25)  # within the speed-factor range
    assert trips["stops"][0] == 0 and trips["waiting_time"][0] == 0
    assert r.summary["delay.mean"] == pytest.approx(float(trips["delay"][0]))


def test_conservation_and_sampling() -> None:
    sim = Simulation.from_scenario(bundled("grid_3x3"), seed=2, duration=600)
    r = sim.run()
    ts, trips = r.tables["timeseries"], r.tables["trips"]
    assert np.allclose(ts["time"], np.arange(10, 601, 10))  # metrics.interval = 10 s
    assert trips["uid"].size == r.summary["vehicles.arrived"] == ts["arrived"][-1]
    assert ts["arrived_w"].sum() == ts["arrived"][-1]
    assert ts["inserted_w"].sum() == r.summary["vehicles.inserted"]
    s = r.summary
    assert (
        s["vehicles.generated"]
        == s["vehicles.arrived"]
        + s["vehicles.en_route"]
        + s["vehicles.backlog"]
        + s["vehicles.removed"]
    )
    assert np.all(trips["travel_time"] > 0) and np.all(trips["delay"] > -1e-6)
    assert s["vht"] == pytest.approx(ts["vht_w"].sum())
    assert s["vkt"] == pytest.approx(ts["vkt_w"].sum(), rel=1e-9)
    ints = r.tables["intersections"]
    assert set(ints["intersection"].tolist()) == {f"r{i}c{j}" for i in range(3) for j in range(3)}
    assert ints["crossings_w"].sum() > s["vehicles.arrived"]  # every trip crosses >= 1 node


def test_warmup_excludes_early_trips() -> None:
    scen = generate("single_intersection", kind="uncontrolled", duration=600)
    full = Simulation(scen, seed=1).run().summary
    late = Simulation(scen, seed=1, metrics={"warmup": 300}).run().summary
    assert late["vehicles.arrived"] == full["vehicles.arrived"]  # counts stay totals
    assert late["vht"] < full["vht"]
    assert late["travel_time.mean"] != full["travel_time.mean"]


@pytest.mark.parametrize("fmt", ["csv", "json", "parquet"])
def test_export_round_trip(tmp_path: Path, fmt: str) -> None:
    if fmt == "parquet":
        pytest.importorskip("polars")
    r = Simulation.from_scenario(bundled("single_intersection"), duration=120).run()
    paths = r.export(tmp_path, format=fmt)  # type: ignore[arg-type]
    assert sorted(p.name for p in paths) == sorted(f"{t}.{fmt}" for t in r.tables)
    back = read_table(tmp_path / f"trips.{fmt}")
    for col in ("travel_time", "delay", "uid"):
        assert np.allclose(back[col].astype(float), r.tables["trips"][col].astype(float))
    assert back["vehicle_id"].tolist() == r.tables["trips"]["vehicle_id"].tolist()


def test_saved_result_reloads_tables(tmp_path: Path) -> None:
    from urbanflow import SimulationResult

    r = Simulation.from_scenario(bundled("single_intersection"), duration=60).run()
    r.save(tmp_path)
    again = SimulationResult.load(tmp_path)
    assert np.allclose(again.tables["timeseries"]["active"], r.tables["timeseries"]["active"])


def test_latest_and_history() -> None:
    sim = Simulation.from_scenario(bundled("single_intersection"), duration=600)
    assert sim.metrics.latest() == {}
    sim.run()
    assert sim.metrics.latest()["time"] == 600
    hist = sim.metrics.history(max_points=7)
    assert hist["time"].size == 7 and hist["time"][0] == 10 and hist["time"][-1] == 600
