from __future__ import annotations

import pytest

from urbanflow import generate
from urbanflow.core.errors import ScenarioValidationError
from urbanflow.scenario.derive import shortest_road_path


def test_default_grid_counts_and_validity() -> None:
    s = generate("grid")
    summary = s.summary()
    assert summary["intersections"] == 9 + 12  # nodes + boundaries
    assert summary["roads"] == 2 * (3 * 4 + 3 * 4)  # two-way links along 3 rows and 3 cols
    assert summary["flows"] == 12 * 3  # every entry: far / straight / near
    assert not s.issues or all(i.severity == "warning" for i in s.issues)
    kinds = {ix.id: ix.kind for ix in s.resolved.network.intersections}
    assert all(kinds[f"r{r}c{c}"] == "signalized" for r in range(3) for c in range(3))


def test_od_paths_turn_once_at_the_first_node() -> None:
    s = generate("grid", rows=2, cols=3)
    flows = {f.id: f for f in s.resolved.demand.flows}
    far = flows["bW0_r0c0:far"]  # east-bound, far-side = north
    path = shortest_road_path(s.resolved, far.origin, far.destination)
    assert path == ["bW0_r0c0", "r0c0_r1c0", "r1c0_bN0"]
    straight = flows["bS1_r0c1:straight"]
    assert shortest_road_path(s.resolved, straight.origin, straight.destination) == [
        "bS1_r0c1",
        "r0c1_r1c1",
        "r1c1_bN1",
    ]


def test_non_uniform_spacing() -> None:
    s = generate("grid", rows=2, cols=3, col_spacing=[200.0, 400.0], row_spacing=[250.0])
    pts = {ix.id: tuple(ix.point) for ix in s.spec.network.intersections}
    assert pts["r0c2"] == (600.0, 0.0)
    assert pts["r1c1"] == (200.0, 250.0)


def test_rate_and_ratios() -> None:
    s = generate("grid", rows=1, cols=1, entry_rate=600)
    by_entry: dict[str, float] = {}
    for f in s.resolved.demand.flows:
        assert f.rate is not None
        by_entry[f.origin or ""] = by_entry.get(f.origin or "", 0.0) + f.rate
    assert by_entry and all(v == pytest.approx(600) for v in by_entry.values())


@pytest.mark.parametrize(
    "params",
    [{"rows": 3, "row_spacing": [300.0]}, {"spacing": 15.0}, {"boundary_length": 5.0}],
)
def test_bad_params(params: dict[str, object]) -> None:
    with pytest.raises(ScenarioValidationError):
        generate("grid", **params)
