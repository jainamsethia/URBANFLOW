"""Generator registry, shared helpers and the single_intersection generator (plan E.10)."""

from __future__ import annotations

from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

import urbanflow
from urbanflow.core.errors import NotFoundError, ScenarioValidationError
from urbanflow.scenario import Scenario, ScenarioBuilder
from urbanflow.scenario.generators import (
    BUNDLED_SCENARIOS,
    GENERATORS,
    generate,
    list_generators,
    register_generator,
)
from urbanflow.scenario.generators._common import (
    TurnRatios,
    longest_boundary_path,
    od_flows,
    split_flows_by_profile,
)
from urbanflow.scenario.generators.single_intersection import SingleIntersectionParams
from urbanflow.scenario.io import bundled

REPO = Path(__file__).resolve().parents[3]


class RingParams(BaseModel):
    n: int = 3


def _ring(p: RingParams) -> Scenario:
    """Ring road (test plugin)."""
    b = ScenarioBuilder(f"ring-{p.n}")
    b.boundary("A", (0, 0)).boundary("B", (100, 0)).road("A_B", "A", "B")
    return b.build()


@pytest.fixture
def ring() -> Iterator[None]:
    register_generator("ring", params=RingParams)(_ring)
    yield
    GENERATORS._items.pop("ring", None)


def test_registry_lists_builtins() -> None:
    names = [name for name, _ in list_generators()]
    assert "single_intersection" in names and names == sorted(names)
    entry = GENERATORS.get("single_intersection")
    assert entry.source == "builtin" and entry.params is SingleIntersectionParams
    assert entry.description == "3/4-arm junction, any control kind."


def test_custom_generator_registration(ring: None) -> None:
    entry = GENERATORS.get("ring")
    assert entry.description == "Ring road (test plugin)." and entry.source == "test_generators"
    scenario = generate("ring", n=5)
    assert scenario.name == "ring-5"
    info = scenario.spec.meta.generator
    assert info is not None and info.name == "ring" and dict(info.params) == {"n": 5}
    assert info.urbanflow_version == urbanflow.__version__
    assert scenario.resolved.meta.generator == info


def test_plugins_load_through_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEntryPoint:
        name = "fake"
        value = "fake_pkg.generators"

        def load(self) -> None:
            register_generator("fake_ring", params=RingParams)(_ring)

    def entry_points(group: str) -> list[FakeEntryPoint]:
        assert group == "urbanflow.plugins"
        return [FakeEntryPoint()]

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    monkeypatch.setattr(GENERATORS, "_plugins_loaded", False)
    try:
        assert generate("fake_ring").name == "ring-3"
    finally:
        GENERATORS._items.pop("fake_ring", None)


def test_unknown_generator_lists_the_available_ones() -> None:
    with pytest.raises(NotFoundError) as info:
        generate("single_intersectoin")
    assert '(did you mean "single_intersection"?)' in str(info.value)


@pytest.mark.parametrize(
    ("params", "code", "path"),
    [
        ({"lanes": 0}, "E004", "params.lanes"),
        ({"lanez": 2}, "E002", "params.lanez"),
        ({"kind": "roundabout"}, "E005", "params.kind"),
        ({"turn_ratios": {"far": 0.5}}, "E009", "params.turn_ratios"),
        ({"approach_rates": {"Q": 100}}, "E009", "params"),
        ({"arm_length": 10}, "E009", "params"),
        ({"arrival": "binomial", "demand_rate": 5000}, "E009", "params"),
    ],
)
def test_parameter_errors_have_params_paths(params: dict[str, Any], code: str, path: str) -> None:
    with pytest.raises(ScenarioValidationError) as info:
        generate("single_intersection", params)
    assert [(i.code, i.path) for i in info.value.issues] == [(code, path)]


def test_partial_nested_parameters_update_the_default() -> None:
    scenario = generate("single_intersection", turn_ratios={"far": 0.2, "straight": 0.65})
    info = scenario.spec.meta.generator
    assert info is not None and info.params["turn_ratios"] == {
        "far": 0.2,
        "straight": 0.65,
        "near": 0.15,
    }


@pytest.mark.parametrize("kind", ["signalized", "priority", "uncontrolled"])
@pytest.mark.parametrize("arms", [3, 4])
def test_single_intersection_variants_are_valid(kind: str, arms: int) -> None:
    scenario = generate("single_intersection", kind=kind, arms=arms)
    assert scenario.issues == ()
    junction = scenario.resolved.network.intersections[0]
    assert junction.id == "J" and junction.kind == kind
    assert (junction.signal is not None) == (kind == "signalized")
    counts = scenario.summary()
    assert counts["flows"] == (12 if arms == 4 else 6)
    assert counts["roads"] == 2 * arms
    assert "seed" not in scenario.spec.simulation.model_fields_set


def test_single_intersection_demand() -> None:
    scenario = generate("single_intersection", approach_rates={"N": 900.0})
    flows = {f.id: f for f in scenario.spec.demand.flows}
    assert set(flows) == {f"{a}_to_{b}" for a in "NESW" for b in "NESW" if a != b}
    assert flows["N_to_S"].origin == "N_in" and flows["N_to_S"].destination == "S_out"
    assert flows["N_to_S"].rate == pytest.approx(900 * 0.7)
    assert flows["N_to_E"].rate == pytest.approx(900 * 0.15)  # far side (left) of N
    total = sum(f.rate or 0 for f in flows.values() if f.id.startswith("E_"))
    assert total == pytest.approx(600)
    # a T-junction renormalises the missing turn so each approach keeps its demand
    tee = generate("single_intersection", arms=3)
    east = [f for f in tee.spec.demand.flows if f.id.startswith("E_")]
    assert sum(f.rate or 0 for f in east) == pytest.approx(600)


def test_signal_parameters_reach_the_scenario() -> None:
    scenario = generate("single_intersection", green=25, yellow=4, template="protected_left")
    signal = scenario.spec.network.intersections[0].signal
    assert signal is not None and signal.phases is not None
    assert (signal.yellow, len(signal.phases)) == (4, 4)
    assert {p.duration for p in signal.phases} == {25}


def test_generators_are_deterministic() -> None:
    assert generate("single_intersection").to_json() == generate("single_intersection").to_json()


def test_bundled_scenarios_match_their_generators() -> None:
    for name, (generator, params) in BUNDLED_SCENARIOS.items():
        expected = generate(generator, dict(params)).to_json() + "\n"
        text = bundled(name).read_text(encoding="utf-8")
        assert text == expected, "run: uv run python scripts/regen_bundled_scenarios.py"
        assert Scenario.load(bundled(name)).issues == ()


# ---------------------------------------------------------------------------- _common
def test_turn_ratios_must_sum_to_one() -> None:
    TurnRatios(far=0.2, straight=0.5, near=0.3)
    with pytest.raises(ValidationError, match="must sum to 1"):
        TurnRatios(far=0.2, straight=0.5, near=0.2)


def test_od_flows_renormalise_missing_turns() -> None:
    b = ScenarioBuilder("t")
    b.boundary("A", (0, 0)).boundary("B", (1, 0))
    ratios = TurnRatios(far=0.2, straight=0.6, near=0.2)
    ids = od_flows(b, ["a"], {"a": {"straight": "x", "near": "y"}}, ratios, 800.0)
    assert ids == ["a:straight", "a:near"]
    flows = {f.id: f for f in b._flows.values()}
    assert flows["a:straight"].rate == pytest.approx(600) and flows["a:near"].rate == pytest.approx(
        200
    )
    assert flows["a:near"].arrival == "poisson"


def _corridor() -> Scenario:
    b = ScenarioBuilder("corridor")
    b.boundary("W", (-300, 0)).intersection("J1", (0, 0)).intersection("J2", (300, 0))
    b.boundary("E", (400, 0)).boundary("N1", (0, 100)).boundary("S2", (300, -100))
    b.two_way("W", "J1").two_way("J1", "J2").two_way("J2", "E")
    b.two_way("N1", "J1").two_way("J2", "S2")
    b.flow("f", route=["W_J1", "J1_J2"], rate=100, begin=100, end=700)
    b.flow("capped", route=["W_J1"], rate=100, count=3)
    return b.build()


def test_longest_boundary_path() -> None:
    # W->E and E->W are both 700 m; ties go to the smaller (entry, exit) ids
    assert longest_boundary_path(_corridor().spec) == ["E_J2", "J2_J1", "J1_W"]


def test_split_flows_by_profile_integrates() -> None:
    spec = _corridor().resolved
    flat = split_flows_by_profile(spec, lambda _f, _t: 1.0, 300.0)
    slices = [f for f in flat.demand.flows if f.id.startswith("f:")]
    assert [f.id for f in slices] == ["f:0", "f:1", "f:2"]
    assert [(f.begin, f.end) for f in slices] == [(100, 300), (300, 600), (600, 700)]
    assert sum((f.rate or 0) * (float(f.end or 0) - f.begin) for f in slices) == pytest.approx(
        100 * 600
    )
    assert any(f.id == "capped" for f in flat.demand.flows)  # count-limited flows stay whole

    def peak(_flow: Any, t: float) -> float:
        return max(0.0, 1 - abs(t - 400) / 400)

    shaped = split_flows_by_profile(spec, peak, 100.0)
    parts = [f for f in shaped.demand.flows if f.id.startswith("f:")]
    expected = sum(100 * peak(None, (lo + lo + 100) / 2) * 100 for lo in range(100, 700, 100))
    total = sum((f.rate or 0) * (float(f.end or 0) - f.begin) for f in parts)
    assert total == pytest.approx(expected)


# ---------------------------------------------------------------------------- properties
params_strategy = st.fixed_dictionaries(
    {
        "arms": st.sampled_from([3, 4]),
        "lanes": st.integers(1, 4),
        "arm_length": st.floats(60.0, 1000.0),
        "speed_limit": st.floats(5.0, 30.0),
        "kind": st.sampled_from(["signalized", "priority", "uncontrolled"]),
        "template": st.sampled_from(["auto", "two_phase", "protected_left", "split"]),
        "green": st.integers(5, 90),
        "yellow": st.integers(0, 6),
        "demand_rate": st.floats(10.0, 1200.0),
        "arrival": st.sampled_from(["uniform", "poisson", "binomial"]),
        "turn_ratios": st.sampled_from(
            [{"far": 0.1, "straight": 0.8, "near": 0.1}, {"far": 0.0, "straight": 1.0, "near": 0.0}]
        ),
    }
)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(params_strategy)
def test_parameter_ranges_always_validate(params: dict[str, Any]) -> None:
    scenario = generate("single_intersection", params)
    errors = [i for i in scenario.issues if i.severity == "error"]
    assert errors == []
    warnings = {i.code for i in scenario.issues}
    assert warnings <= {"W303"}  # yellow = 0 is allowed but warned about
