"""Default resolution (plan E.7 §1.5): turns, lane mapping, kinds, signals, idempotence."""

from __future__ import annotations

import logging
import math
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from urbanflow.core.types import DriveSide, IntersectionKind, TurnKind
from urbanflow.scenario import Scenario, ScenarioBuilder, ScenarioSpec, derive
from urbanflow.scenario.derive import (
    SIGNAL_TEMPLATES,
    RoadGraph,
    classify_turns,
    group_approaches,
    map_lanes,
    resolve,
    resolve_vehicle_types,
    road_successors,
    shortest_road_path,
    signal_from_template,
    topology,
    wrap_angle,
)
from urbanflow.scenario.schema import DEFAULT_VEHICLE_TYPES, VehicleTypeSpec
from urbanflow.scenario.validate import validate_spec

L, S, R, U = TurnKind.left, TurnKind.straight, TurnKind.right, TurnKind.uturn
ARM = 200.0


def junction(
    angles: dict[str, float],
    *,
    lanes: int | dict[str, int] = 2,
    drive_side: str = "right",
    mirror: bool = False,
    kind: str | None = None,
    allow_uturns: bool = False,
) -> ScenarioSpec:
    """Centre J with one boundary per arm at ``angle`` degrees and roads ``{arm}_in/_out``."""
    b = ScenarioBuilder("junction", drive_side=drive_side)  # type: ignore[arg-type]
    b.intersection("J", (0.0, 0.0), kind=kind)
    for arm, deg in angles.items():
        x, y = ARM * math.cos(math.radians(deg)), ARM * math.sin(math.radians(deg))
        b.boundary(arm, (-x if mirror else x, y))
        n = lanes[arm] if isinstance(lanes, dict) else lanes
        b.road(f"{arm}_in", arm, "J", lanes=n).road(f"{arm}_out", "J", arm, lanes=n)
    spec = b.to_spec()
    if allow_uturns:
        network = spec.network.model_copy(update={"allow_uturns": True})
        spec = spec.model_copy(update={"network": network})
    return spec


FOUR_WAY = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}


def centre(spec: ScenarioSpec) -> Any:
    return resolve(spec).network.intersections[0]


def lanes_of(spec: ScenarioSpec) -> dict[str, set[int]]:
    return {str(m.id): {c.from_lane for c in m.connections or ()} for m in centre(spec).movements}


# ---------------------------------------------------------------------------- lane mapping
LSR = [("L", 1.5, L), ("S", 0.0, S), ("R", -1.5, R)]
LR = [("L", 1.5, L), ("R", -1.5, R)]
LLR = [("L1", 2.5, L), ("L2", 1.0, L), ("R", -1.5, R)]  # e.g. a 5-way junction, no straight
MAPPING_TABLE = [
    (1, LSR, {"L": {0}, "S": {0}, "R": {0}}),
    (2, LSR, {"L": {0}, "S": {0, 1}, "R": {1}}),
    (3, LSR, {"L": {0}, "S": {1}, "R": {2}}),
    (4, LSR, {"L": {0}, "S": {1, 2}, "R": {3}}),
    (5, LSR, {"L": {0}, "S": {1, 2, 3}, "R": {4}}),
    (1, LR, {"L": {0}, "R": {0}}),
    (2, LR, {"L": {0}, "R": {1}}),
    (3, LR, {"L": {0, 1}, "R": {2}}),
    (4, LR, {"L": {0, 1, 2}, "R": {3}}),
    (5, LR, {"L": {0, 1, 2, 3}, "R": {4}}),
    # n < m: movement i gets round(i (n-1)/(m-1)); i = 1 gives 0.5, which rounds *up* (not
    # to even as Python's round() would)
    (2, LLR, {"L1": {0}, "L2": {1}, "R": {1}}),
    (3, [*LSR, ("R2", -2.5, R)], {"L": {0}, "S": {0, 1, 2}, "R": {1}, "R2": {2}}),
]


@pytest.mark.parametrize("side", [DriveSide.right, DriveSide.left])
@pytest.mark.parametrize(("n", "movements", "expected"), MAPPING_TABLE)
def test_lane_mapping_table(
    n: int,
    movements: list[tuple[str, float, TurnKind]],
    expected: dict[str, set[int]],
    side: DriveSide,
) -> None:
    if side is DriveSide.left:  # mirror image: headings flip sign, left and right swap
        swap = {L: R, R: L, S: S}
        movements = [({"L": "R", "R": "L"}.get(m, m), -d, swap[t]) for m, d, t in movements]
        expected = {{"L": "R", "R": "L"}.get(m, m): v for m, v in expected.items()}
    mapping = map_lanes(n, movements, side, dict.fromkeys(expected, 2))
    assert {m: {a for a, _ in pairs} for m, pairs in mapping.items()} == expected
    for pairs in mapping.values():  # one connection per (source lane, movement)
        assert len({a for a, _ in pairs}) == len(pairs)


def test_target_lanes() -> None:
    mapping = map_lanes(4, LSR, DriveSide.right, {"L": 2, "S": 2, "R": 2})
    assert mapping == {"L": [(0, 0)], "S": [(1, 1), (2, 1)], "R": [(3, 1)]}
    # no straight: extra lanes go to the widest target; a near-side block keeps the curb
    mapping = map_lanes(3, LR, DriveSide.right, {"L": 2, "R": 3})
    assert mapping == {"L": [(0, 0)], "R": [(1, 1), (2, 2)]}
    assert map_lanes(2, [("U", -math.pi, U), *LR], DriveSide.right, {"U": 2, "L": 2, "R": 2}) == {
        "L": [(0, 0)],
        "R": [(1, 1)],
        "U": [(0, 0)],
    }


# ---------------------------------------------------------------------------- turns
def test_four_way_turns() -> None:
    turns = classify_turns(junction(FOUR_WAY), "J")
    assert turns[("N_in", "S_out")] is S
    assert turns[("N_in", "E_out")] is L
    assert turns[("N_in", "W_out")] is R
    assert turns[("N_in", "N_out")] is U
    assert len(turns) == 16


def test_t_junction_turns() -> None:
    turns = classify_turns(junction({"E": 0, "N": 90, "W": 180}), "J")
    assert (turns[("N_in", "E_out")], turns[("N_in", "W_out")]) == (L, R)
    assert turns[("E_in", "W_out")] is S and turns[("E_in", "N_out")] is R
    assert S not in {t for (a, _), t in turns.items() if a == "N_in"}


def test_skewed_junction_turns() -> None:
    turns = classify_turns(junction({"a": 0, "b": 100, "c": 190, "d": 280}), "J")
    assert turns[("a_in", "c_out")] is S  # |delta| = 10 deg
    assert turns[("a_in", "b_out")] is R
    assert turns[("a_in", "d_out")] is L


def test_five_way_straight_tie_breaks_by_id() -> None:
    angles = {f"a{k}": 72.0 * k for k in range(5)}
    turns = classify_turns(junction(angles), "J")
    # from a0 both a2 (-36 deg) and a3 (+36 deg) qualify; the smaller id wins
    assert turns[("a0_in", "a2_out")] is S
    assert turns[("a0_in", "a3_out")] is L
    assert turns[("a0_in", "a1_out")] is R
    assert sum(t is S for (a, _), t in turns.items() if a == "a0_in") == 1


def test_uturns_only_when_allowed() -> None:
    assert "N_in->N_out" not in lanes_of(junction(FOUR_WAY))
    with_u = lanes_of(junction(FOUR_WAY, allow_uturns=True))
    assert with_u["N_in->N_out"] == {0}


@pytest.mark.parametrize(
    ("delta", "turn"),
    [(45.0, S), (-45.0, S), (46.0, L), (-46.0, R), (44.0, S), (-44.0, S)],
)
def test_straight_threshold(delta: float, turn: TurnKind) -> None:
    """|delta| <= 45 deg can be straight (E.7 §1.5 step 5); 46 deg is a turn."""
    # arm a is east of J, so a_in heads west (180 deg); b_out leaves J at 180 + delta
    turns = classify_turns(junction({"a": 0.0, "b": 180.0 + delta}), "J")
    assert turns[("a_in", "b_out")] is turn


def _bent_return(heading: float) -> ScenarioSpec:
    """J with arm a (east); a_out leaves J at ``heading`` deg and bends back to a."""
    b = ScenarioBuilder("uturn")
    b.intersection("J", (0.0, 0.0), kind="uncontrolled").boundary("a", (ARM, 0.0))
    kink = (50 * math.cos(math.radians(heading)), 50 * math.sin(math.radians(heading)))
    b.road("a_in", "a", "J").road("a_out", "J", "a", points=[(0.0, 0.0), kink, (ARM, 0.0)])
    return b.to_spec()


@pytest.mark.parametrize(
    ("heading", "turn"),
    [(45.0, U), (315.0, U), (46.0, R), (314.0, L)],  # |delta| = 135, 135, 134, 134 deg
)
def test_uturn_threshold(heading: float, turn: TurnKind) -> None:
    """A road back to the origin is a U-turn only if |delta| >= 135 deg."""
    assert classify_turns(_bent_return(heading), "J")[("a_in", "a_out")] is turn
    derived = {m.id for m in centre(_bent_return(heading)).movements or ()}
    assert ("a_in->a_out" in derived) is (turn is not U)  # U-turns need allow_uturns


def test_wrap_angle() -> None:
    assert wrap_angle(math.pi) == -math.pi
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert -math.pi <= wrap_angle(-7.0) < math.pi


# ---------------------------------------------------------------------------- kinds, radii
def test_kinds_and_radii() -> None:
    b = ScenarioBuilder("kinds")
    b.boundary("A", (0, 0)).intersection("K", (100, 0)).intersection("B", (200, 0))
    b.intersection("iso", (500, 500))
    b.road("A_K", "A", "K", lanes=[{"width": 3.5}, {"width": 3.0}]).road("K_B", "K", "B")
    topo = topology(b.to_spec())
    assert topo.kinds == {
        "A": IntersectionKind.boundary,
        "K": IntersectionKind.uncontrolled,
        "B": IntersectionKind.boundary,
        "iso": IntersectionKind.boundary,
    }
    assert topo.radii["K"] == pytest.approx(3.5 + 3.0 + 2.0)
    assert topo.radii["A"] == 0.0
    assert topology(junction(FOUR_WAY)).kinds["J"] is IntersectionKind.signalized
    assert topology(junction(FOUR_WAY)).radii["J"] == pytest.approx(8.4)
    assert topology(junction(FOUR_WAY, kind="priority")).kinds["J"] is IntersectionKind.priority


# ---------------------------------------------------------------------------- signals
def test_group_approaches() -> None:
    groups = group_approaches({"a": 0.0, "b": math.pi, "c": math.pi / 2, "d": -math.pi / 2})
    assert groups == [("a", "b"), ("c", "d")]
    # ordered by bearing mod pi: 0, 4 - pi = 0.86, 2
    assert group_approaches({"x": 0.0, "y": 2.0, "z": 4.0}) == [("x",), ("z",), ("y",)]


def _phases(spec: ScenarioSpec) -> list[dict[str, str]]:
    signal = centre(spec).signal
    return [dict(p.green) for p in signal.phases]


def test_two_phase_template() -> None:
    phases = _phases(junction(FOUR_WAY))
    assert len(phases) == 2
    ew, ns = phases
    assert ew["E_in->W_out"] == "G" and ew["E_in->N_out"] == "G" and ew["E_in->S_out"] == "g"
    assert ns == {
        "N_in->S_out": "G",
        "N_in->W_out": "G",
        "N_in->E_out": "g",
        "S_in->N_out": "G",
        "S_in->E_out": "G",
        "S_in->W_out": "g",
    }


def test_other_templates() -> None:
    spec = junction(FOUR_WAY)
    ix = centre(spec)
    bearings = {"E_in": math.pi, "W_in": 0.0, "N_in": -math.pi / 2, "S_in": math.pi / 2}
    groups = group_approaches(bearings)
    protected = signal_from_template(
        ix.signal, groups, ix.movements, DriveSide.right, "protected_left"
    )
    assert [len(p.green) for p in protected.phases] == [4, 4, 4, 4]
    first, second = (dict(p.green) for p in protected.phases[:2])
    assert set(first) == {"E_in->W_out", "E_in->N_out", "W_in->E_out", "W_in->S_out"}
    assert set(second) == {"E_in->S_out", "E_in->N_out", "W_in->N_out", "W_in->S_out"}
    split = signal_from_template(ix.signal, groups, ix.movements, DriveSide.right, "split")
    assert [sorted({m.split("->")[0] for m in p.green}) for p in split.phases] == [
        ["E_in"],
        ["W_in"],
        ["N_in"],
        ["S_in"],
    ]
    assert [p.id for p in split.phases] == ["p0", "p1", "p2", "p3"]


def test_templates_dispatch_through_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """AD.1 8.3: a template is a SIGNAL_TEMPLATES entry (here one all-protected phase)."""
    assert set(SIGNAL_TEMPLATES) == {"two_phase", "protected_left", "split"}

    def one_phase(group: tuple[str, ...], by_road: Any, _far: Any, _near: Any) -> Any:
        return [{str(m.id): "G" for road in group for m in by_road.get(road, ())}]

    monkeypatch.setattr(derive, "SIGNAL_TEMPLATES", {**SIGNAL_TEMPLATES, "split": one_phase})
    ix = centre(junction(FOUR_WAY))
    split = signal_from_template(
        ix.signal,
        [tuple(sorted({m.from_road for m in ix.movements}))],
        ix.movements,
        DriveSide.right,
        "split",
    )
    assert [len(p.green) for p in split.phases] == [12]


def test_auto_template_splits_without_opposing_pairs() -> None:
    phases = _phases(junction({"a": 90, "b": 210, "c": 330}, kind="signalized"))
    assert len(phases) == 3 and all(set(p.values()) == {"G"} for p in phases)


# ---------------------------------------------------------------------------- priority
def test_major_roads_and_priorities() -> None:
    spec = junction(FOUR_WAY, kind="priority", lanes={"E": 3, "W": 3, "N": 1, "S": 1})
    ix = centre(spec)
    assert ix.major_roads == ("E_in", "W_in")
    priority = {m.id: m.priority for m in ix.movements}
    assert priority["E_in->W_out"] == "major" and priority["N_in->S_out"] == "minor"
    assert ix.signal is None


# ---------------------------------------------------------------------------- mirroring
def test_left_hand_traffic_mirrors_right_hand() -> None:
    """Mirroring the geometry (x -> -x) and the driving side keeps every lane assignment
    and swaps left and right turns (a right turn is near-side on the right, and its mirror
    image is a near-side left turn on the left)."""
    right = junction(FOUR_WAY, lanes=3)
    left = junction(FOUR_WAY, lanes=3, drive_side="left", mirror=True)
    assert lanes_of(left) == lanes_of(right)
    assert lanes_of(right)["N_in->W_out"] == {2}  # near side = curb lane
    swap = {L: R, R: L, S: S, U: U}
    turns = classify_turns(left, "J")
    assert {k: swap[t] for k, t in classify_turns(right, "J").items()} == turns
    assert centre(left).signal.phases is not None


# ---------------------------------------------------------------------------- resolution
def test_resolve_fills_every_default(demo_data: dict[str, Any]) -> None:
    resolved = Scenario.from_dict(demo_data).resolved
    roads = {r.id: r for r in resolved.network.roads}
    assert roads["N_in"].points == ((0.0, 200.0), (0.0, 0.0))
    assert roads["W_in"].lanes[1].speed_limit == 11.11 and roads["W_in"].lanes[0].width == 3.2
    types = {vt.id: vt for vt in resolved.vehicle_types}
    assert types["car"].length == 4.5 and types["car"].width == 1.8
    assert types["city_bus"].speed_factor.min == 0.8
    assert [vt.id for vt in resolved.vehicle_types] == sorted(types)
    assert resolved.demand.flows[0].vehicle_type == "car"
    assert resolved.demand.flows[2].vehicle_type is None  # type_mix flow
    stops = resolved.demand.transit[0].stops
    assert [(s.id, s.lane) for s in stops] == [("bus1.0", 1), ("bus1.1", 1)]
    kinds = [ix.kind for ix in resolved.network.intersections]
    assert kinds[0] is IntersectionKind.signalized and kinds[1] is IntersectionKind.boundary
    assert resolved.network.intersections[0].radius == pytest.approx(8.4)


def test_resolve_leaves_skipped_intersections_alone(demo_data: dict[str, Any]) -> None:
    spec = ScenarioSpec.model_validate(demo_data)
    resolved = resolve(spec, skip=frozenset({"J"}))
    assert resolved.network.intersections[0] == spec.network.intersections[0]


def test_explicit_movements_get_derived_details(demo_data: dict[str, Any]) -> None:
    demo_data["network"]["intersections"][0]["movements"] = [
        {"from_road": "N_in", "to_road": "S_out"},
        {"from_road": "N_in", "to_road": "E_out", "connections": [{"from_lane": 1, "to_lane": 1}]},
    ]
    spec = ScenarioSpec.model_validate(demo_data)
    movements = resolve(spec).network.intersections[0].movements
    assert [(m.id, m.turn) for m in movements] == [("N_in->S_out", S), ("N_in->E_out", L)]
    assert [(c.from_lane, c.to_lane) for c in movements[1].connections] == [(1, 1)]
    assert movements[0].connections is not None


def test_invalid_override_merges_keep_the_built_in() -> None:
    """resolve never raises: an override breaking a rule once merged is dropped (E007)."""
    bad = VehicleTypeSpec.model_validate({"id": "emergency", "speed_factor": {"max": 1.25}})
    good = VehicleTypeSpec.model_validate({"id": "truck", "emergency_decel": 1.8})
    types = {vt.id: vt for vt in resolve_vehicle_types([bad, good])}
    assert types["emergency"] == DEFAULT_VEHICLE_TYPES["emergency"]
    assert (types["truck"].decel, types["truck"].emergency_decel) == (1.5, 1.8)


def test_resolve_never_raises_on_long_derived_ids() -> None:
    """Derived ids longer than 128 characters are E006 in validation, not a crash here."""
    long = "r" * 70
    b = ScenarioBuilder("long")
    b.boundary("A", (-100.0, 0.0)).intersection("K", (0.0, 0.0), kind="signalized")
    b.boundary("B", (100.0, 0.0)).road(f"{long}_in", "A", "K").road(f"{long}_out", "K", "B")
    ix = resolve(b.to_spec()).network.intersections[1]
    assert [m.id for m in ix.movements] == [f"{long}_in->{long}_out"]
    assert ix.signal is not None and ix.signal.phases is not None


def test_road_graph(demo_data: dict[str, Any]) -> None:
    resolved = Scenario.from_dict(demo_data).resolved
    successors = road_successors(resolved)
    assert successors["N_in"] == ["E_out", "S_out", "W_out"]
    assert successors["N_out"] == []
    assert shortest_road_path(resolved, "N_in", "S_out") == ["N_in", "S_out"]
    assert shortest_road_path(resolved, "S_out", "N_out") is None
    assert shortest_road_path(resolved, "N_in", "N_in") == ["N_in"]
    assert shortest_road_path(resolved, "N_in", "nope") is None
    graph = RoadGraph(resolved)
    lengths, _ = graph.tree("W_in")
    assert lengths["E_out"] == pytest.approx(400.0)


# ---------------------------------------------------------------------------- properties
@st.composite
def grids(draw: st.DrawFn) -> ScenarioSpec:
    rows, cols = draw(st.integers(1, 3)), draw(st.integers(1, 3))
    spacing = draw(st.floats(60.0, 400.0))
    lanes = draw(st.integers(1, 3))
    side = draw(st.sampled_from(["right", "left"]))
    b = ScenarioBuilder("grid", drive_side=side)  # type: ignore[arg-type]
    for r in range(rows):
        for c in range(cols):
            b.intersection(f"J{r}_{c}", (c * spacing, r * spacing))
    for r in range(rows):
        b.boundary(f"W{r}", (-spacing / 2, r * spacing))
        b.boundary(f"E{r}", ((cols - 0.5) * spacing, r * spacing))
        b.two_way(f"W{r}", f"J{r}_0", lanes=lanes).two_way(f"J{r}_{cols - 1}", f"E{r}", lanes=lanes)
        for c in range(cols - 1):
            b.two_way(f"J{r}_{c}", f"J{r}_{c + 1}", lanes=lanes)
    for c in range(cols):
        b.boundary(f"S{c}", (c * spacing, -spacing / 2))
        b.boundary(f"N{c}", (c * spacing, (rows - 0.5) * spacing))
        b.two_way(f"S{c}", f"J0_{c}", lanes=lanes).two_way(f"J{rows - 1}_{c}", f"N{c}", lanes=lanes)
        for r in range(rows - 1):
            b.two_way(f"J{r}_{c}", f"J{r + 1}_{c}", lanes=lanes)
    b.flow("across", origin="W0_J0_0", destination=f"J0_{cols - 1}_E0", rate=300)
    spec = b.to_spec()
    if draw(st.booleans()):
        network = spec.network.model_copy(update={"allow_uturns": True})
        spec = spec.model_copy(update={"network": network})
    return spec


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(grids())
def test_resolve_is_idempotent_and_valid(spec: ScenarioSpec) -> None:
    once = resolve(spec)
    assert resolve(once) == once
    report = validate_spec(spec)
    assert report.issues == (), [str(i) for i in report.issues]


def test_derive_decisions_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    """AB 6.4: DEBUG per intersection with kind, radius, n_movements and template."""
    with caplog.at_level(logging.DEBUG, logger="urbanflow.scenario.derive"):
        resolve(junction({"a": 90, "b": 210, "c": 330}, kind="signalized"))
    (record,) = [r for r in caplog.records if r.__dict__.get("intersection") == "J"]
    fields = record.__dict__
    assert (fields["kind"], fields["n_movements"], fields["template"]) == ("signalized", 6, "split")
    assert fields["radius"] > 0
