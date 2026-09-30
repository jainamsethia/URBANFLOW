"""Validation pipeline: one table row per issue code, cascades, formatter and docs (plan E.8)."""

from __future__ import annotations

import copy
import importlib.util
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from urbanflow.core.constants import MAX_SCENARIO_BYTES
from urbanflow.core.errors import ScenarioValidationError, Severity, ValidationIssue
from urbanflow.network import compile_network
from urbanflow.scenario import Scenario, ScenarioBuilder
from urbanflow.scenario.io import parse_json
from urbanflow.scenario.schema import DEFAULT_VEHICLE_TYPES
from urbanflow.scenario.validate import (
    ISSUE_CODES,
    ValidationReport,
    issue,
    validate_data,
    validate_spec,
)

REPO = Path(__file__).resolve().parents[3]
Data = dict[str, Any]
Path_ = tuple[str | int, ...]
Producer = Callable[[Data], Sequence[ValidationIssue]]

# Registry-backed deep-check codes (urbanflow.check), which arrive with the engine (plan T:
# P2+); they are defined in ISSUE_CODES but not produced yet. The compiler's X codes (E806,
# E905, W302, W304) are covered below through network.compile_network.
DEFERRED_X_CODES = frozenset({"E901", "E902", "E903", "E904"})


# ---------------------------------------------------------------------------- mutation helpers
def _parent(data: Any, path: Path_) -> Any:
    for key in path[:-1]:
        data = data[key]
    return data


def put(path: Path_, value: Any) -> Callable[[Data], None]:
    def run(data: Data) -> None:
        _parent(data, path)[path[-1]] = value

    return run


def drop(path: Path_) -> Callable[[Data], None]:
    def run(data: Data) -> None:
        del _parent(data, path)[path[-1]]

    return run


def add(path: Path_, value: Any) -> Callable[[Data], None]:
    def run(data: Data) -> None:
        _parent(data, path)[path[-1]].append(copy.deepcopy(value))

    return run


def explicit_movements(data: Data) -> None:
    """Make J's derived movements explicit (so single movements can be broken)."""
    resolved = Scenario.from_dict(copy.deepcopy(data)).resolved
    movements = resolved.network.intersections[0].movements or ()
    data["network"]["intersections"][0]["movements"] = [
        m.model_dump(mode="json", exclude_none=True) for m in movements
    ]


def chain(*steps: Callable[[Data], None]) -> Callable[[Data], None]:
    def run(data: Data) -> None:
        for step in steps:
            step(data)

    return run


def mutate(*steps: Callable[[Data], None]) -> Producer:
    def run(data: Data) -> Sequence[ValidationIssue]:
        chain(*steps)(data)
        return validate_data(data).issues

    return run


def compiled(*steps: Callable[[Data], None]) -> Producer:
    """Issues of compiling the mutated document (deep stage X, network compiler)."""

    def run(data: Data) -> Sequence[ValidationIssue]:
        chain(*steps)(data)
        try:
            return compile_network(Scenario.from_dict(data)).report.warnings
        except ScenarioValidationError as exc:
            return exc.issues

    return run


def text(raw: str) -> Producer:
    def run(_data: Data) -> Sequence[ValidationIssue]:
        try:
            return validate_data(parse_json(raw)).issues
        except ScenarioValidationError as exc:
            return exc.issues

    return run


def builder(op: Callable[[ScenarioBuilder], object]) -> Producer:
    def run(data: Data) -> Sequence[ValidationIssue]:
        b = Scenario.from_dict(data).edit()
        with pytest.raises(ScenarioValidationError) as info:
            op(b)
        return info.value.issues

    return run


J: Path_ = ("network", "intersections", 0)
SIG: Path_ = (*J, "signal")
ROADS: Path_ = ("network", "roads")
INTS: Path_ = ("network", "intersections")
FLOWS: Path_ = ("demand", "flows")
TRIPS: Path_ = ("demand", "trips")
LINE: Path_ = ("demand", "transit", 0)
MOVS = "network.intersections[0].movements"
SIGNAL = "network.intersections[0].signal"

ISOLATED = {"id": "Z", "point": [1000, 1000]}
SIDE_NETWORK = (  # A -> K -> B, K derives to "uncontrolled" with movement "A_K->K_B"
    add(INTS, {"id": "A", "point": [1000, 0]}),
    add(INTS, {"id": "K", "point": [1100, 0]}),
    add(INTS, {"id": "B", "point": [1200, 0]}),
    add(ROADS, {"id": "A_K", "from": "A", "to": "K", "lanes": [{}]}),
    add(ROADS, {"id": "K_B", "from": "K", "to": "B", "lanes": [{}]}),
)


def _without_n_in(data: Data) -> None:
    ix = data["network"]["intersections"][0]
    ix["movements"] = [m for m in ix["movements"] if m["from_road"] != "N_in"]
    for phase in ix["signal"]["phases"]:
        phase["green"] = {k: v for k, v in phase["green"].items() if not k.startswith("N_in")}
    data["demand"]["flows"].pop(0)  # its route N_in -> S_out would lose its movement


def _lane_one_unused(data: Data) -> None:
    for mov in data["network"]["intersections"][0]["movements"]:
        if mov["id"] == "N_in->S_out":
            mov["connections"] = [{"from_lane": 0, "to_lane": 0}]
        if mov["id"] == "N_in->W_out":
            mov["connections"] = [{"from_lane": 0, "to_lane": 1}]


def _bulge(data: Data) -> None:
    """N_in lane 0 bulges east across the opposing straight and back: they cross twice."""
    explicit_movements(data)
    _movement(data, "N_in->S_out")["connections"][0]["shape"] = [[-1.6, 8.4], [4, 0], [-1.6, -8.4]]


HAIRPIN = (  # a 2 m wide right-hand hairpin: the 1.6 m lane offset folds back (E905)
    add(INTS, {"id": "H1", "point": [1000, 0]}),
    add(INTS, {"id": "H2", "point": [1000, -30]}),
    add(
        ROADS,
        {
            "id": "H",
            "from": "H1",
            "to": "H2",
            "points": [[1000, 0], [1060, 0], [1060, -2], [1040, -2], [1040, -30], [1000, -30]],
            "lanes": [{}],
        },
    ),
)
LONG = "x" * 70
LONG_IDS = (  # A -> K -> B with two 70-character road ids through K (derived movements)
    add(INTS, {"id": "A", "point": [1000, 0]}),
    add(INTS, {"id": "K", "point": [1100, 0]}),
    add(INTS, {"id": "B", "point": [1200, 0]}),
    add(ROADS, {"id": f"{LONG}_in", "from": "A", "to": "K", "lanes": [{}]}),
    add(ROADS, {"id": f"{LONG}_out", "from": "K", "to": "B", "lanes": [{}]}),
)
SHORT_LINK = (  # A2 -> K -> B2 with a 13.8 m lane on A2_K: too short for a bus (14.5 m)
    add(INTS, {"id": "A2", "point": [1000, 0]}),
    add(INTS, {"id": "K", "point": [1019, 0]}),
    add(INTS, {"id": "B2", "point": [1200, 0]}),
    add(ROADS, {"id": "A2_K", "from": "A2", "to": "K", "lanes": [{}]}),
    add(ROADS, {"id": "K_B2", "from": "K", "to": "B2", "lanes": [{}]}),
    add(FLOWS, {"id": "buses", "route": ["A2_K", "K_B2"], "vehicle_type": "bus", "rate": 10}),
)


def _movement(data: Data, mid: str) -> Data:
    return next(m for m in data["network"]["intersections"][0]["movements"] if m["id"] == mid)


@dataclass(frozen=True)
class Case:
    code: str
    produce: Producer
    expected: frozenset[tuple[str, str]]
    variant: str = ""


def case(code: str, produce: Producer, *expected: tuple[str, str], variant: str = "") -> Case:
    return Case(code, produce, frozenset(expected), variant)


CASES = [
    # ---- L: load
    case("E000", text('{\n  "format": 1,\n'), ("E000", "$")),
    case("E000", text('{"a": NaN}'), ("E000", "$"), variant="nan"),
    case("E010", mutate(put(("format",), "cityflow")), ("E010", "format")),
    case("E011", mutate(put(("version",), "1.9")), ("E011", "version")),
    case("E012", mutate(put(("version",), "0.1")), ("E012", "version")),
    case("E013", text(" " * (MAX_SCENARIO_BYTES + 1)), ("E013", "$")),
    # ---- S: structural
    case("E001", mutate(drop(("meta", "name"))), ("E001", "meta.name")),
    case("E002", mutate(put(("meta", "nmae"), "x")), ("E002", "meta.nmae")),
    case("E003", mutate(put(("simulation", "seed"), "abc")), ("E003", "simulation.seed")),
    case(
        "E004",
        mutate(put((*ROADS, 0, "lanes", 0, "width"), 9)),
        ("E004", "network.roads[0].lanes[0].width"),
    ),
    case(
        "E004",
        mutate(put((*ROADS, 0, "lanes"), [])),
        ("E004", "network.roads[0].lanes"),
        variant="items",
    ),
    case("E005", mutate(put(("network", "drive_side"), "middle")), ("E005", "network.drive_side")),
    case("E006", mutate(put((*ROADS, 0, "id"), "bad id")), ("E006", "network.roads[0].id")),
    case("E007", mutate(put((*FLOWS, 0, "period"), 5)), ("E007", "demand.flows[0]")),
    case(
        "E009",
        mutate(put(("vehicle_types", 0, "color"), "#12")),
        ("E009", "vehicle_types[0].color"),
    ),
    case(
        "E006",
        mutate(*LONG_IDS),
        ("E006", "network.roads[8].id"),  # the derived "{from}->{to}" id would be too long
        variant="derived",
    ),
    case(
        "E006",
        mutate(
            *LONG_IDS,
            put((*INTS, 6, "movements"), [{"from_road": f"{LONG}_in", "to_road": f"{LONG}_out"}]),
        ),
        ("E006", "network.intersections[6].movements[0]"),  # explicit, id left to derive
        variant="explicit",
    ),
    case(
        "E006",
        mutate(put((*LINE, "id"), "x" * 127)),  # derived "{line}.{k}" is 129 characters
        ("E006", "demand.transit[0].stops[0]"),
        ("E006", "demand.transit[0].stops[1]"),
        variant="stop",
    ),
    case(
        "E007",
        mutate(add(("vehicle_types",), {"id": "emergency", "speed_factor": {"max": 1.25}})),
        ("E007", "vehicle_types[2].speed_factor"),  # merged: min 1.3 > max 1.25
        variant="merged-factor",
    ),
    case(
        "E007",
        mutate(add(("vehicle_types",), {"id": "truck", "emergency_decel": 1.2})),
        ("E007", "vehicle_types[2]"),  # merged: decel 1.5 > 1.2
        variant="merged-decel",
    ),
    # ---- B: builder
    case("E021", builder(lambda b: b.remove("road", "N_inn")), ("E021", "id")),
    case("E022", builder(lambda b: b.remove("road", "N_in")), ("E022", "id")),
    case("E024", builder(lambda b: b.update("road", "N_in", id="X")), ("E024", "id")),
    # ---- P: pre-derive
    case(
        "E101",
        mutate(add(INTS, {"id": "N", "point": [0, 400]})),
        ("E101", "network.intersections[5].id"),
    ),
    case(
        "E102",
        mutate(add(ROADS, {"id": "N_in", "from": "J", "to": "N", "lanes": [{}]})),
        ("E102", "network.roads[8].id"),
    ),
    case("E103", mutate(put((*ROADS, 0, "from"), "NN")), ("E103", "network.roads[0].from")),
    case("E104", mutate(put((*ROADS, 0, "from"), "J")), ("E104", "network.roads[0]")),
    case(
        "E105",
        mutate(
            add(INTS, {"id": "X1", "point": [1000, 0]}),
            add(INTS, {"id": "X2", "point": [1100, 0]}),
            add(ROADS, {"id": "N_in_0", "from": "X1", "to": "X2", "lanes": [{}]}),
        ),
        ("E105", "network.roads[8].id"),
    ),
    case(
        "E106", mutate(put((*INTS, 1, "signal"), {})), ("E106", "network.intersections[1].signal")
    ),
    case(
        "E107",
        mutate(put((*INTS, 1, "movements"), [])),
        ("E107", "network.intersections[1].movements"),
    ),
    case(
        "E108",
        mutate(
            put((*J, "kind"), "priority"), drop(SIG), put((*J, "major_roads"), ["N_in", "N_out"])
        ),
        ("E108", "network.intersections[0].major_roads[1]"),
    ),
    case("E401", mutate(add(("vehicle_types",), {"id": "car"})), ("E401", "vehicle_types[2].id")),
    case(
        "E801",
        mutate(put((*ROADS, 6, "points"), [[-200, 0], [-100, 0], [-100, 0], [0, 0]])),
        ("E801", "network.roads[6].points[2]"),
    ),
    case(
        "E802",
        mutate(put((*INTS, 1, "point"), [0, 10])),
        ("E802", "network.roads[0]"),
        ("E802", "network.roads[1]"),
    ),
    case("W101", mutate(add(INTS, ISOLATED)), ("W101", "network.intersections[5]")),
    case(
        "W102",
        mutate(put((*J, "major_roads"), ["N_in", "S_in"])),
        ("W102", "network.intersections[0].major_roads"),
    ),
    case(
        "W103",
        mutate(
            add(INTS, {"id": "Z1", "point": [1000, 1000], "kind": "uncontrolled", "radius": 10}),
            add(INTS, {"id": "Z2", "point": [1005, 1000], "kind": "uncontrolled", "radius": 10}),
        ),
        ("W101", "network.intersections[5]"),
        ("W101", "network.intersections[6]"),
        ("W103", "network.intersections[6]"),
    ),
    case(
        "W801",
        mutate(put((*ROADS, 6, "points"), [[-200, 0], [-100, 0], [-20, 0]])),
        ("W801", "network.roads[6].points"),
    ),
    case(
        "W802",
        mutate(put((*ROADS, 6, "points"), [[-200, 0], [-50, 0], [-50, 5], [0, 0]])),
        ("W802", "network.roads[6].points[1]"),
        ("W802", "network.roads[6].points[2]"),
    ),
    # ---- Q: movements
    case(
        "E201",
        mutate(
            explicit_movements, add((*J, "movements"), {"from_road": "N_out", "to_road": "S_out"})
        ),
        ("E201", f"{MOVS}[12].from_road"),
    ),
    case(
        "E202",
        mutate(
            explicit_movements, add((*J, "movements"), {"from_road": "N_in", "to_road": "S_in"})
        ),
        ("E202", f"{MOVS}[12].to_road"),
    ),
    case(
        "E203",
        mutate(
            explicit_movements, add((*J, "movements"), {"from_road": "N_in", "to_road": "S_ot"})
        ),
        ("E203", f"{MOVS}[12].to_road"),
    ),
    case(
        "E204",
        mutate(
            explicit_movements,
            add((*J, "movements"), {"id": "N_in->S_out", "from_road": "N_in", "to_road": "N_out"}),
        ),
        ("E204", f"{MOVS}[12].id"),
    ),
    case(
        "E205",
        mutate(
            explicit_movements,
            add((*J, "movements"), {"id": "dup", "from_road": "N_in", "to_road": "S_out"}),
        ),
        ("E205", f"{MOVS}[12]"),
    ),
    case(
        "E206",
        mutate(explicit_movements, put((*J, "movements", 0, "connections", 0, "from_lane"), 5)),
        ("E206", f"{MOVS}[0].connections[0].from_lane"),
    ),
    case(
        "E207",
        mutate(explicit_movements, put((*J, "movements", 0, "connections", 0, "to_lane"), 7)),
        ("E207", f"{MOVS}[0].connections[0].to_lane"),
    ),
    case(
        "E208",
        mutate(
            explicit_movements,
            add((*J, "movements", 2, "connections"), {"from_lane": 0, "to_lane": 0}),
        ),
        ("E208", f"{MOVS}[2].connections[2]"),
    ),
    case("W201", mutate(explicit_movements, _without_n_in), ("W201", "network.intersections[0]")),
    case(
        "W202", mutate(explicit_movements, _lane_one_unused), ("W202", "network.intersections[0]")
    ),
    # ---- Q: signals
    case(
        "E301",
        mutate(add(INTS, {**ISOLATED, "kind": "signalized"})),
        ("W101", "network.intersections[5]"),
        ("E301", "network.intersections[5].signal"),
    ),
    case(
        "E302",
        mutate(
            drop((*SIG, "phases", 1, "green", "E_in->W_out")),
            put((*SIG, "phases", 1, "green", "E_in->W_ot"), "G"),
        ),
        ("E302", f"{SIGNAL}.phases[1]"),
        ("W301", SIGNAL),
    ),
    case(
        "E303",
        mutate(*SIDE_NETWORK, put((*SIG, "phases", 0, "green", "A_K->K_B"), "G")),
        ("E303", f"{SIGNAL}.phases[0]"),
    ),
    case("E304", mutate(put((*SIG, "phases", 1, "id"), "NS")), ("E304", f"{SIGNAL}.phases[1].id")),
    case("E305", mutate(put((*SIG, "initial_phase"), 2)), ("E305", f"{SIGNAL}.initial_phase")),
    case(
        "E306",
        mutate(
            put((*SIG, "phases", 0, "min_green"), 50), put((*SIG, "phases", 0, "max_green"), 40)
        ),
        ("E306", f"{SIGNAL}.phases[0]"),
    ),
    case(
        "E306",
        mutate(put((*SIG, "phases", 0, "duration"), 3)),
        ("E306", f"{SIGNAL}.phases[0]"),
        variant="duration",
    ),
    case("W301", mutate(drop((*SIG, "phases", 0, "green", "N_in->S_out"))), ("W301", SIGNAL)),
    case("W303", mutate(put((*SIG, "yellow"), 0)), ("W303", f"{SIGNAL}.yellow")),
    case("W305", mutate(put(("simulation", "dt"), 0.4)), ("W305", SIGNAL)),
    # ---- Q: demand
    case(
        "E402",
        mutate(put((*TRIPS, 0, "vehicle_type"), "cr")),
        ("E402", "demand.trips[0].vehicle_type"),
    ),
    case(
        "E402",
        mutate(put((*FLOWS, 2, "type_mix"), {"car": 1, "tram": 1})),
        ("E402", 'demand.flows[2].type_mix["tram"]'),
        variant="type_mix",
    ),
    case("E501", mutate(put((*TRIPS, 0, "id"), "ns")), ("E501", "demand.trips[0].id")),
    case(
        "E501",
        mutate(put((*TRIPS, 0, "id"), "ns.3")),
        ("E501", "demand.trips[0].id"),
        variant="collision",
    ),
    case("E502", mutate(put((*FLOWS, 1, "origin"), "S_inn")), ("E502", "demand.flows[1].origin")),
    case(
        "E503",
        mutate(put((*FLOWS, 3, "route"), ["E_in", "E_out"])),
        ("E503", "demand.flows[3].route[1]"),
    ),
    case(
        "E503",
        mutate(put((*FLOWS, 3, "route"), ["N_in", "E_in"])),
        ("E503", "demand.flows[3].route[1]"),
        variant="gap",
    ),
    case(
        "E504", mutate(put((*FLOWS, 1, "origin"), "S_out")), ("E504", "demand.flows[1].destination")
    ),
    case(
        "E505", mutate(put((*FLOWS, 0, "depart_lane"), 5)), ("E505", "demand.flows[0].depart_lane")
    ),
    case(
        "E506",
        mutate(put((*FLOWS, 2, "rate"), 5000)),
        ("E506", "demand.flows[2].rate"),
        ("W501", "demand.flows[2].rate"),
    ),
    case(
        "E507",
        mutate(put(("simulation", "duration"), None)),
        ("E507", "demand.flows[0]"),
        ("E507", "demand.flows[1]"),
        ("E507", "demand.flows[3]"),
    ),
    case("W501", mutate(put((*FLOWS, 0, "rate"), 5000)), ("W501", "demand.flows[0].rate")),
    case("W502", mutate(put((*TRIPS, 0, "depart"), 4000)), ("W502", "demand.trips[0].depart")),
    # ---- Q: transit
    case(
        "E601",
        mutate(put((*LINE, "stops", 1, "road"), "N_in")),
        ("E601", "demand.transit[0].stops[1].road"),
    ),
    case(
        "E602",
        mutate(
            put(
                (*LINE, "stops"),
                [{"road": "E_out", "position": 60}, {"road": "W_in", "position": 120}],
            )
        ),
        ("E602", "demand.transit[0].stops[1]"),
    ),
    case(
        "E604",
        mutate(put((*LINE, "stops", 0, "lane"), 5)),
        ("E604", "demand.transit[0].stops[0].lane"),
    ),
    case(
        "E606",
        mutate(put((*LINE, "stops", 1, "id"), "bus1.0")),
        ("E606", "demand.transit[0].stops[1].id"),
    ),
    case(
        "E807",
        mutate(put((*LINE, "stops", 1, "position"), 250)),
        ("E807", "demand.transit[0].stops[1].position"),
    ),
    case(
        "W306",
        mutate(
            put((*LINE, "route"), ["W_in", "N_out"]),
            put((*LINE, "stops"), [{"road": "W_in", "position": 150}]),
        ),
        ("W306", "demand.transit[0].stops[0]"),
    ),
    # ---- Q: simulation
    case("W701", mutate(put(("simulation", "dt"), 0.1)), ("W701", "simulation.dt")),
    case("W702", mutate(put(("simulation", "duration"), 3600.5)), ("W702", "simulation.duration")),
    # ---- X: network compiler
    case("E806", compiled(*SHORT_LINK), ("E806", "network.roads[8].lanes[0]")),
    case("E905", compiled(*HAIRPIN), ("E905", "network.roads[8]")),
    case(
        "W302",
        compiled(put((*SIG, "phases", 0, "green", "N_in->E_out"), "G")),
        ("W302", f"{SIGNAL}.phases[0]"),
    ),
    case(
        "W304",
        compiled(_bulge),
        ("W304", "network.intersections[0]"),
        ("W302", f"{SIGNAL}.phases[0]"),  # the two protected straights now cross
    ),
]


def _id(c: Case) -> str:
    return f"{c.code}-{c.variant}" if c.variant else c.code


def test_demo_is_valid(demo_data: Data) -> None:
    report = validate_data(demo_data)
    assert report.issues == ()
    assert report.ok and report.spec is not None and report.resolved is not None


def test_w305_counts_only_the_intergreens_that_happen(demo_data: Data) -> None:
    """Review repro: NS -> NS_PLUS loses no green (no yellow, no all-red), so at dt 0.4 the
    realised cycle is 90 + 2 (3.2 + 1.2) = 98.8 s against 98 s (not 103.2 / 102), which is
    the fixed-time controller's C_q."""
    from urbanflow.signals import build_programs
    from urbanflow.signals.controllers import realised_cycle

    signal = demo_data["network"]["intersections"][0]["signal"]
    ns, ew = signal["phases"]
    plus = {**copy.deepcopy(ns), "id": "NS_PLUS"}
    plus["green"]["E_in->S_out"] = "g"  # a superset of NS
    signal["phases"] = [ns, plus, ew]
    for phase in signal["phases"]:
        phase["duration"] = 30
    demo_data["simulation"]["dt"] = 0.4
    w305 = [i for i in validate_data(demo_data).issues if i.code == "W305"]
    assert [i.message for i in w305] == [
        "phase durations, yellow or all_red are not multiples of dt=0.4 s; the realised "
        "cycle is 98.8 s instead of 98.0 s"
    ]
    scenario = Scenario.from_dict(demo_data)
    (prog,) = build_programs(compile_network(scenario), scenario.resolved.network)
    assert realised_cycle(prog, 0.4) == pytest.approx(98.8)
    # without the superset phase every change has an intergreen: 60 + 2 (3.2 + 1.2)
    signal["phases"] = [ns, ew]
    w305 = [i for i in validate_data(demo_data).issues if i.code == "W305"]
    assert "the realised cycle is 68.8 s instead of 68.0 s" in w305[0].message


@pytest.mark.parametrize("c", CASES, ids=_id)
def test_issue_code(c: Case, demo_data: Data) -> None:
    found = c.produce(demo_data)
    assert {(i.code, i.path) for i in found} == c.expected
    assert len(found) == len(c.expected), [str(i) for i in found]
    for i in found:
        assert i.severity is ISSUE_CODES[i.code].severity
        assert i.message and not re.search(r"\{[A-Za-z_]\w*\}", i.message)  # all filled


def test_every_code_is_covered() -> None:
    covered = {c.code for c in CASES}
    missing = set(ISSUE_CODES) - covered - DEFERRED_X_CODES
    assert not missing, f"no test case for: {sorted(missing)}"
    assert {ISSUE_CODES[c].stage for c in DEFERRED_X_CODES} == {"X"}
    assert set(ISSUE_CODES) >= DEFERRED_X_CODES


def test_issue_formats_registered_templates() -> None:
    found = issue("E104", ("network", "roads", 3), id="J")
    assert found == ValidationIssue(
        "network.roads[3]", 'road starts and ends at intersection "J"', "E104", Severity.error
    )
    alt = issue("E306", "p", variant=1, d=3.0, g=5.0)
    assert alt.message == "duration 3.0 s is below min_green 5.0 s"
    assert issue("W701", "simulation.dt", dt=1.5).severity is Severity.warning


def test_one_bad_reference_gives_one_error(demo_data: Data) -> None:
    demo_data["network"]["roads"][0]["to"] = "JJ"  # N_in: flows, trips and J all touch it
    report = validate_data(demo_data)
    assert [(i.code, i.path) for i in report.issues] == [("E103", "network.roads[0].to")]
    assert 'unknown intersection "JJ" (did you mean "J"?)' in report.issues[0].message


def test_broken_explicit_movement_gives_one_error(demo_data: Data) -> None:
    """A typo in a movement's from_road is E203 only: no E302 for the phase entry naming the
    movement's old id (which would come with a misleading did-you-mean hint), and no
    E503/E504 for routes that need the movement."""
    explicit_movements(demo_data)
    movements = demo_data["network"]["intersections"][0]["movements"]
    k = next(i for i, m in enumerate(movements) if m["id"] == "N_in->E_out")
    del movements[k]["id"]
    movements[k]["from_road"] = "N_ni"
    demo_data["demand"]["flows"] += [  # routed through the broken movement: no E503/E504
        {"id": "ne_route", "route": ["N_in", "E_out"], "rate": 60},
        {"id": "ne_od", "origin": "N_in", "destination": "E_out", "rate": 60},
    ]
    report = validate_data(demo_data)
    assert [(i.code, i.path) for i in report.issues] == [("E203", f"{MOVS}[{k}].from_road")]


@pytest.mark.parametrize(
    "vehicle_type",
    [
        {"id": "emergency", "speed_factor": {"max": 1.25}},
        {"id": "truck", "emergency_decel": 1.2},
        {"id": "bus", "speed_factor": {"min": 1.5}},
    ],
)
def test_invalid_built_in_overrides_are_reported_not_raised(
    demo_data: Data, vehicle_type: Data
) -> None:
    demo_data["vehicle_types"].append(vehicle_type)
    report = validate_data(demo_data)  # must never raise (E.8 §2.1)
    assert [i.code for i in report.issues] == ["E007"]
    assert report.resolved is not None
    resolved = next(vt for vt in report.resolved.vehicle_types if vt.id == vehicle_type["id"])
    assert resolved == DEFAULT_VEHICLE_TYPES[vehicle_type["id"]]  # the override is dropped


def test_valid_partial_built_in_overrides(demo_data: Data) -> None:
    demo_data["vehicle_types"] += [
        {"id": "truck", "emergency_decel": 1.8},
        {"id": "emergency", "speed_factor": {"mean": 1.3}},
    ]
    report = validate_data(demo_data)
    assert report.issues == ()
    assert report.resolved is not None
    types = {vt.id: vt for vt in report.resolved.vehicle_types}
    assert (types["truck"].decel, types["truck"].emergency_decel) == (1.5, 1.8)
    assert types["emergency"].speed_factor.max == 1.3


def test_long_derived_ids_are_reported_not_raised(demo_data: Data) -> None:
    chain(*LONG_IDS)(demo_data)
    demo_data["demand"]["flows"].append(
        {"id": "long", "route": [f"{LONG}_in", f"{LONG}_out"], "rate": 60}
    )
    report = validate_data(demo_data)
    assert [(i.code, i.path) for i in report.issues] == [("E006", "network.roads[8].id")]
    assert f'"{LONG}_in->{LONG}_out"' in report.issues[0].message


def test_validate_spec_checks_the_version(demo_data: Data) -> None:
    spec = Scenario.from_dict(demo_data).spec
    for version, code in (("9.9", "E012"), ("1.9", "E011")):
        report = validate_spec(spec.model_copy(update={"version": version}))
        assert [(i.code, i.path) for i in report.issues] == [(code, "version")]
        with pytest.raises(ScenarioValidationError):
            Scenario.from_spec(spec.model_copy(update={"version": version}))


def test_formatter_output_matches_the_plan(demo_data: Data) -> None:
    green = demo_data["network"]["intersections"][0]["signal"]["phases"][1]["green"]
    green["E_in->W_ot"] = green.pop("E_in->W_out")
    demo_data["demand"]["flows"][3]["route"] = ["E_in", "E_out"]
    demo_data["demand"]["trips"][0]["vehicle_type"] = "cr"
    demo_data["demand"]["transit"][0]["stops"][1]["position"] = 250
    report = validate_data(demo_data)
    with pytest.raises(ScenarioValidationError) as info:
        report.raise_for_errors()
    assert str(info.value) == (
        "Scenario validation failed:\n"
        "  - network.intersections[0].signal.phases[1]: phase contains unknown movement "
        '"E_in->W_ot" (did you mean "E_in->W_out"?)\n'
        '  - demand.flows[3].route[1]: route is not connected: no movement from "E_in" to '
        '"E_out" at intersection "J"\n'
        '  - demand.trips[0].vehicle_type: unknown vehicle type "cr" (did you mean "car"?) '
        "(available: bus, car, city_bus, emergency, truck)\n"
        "  - demand.transit[0].stops[1].position: stop position 250.0 m must be in [12.0, "
        "190.6] (bus length .. lane length \u2212 1 m)"
    )
    assert [w.code for w in info.value.warnings] == ["W301"]
    payload = info.value.to_dict()
    assert len(payload["errors"]) == 4 and payload["warnings"][0]["code"] == "W301"


def test_strict_promotes_warnings(demo_data: Data) -> None:
    demo_data["simulation"]["dt"] = 0.1
    report = validate_data(demo_data)
    assert report.ok and [w.code for w in report.warnings] == ["W701"]
    report.raise_for_errors()  # warnings alone do not fail
    with pytest.raises(ScenarioValidationError) as info:
        report.raise_for_errors(strict=True)
    assert [i.code for i in info.value.errors] == ["W701"]
    assert "simulation.dt: dt=0.1 s is outside" in str(info.value)


@pytest.mark.parametrize("data", [None, [], "text", 3, {"format": "urbanflow.scenario"}])
def test_validate_data_never_raises(data: Any) -> None:
    report = validate_data(data)
    assert not report.ok and report.spec is None


def test_output_order_is_stage_then_document(demo_data: Data) -> None:
    demo_data["simulation"]["dt"] = 0.1  # Q stage, simulation check (last)
    demo_data["vehicle_types"].append({"id": "car"})  # P stage
    demo_data["demand"]["trips"][0]["vehicle_type"] = "cr"  # Q stage, demand check
    codes = [i.code for i in validate_data(demo_data).issues]
    assert codes == ["E401", "E402", "W701"]


def test_validate_spec_returns_the_resolved_spec(demo_data: Data) -> None:
    spec = Scenario.from_dict(demo_data).spec
    report = validate_spec(spec)
    assert isinstance(report, ValidationReport)
    assert report.spec is spec and report.resolved is not None
    assert report.resolved.network.intersections[0].movements is not None


def _load_docs_script() -> Any:
    path = REPO / "scripts" / "gen_scenario_docs.py"
    spec = importlib.util.spec_from_file_location("gen_scenario_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenario_format_docs_are_in_sync() -> None:
    """docs/scenario-format.md's generated blocks (codes, schema, example) are current."""
    docs = _load_docs_script()
    page = docs.PAGE.read_text(encoding="utf-8")
    assert docs.render_page(page) == page, "run: uv run python scripts/gen_scenario_docs.py"
    for code in ISSUE_CODES:
        assert f"| {code} |" in page
