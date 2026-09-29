"""Scenario validation: the staged pipeline and the issue-code registry (plan E.8).

Stages: L load (``io.py``) -> S structural (pydantic) -> P pre-derive -> D derive -> Q
post-derive. X (deep: registries and compile diagnostics) runs in ``urbanflow.check``.
Only S is a hard gate; within P and Q every issue is collected, and the shared ``broken``
id sets keep one bad reference from producing a cascade of follow-on errors. Output order
is stage order, then check order, then document order.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final, Literal, NamedTuple, get_args

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from urbanflow.core import constants as C
from urbanflow.core.errors import (
    ScenarioValidationError,
    Severity,
    ValidationIssue,
    format_path,
    suggest,
)
from urbanflow.core.types import IntersectionKind, TurnKind
from urbanflow.scenario.derive import (
    RoadGraph,
    Topology,
    merge_vehicle_type,
    polyline_length,
    resolve,
    road_width,
    turn_table,
    wrap_angle,
)
from urbanflow.scenario.derive import topology as build_topology
from urbanflow.scenario.io import migrate, version_issue
from urbanflow.scenario.schema import (
    DEFAULT_VEHICLE_TYPES,
    EXCLUSIVE_MESSAGES,
    ID_PATTERN,
    ConnectionSpec,
    FlowSpec,
    RoadSpec,
    ScenarioSpec,
    TransitLineSpec,
    TripSpec,
    VehicleTypeSpec,
)

__all__ = [
    "ISSUE_CODES",
    "POST_CHECKS",
    "PRE_CHECKS",
    "Broken",
    "Check",
    "CheckContext",
    "IssueStage",
    "IssueTemplate",
    "ValidationReport",
    "issue",
    "issues_from_pydantic",
    "validate_data",
    "validate_spec",
]

IssueStage = Literal["L", "S", "P", "Q", "X", "B"]
Loc = tuple[str | int, ...]


class IssueTemplate(NamedTuple):
    """Severity, message template and stage of one issue code.

    Codes with several messages list the others in ``alternatives`` (``issue(variant=k)``).
    Stage ``B`` marks builder/editor-op errors (E.9); ``X`` codes are emitted by the deep
    check (``urbanflow.check``) and the network compiler.
    """

    severity: Severity
    template: str
    stage: IssueStage
    alternatives: tuple[str, ...] = ()


_E, _W = Severity.error, Severity.warning
_EXCLUSIVE = tuple(EXCLUSIVE_MESSAGES.values())
_LANE_ID = re.compile(r"(.+)_(0|[1-9][0-9]*)")
_LANE_MISSING = 'lane {lane} does not exist: road "{road}" has {n} lanes (0-{max})'

ISSUE_CODES: Final[Mapping[str, IssueTemplate]] = MappingProxyType(
    {
        # ---- L: load
        "E000": IssueTemplate(
            _E,
            "invalid JSON: {detail} (line {line}, column {col})",
            "L",
            (
                "file is not valid UTF-8: {detail}",
                "NaN/Infinity are not allowed (line {line})",
                "invalid JSON: {detail}",
            ),
        ),
        "E010": IssueTemplate(
            _E, 'not an UrbanFlow scenario: "format" must be "urbanflow.scenario" (got {got})', "L"
        ),
        "E011": IssueTemplate(
            _E, "scenario version {v} is newer than supported {cur}; upgrade urbanflow", "L"
        ),
        "E012": IssueTemplate(
            _E, "unsupported scenario version {v}; no migration path to {cur}", "L"
        ),
        "E013": IssueTemplate(_E, "file is {size:.1f} MB; the limit is {limit} MB", "L"),
        # ---- S: structural (pydantic)
        "E001": IssueTemplate(_E, 'missing required field "{field}"', "S"),
        "E002": IssueTemplate(_E, 'unknown field "{field}"{hint}', "S"),
        "E003": IssueTemplate(_E, "expected {expected}, got {got}", "S"),
        "E004": IssueTemplate(
            _E,
            "must be {op} {bound} (got {input})",
            "S",
            (
                "must have at least {n} items",
                "must have at most {n} items",
                "must be at least {n} characters long",
                "must be at most {n} characters long",
            ),
        ),
        "E005": IssueTemplate(_E, "invalid value {input}; expected one of: {choices}", "S"),
        "E006": IssueTemplate(
            _E, "invalid id {input}: use 1-128 characters from A-Z a-z 0-9 _ . : - >", "S"
        ),
        "E007": IssueTemplate(_E, _EXCLUSIVE[0], "S", _EXCLUSIVE[1:]),
        "E009": IssueTemplate(_E, "{message}", "S"),
        # ---- B: builder (E.9)
        "E021": IssueTemplate(_E, 'unknown {kind} "{id}"{hint}', "B"),
        "E022": IssueTemplate(
            _E,
            '{kind} "{id}" is still referenced by {refs}; remove those first or use cascade=True',
            "B",
        ),
        "E024": IssueTemplate(
            _E, "the id of an existing {kind} cannot be changed; remove it and add a new one", "B"
        ),
        # ---- P: pre-derive (network)
        "E101": IssueTemplate(
            _E,
            'duplicate intersection id "{id}" (first defined at network.intersections[{j}])',
            "P",
        ),
        "E102": IssueTemplate(
            _E, 'duplicate road id "{id}" (first defined at network.roads[{j}])', "P"
        ),
        "E103": IssueTemplate(_E, 'unknown intersection "{id}"{hint}', "P"),
        "E104": IssueTemplate(_E, 'road starts and ends at intersection "{id}"', "P"),
        "E105": IssueTemplate(
            _E, 'road id "{id}" collides with lane id "{lane}" derived from road "{other}"', "P"
        ),
        "E106": IssueTemplate(
            _E, 'only signalized intersections can have a signal (kind is "{kind}")', "P"
        ),
        "E107": IssueTemplate(_E, "boundary intersections cannot have movements", "P"),
        "E108": IssueTemplate(_E, 'major road "{road}" does not end at intersection "{id}"', "P"),
        "E401": IssueTemplate(_E, 'duplicate vehicle type id "{id}"', "P"),
        "E801": IssueTemplate(
            _E, "point repeats the previous point; consecutive points must differ", "P"
        ),
        "E802": IssueTemplate(
            _E,
            'road is {L:.1f} m long but intersections "{a}" and "{b}" reserve {ra:.1f} m + '
            "{rb:.1f} m; lanes would be {rem:.1f} m (minimum {min} m)",
            "P",
        ),
        "W101": IssueTemplate(_W, 'intersection "{id}" is not connected to any road', "P"),
        "W102": IssueTemplate(_W, 'major_roads is ignored for kind "{kind}"', "P"),
        "W103": IssueTemplate(
            _W,
            'intersections "{a}" and "{b}" overlap ({d:.1f} m apart, radii {ra:.1f} + {rb:.1f} m)',
            "P",
        ),
        "W801": IssueTemplate(
            _W,
            'road ends {d:.1f} m from intersection "{j}", outside its radius {r:.1f} m; '
            "connectors bridge the gap",
            "P",
        ),
        "W802": IssueTemplate(
            _W, "sharp bend ({deg:.0f} deg); offset lanes may self-intersect", "P"
        ),
        # ---- Q: movements
        "E201": IssueTemplate(_E, 'road "{road}" does not end at intersection "{id}"', "Q"),
        "E202": IssueTemplate(_E, 'road "{road}" does not start at intersection "{id}"', "Q"),
        "E203": IssueTemplate(_E, 'unknown road "{road}"{hint}', "Q"),
        "E204": IssueTemplate(_E, 'duplicate movement id "{id}" (first defined at {where})', "Q"),
        "E205": IssueTemplate(_E, 'duplicate movement "{a}" -> "{b}" (also movements[{j}])', "Q"),
        "E206": IssueTemplate(_E, _LANE_MISSING, "Q"),
        "E207": IssueTemplate(_E, _LANE_MISSING, "Q"),
        "E208": IssueTemplate(_E, "duplicate connection {a}->{b}", "Q"),
        "W201": IssueTemplate(
            _W,
            'road "{road}" enters "{id}" but has no movements out of it; trips can only end '
            'there (use kind "boundary" for exits)',
            "Q",
        ),
        "W202": IssueTemplate(
            _W,
            'lane "{lane}" has no outgoing connection; vehicles must change lanes before the '
            "stop line",
            "Q",
        ),
        # ---- Q: signals
        "E301": IssueTemplate(_E, "signalized intersection has no movements to control", "Q"),
        "E302": IssueTemplate(_E, 'phase contains unknown movement "{m}"{hint}', "Q"),
        "E303": IssueTemplate(
            _E, 'phase contains movement "{m}" of another intersection ("{other}")', "Q"
        ),
        "E304": IssueTemplate(_E, 'duplicate phase id "{id}"', "Q"),
        "E305": IssueTemplate(_E, "initial_phase {k} out of range ({n} phases)", "Q"),
        "E306": IssueTemplate(
            _E,
            "phase min_green ({a} s) exceeds max_green ({b} s)",
            "Q",
            ("duration {d} s is below min_green {g} s",),
        ),
        "W301": IssueTemplate(
            _W, 'movement "{m}" is never green in any phase; its traffic can never enter', "Q"
        ),
        "W303": IssueTemplate(
            _W,
            "yellow is 0 s; vehicles approaching at speed cannot stop safely at phase changes",
            "Q",
        ),
        "W305": IssueTemplate(
            _W,
            "phase durations, yellow or all_red are not multiples of dt={dt} s; the realised "
            "cycle is {cq} s instead of {c} s",
            "Q",
        ),
        # ---- Q: demand
        "E402": IssueTemplate(_E, 'unknown vehicle type "{t}"{hint} (available: {list})', "Q"),
        "E501": IssueTemplate(
            _E,
            'duplicate demand id "{id}" (also demand.{kind2}[{j}])',
            "Q",
            ('id "{id}" collides with vehicle ids generated by {kind} "{other}"',),
        ),
        "E502": IssueTemplate(_E, 'unknown road "{road}"{hint}', "Q"),
        "E503": IssueTemplate(
            _E,
            'route is not connected: no movement from "{a}" to "{b}" at intersection "{j}"',
            "Q",
            ('route is not connected: "{a}" ends at "{ja}" but "{b}" starts at "{jb}"',),
        ),
        "E504": IssueTemplate(_E, '"{d}" is not reachable from "{o}"{via_text}', "Q"),
        "E505": IssueTemplate(_E, 'lane {k} does not exist on road "{road}" ({n} lanes)', "Q"),
        "E506": IssueTemplate(
            _E,
            "binomial arrivals allow at most one vehicle per step: {rate} veh/h > {max:.0f} "
            'veh/h at dt={dt} s (use "poisson" or split the flow)',
            "Q",
        ),
        "E507": IssueTemplate(
            _E,
            'flow has neither "end" nor "count" but simulation.duration is null; the run would '
            "never end",
            "Q",
        ),
        "W501": IssueTemplate(
            _W,
            '{rate} veh/h exceeds the entry capacity of road "{road}" (~{cap:.0f} veh/h); '
            "vehicles will queue outside the network",
            "Q",
        ),
        "W502": IssueTemplate(_W, "starts at {t} s, after the simulation ends ({d} s)", "Q"),
        # ---- Q: transit
        "E601": IssueTemplate(_E, 'stop road "{road}" is not on the line\'s route', "Q"),
        "E602": IssueTemplate(
            _E,
            "stops must follow route order: stop {k} is before stop {k1} along the route",
            "Q",
        ),
        "E604": IssueTemplate(_E, 'lane {k} does not exist on road "{road}" ({n} lanes)', "Q"),
        "E606": IssueTemplate(_E, 'duplicate stop id "{id}"', "Q"),
        "E807": IssueTemplate(
            _E,
            "stop position {p} m must be in [{len}, {Lm1:.1f}] "
            "(bus length .. lane length \u2212 1 m)",  # U+2212 minus, as in the plan
            "Q",
        ),
        "W306": IssueTemplate(
            _W,
            "stop lane cannot reach the next route road and the stop is {d:.0f} m (< 50 m) from "
            "the stop line",
            "Q",
        ),
        # ---- Q: simulation
        "W701": IssueTemplate(_W, "dt={dt} s is outside the recommended 0.2-1.0 s", "Q"),
        "W702": IssueTemplate(
            _W,
            "duration {d} s is not a multiple of dt {dt} s; it is rounded down to {r} s",
            "Q",
        ),
        # ---- X: deep check and compiler (emitted by urbanflow.check / network.compiler)
        "E806": IssueTemplate(
            _E,
            'lane "{id}" is {L:.1f} m long but vehicle type "{t}" routed through it needs '
            "{need:.1f} m (length + min_gap); don't-block-the-box admission (F.3) could never "
            "admit it and insertion would place it past the lane end",
            "X",
        ),
        "E901": IssueTemplate(_E, 'unknown signal controller "{t}"{hint} (available: {list})', "X"),
        "E902": IssueTemplate(_E, "{message}", "X"),
        "E903": IssueTemplate(
            _E, 'unknown parameter "{p}" for car-following model "{m}" (known: {list})', "X"
        ),
        "E904": IssueTemplate(_E, 'unknown {kind} "{name}"{hint} (available: {list})', "X"),
        "E905": IssueTemplate(_E, "lane geometry is degenerate: {detail}", "X"),
        "W302": IssueTemplate(
            _W,
            'movements "{a}" and "{b}" cross but are both protected (G); make one permissive (g)',
            "X",
        ),
        "W304": IssueTemplate(
            _W, 'connectors "{a}" and "{b}" cross twice; their conflict zones were merged', "X"
        ),
    }
)


def issue(code: str, path: str | Loc, *, variant: int = 0, **fmt: Any) -> ValidationIssue:
    """A :class:`ValidationIssue` for ``code`` with its registered message.

    ``path`` is a JSON path or a location tuple (formatted with ``format_path``).
    """
    tpl = ISSUE_CODES[code]
    text = (tpl.template, *tpl.alternatives)[variant].format(**fmt)
    where = path if isinstance(path, str) else format_path(path)
    return ValidationIssue(where, text, code, tpl.severity)


def _key_path(loc: Loc, key: str) -> str:
    """Path of a mapping key, always bracketed: ``demand.flows[0].type_mix["city_bus"]``."""
    return f"{format_path(loc)}[{json.dumps(key, ensure_ascii=False)}]"


# --------------------------------------------------------------------------- pydantic -> issues
_TYPE_EXPECTED: Final = {
    "int_type": "an integer",
    "int_parsing": "an integer",
    "int_from_float": "an integer",
    "float_type": "a number",
    "float_parsing": "a number",
    "finite_number": "a finite number",
    "string_type": "a string",
    "bool_type": "a boolean",
    "bool_parsing": "a boolean",
    "list_type": "an array",
    "tuple_type": "an array",
    "dict_type": "an object",
    "model_type": "an object",
    "model_attributes_type": "an object",
    "none_required": "null",
}
_BOUNDS: Final = {
    "greater_than": (">", "gt"),
    "greater_than_equal": (">=", "ge"),
    "less_than": ("<", "lt"),
    "less_than_equal": ("<=", "le"),
}
_LENGTHS: Final = {
    "too_short": (1, "min_length"),
    "too_long": (2, "max_length"),
    "string_too_short": (3, "min_length"),
    "string_too_long": (4, "max_length"),
}
_TYPE_SEGMENTS: Final = frozenset({"int", "float", "str", "bool", "list", "dict", "tuple", "none"})


def _is_branch(segment: str | int) -> bool:
    return isinstance(segment, str) and (
        "[" in segment or "(" in segment or segment.lower() in _TYPE_SEGMENTS
    )


def _show(value: Any) -> str:
    """A short JSON-like rendering of an input value for messages."""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (str, int, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "an array"
    return "an object" if isinstance(value, Mapping) else type(value).__name__


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float, str)):
        name = {int: "integer", float: "number", str: "string"}[type(value)]
        return f"{name} {_show(value)}"
    return _show(value)


def _model_in(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _model_in(arg)
        if found is not None:
            return found
    return None


def _fields_at(root: type[BaseModel] | None, loc: Sequence[str | int]) -> list[str]:
    """Field names (aliases) of the model at ``loc`` below ``root``, for did-you-mean hints."""
    model = root
    for seg in loc:
        if model is None:
            return []
        if isinstance(seg, int) or _is_branch(seg):
            continue
        info = next((f for n, f in model.model_fields.items() if seg in (n, f.alias)), None)
        model = _model_in(info.annotation) if info is not None else None
    if model is None:
        return []
    return [f.alias or n for n, f in model.model_fields.items()]


def _branch_rank(err: ErrorDetails) -> int:
    """Among union-branch errors for one path, prefer the branch matching the input type."""
    kind, value = err["type"], err.get("input")
    if kind == "literal_error":
        return 0 if isinstance(value, str) else 2
    return 3 if kind in _TYPE_EXPECTED else 1


def _convert(
    err: ErrorDetails, path: str, root: type[BaseModel] | None, names: Collection[str]
) -> ValidationIssue:
    kind, ctx, value = err["type"], err.get("ctx", {}), err.get("input")
    loc = err["loc"]
    if kind in {"missing", "missing_argument", "missing_keyword_only_argument"}:
        return issue("E001", path, field=loc[-1] if loc else "?")
    if kind in {"extra_forbidden", "unexpected_keyword_argument"}:
        name = str(loc[-1])
        candidates = _fields_at(root, loc[:-1]) if root is not None else list(names)
        return issue("E002", path, field=name, hint=suggest(name, candidates))
    if kind == "uf_exclusive":
        return ValidationIssue(path, err["msg"], "E007")
    if kind in _BOUNDS:
        op, key = _BOUNDS[kind]
        return issue("E004", path, op=op, bound=_show(ctx[key]), input=_show(value))
    if kind in _LENGTHS:
        variant, key = _LENGTHS[kind]
        return issue("E004", path, variant=variant, n=ctx[key])
    if kind in {"literal_error", "enum"}:
        choices = str(ctx.get("expected", "")).replace(" or ", ", ").replace("'", '"')
        return issue("E005", path, input=_show(value), choices=choices)
    if kind == "string_pattern_mismatch" and ctx.get("pattern") == ID_PATTERN:
        return issue("E006", path, input=_show(value))
    if kind in _TYPE_EXPECTED:
        return issue("E003", path, expected=_TYPE_EXPECTED[kind], got=_json_type(value))
    return issue("E009", path, message=err["msg"])


def _item_failure_echo(err: ErrorDetails) -> bool:
    """pydantic repeats failed items as a length error ("at least 1 item ..., not 0")."""
    value, ctx = err.get("input"), err.get("ctx", {})
    if err["type"] != "too_short" or not isinstance(value, (list, tuple, Mapping)):
        return False
    return int(ctx.get("actual_length", len(value))) < len(value)


def issues_from_pydantic(
    exc: ValidationError,
    *,
    prefix: Loc = (),
    root: type[BaseModel] | None = ScenarioSpec,
    names: Collection[str] = (),
) -> list[ValidationIssue]:
    """Convert a pydantic error into E001-E009 issues with JSON paths (plan E.8 §2.2).

    ``root`` is the model the error's ``loc`` starts from (for unknown-field hints), or None
    with ``names`` as the candidate names (``validate_call`` arguments). Errors of rejected
    union branches collapse to the most relevant one per path.
    """
    best: dict[str, tuple[int, ErrorDetails]] = {}
    for err in exc.errors():
        if _item_failure_echo(err):
            continue
        path = format_path((*prefix, *err["loc"]))
        rank = _branch_rank(err) if any(_is_branch(s) for s in err["loc"]) else -1
        if path not in best or rank < best[path][0]:
            best[path] = (rank, err)
    return [_convert(err, path, root, names) for path, (_, err) in best.items()]


# --------------------------------------------------------------------------- report
@dataclass(frozen=True, slots=True)
class ValidationReport:
    """All issues of one validation run, plus the spec and its resolved form when available."""

    issues: tuple[ValidationIssue, ...]
    spec: ScenarioSpec | None = None
    resolved: ScenarioSpec | None = None

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.error)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.warning)

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings allowed)."""
        return not self.errors

    def raise_for_errors(self, strict: bool = False, source: str | None = None) -> None:
        """Raise :class:`ScenarioValidationError` on errors (or on warnings when ``strict``)."""
        if strict and self.warnings:
            promoted = tuple(
                replace(i, severity=Severity.error) if i.severity is Severity.warning else i
                for i in self.issues
            )
            raise ScenarioValidationError(promoted, source)
        if self.errors:
            raise ScenarioValidationError(self.issues, source)


# --------------------------------------------------------------------------- pipeline
@dataclass(slots=True)
class Broken:
    """Ids whose definition is invalid; checks skip anything that references them."""

    intersections: set[str] = field(default_factory=set)
    roads: set[str] = field(default_factory=set)
    movements: set[str] = field(default_factory=set)
    """Movements already reported (bad road, duplicate); signal checks don't flag them again."""


@dataclass(slots=True)
class CheckContext:
    """Shared state of one validation run."""

    spec: ScenarioSpec
    index: Topology
    broken: Broken = field(default_factory=Broken)
    lane_lengths: dict[str, float] = field(default_factory=dict)
    """Trimmed lane length (m) of every placeable road."""
    resolved: ScenarioSpec | None = None

    def skipped(self) -> frozenset[str]:
        """Intersections left underived: broken ones and those touching broken roads."""
        out = set(self.broken.intersections)
        for road in self.spec.network.roads:
            if road.id in self.broken.roads:
                out.update((road.from_, road.to))
        return frozenset(out)

    def final(self) -> ScenarioSpec:
        if self.resolved is None:  # pragma: no cover - post checks only run after derive
            raise RuntimeError("post-derive check ran before derive")
        return self.resolved

    def is_first(self, kind: Literal["intersection", "road"], obj: Any) -> bool:
        """True for the first definition of an id (duplicates are reported once)."""
        table: Mapping[str, Any] = (
            self.index.intersections if kind == "intersection" else self.index.roads
        )
        return table.get(obj.id) is obj


Check = Callable[[CheckContext], Iterable[ValidationIssue]]


def _ints(ctx: CheckContext) -> Iterator[tuple[int, Any]]:
    return enumerate(ctx.spec.network.intersections)


# ---- P checks ------------------------------------------------------------------------------
def check_intersection_ids(ctx: CheckContext) -> Iterator[ValidationIssue]:
    first: dict[str, int] = {}
    for i, ix in _ints(ctx):
        if ix.id in first:
            ctx.broken.intersections.add(ix.id)
            yield issue("E101", ("network", "intersections", i, "id"), id=ix.id, j=first[ix.id])
        else:
            first[ix.id] = i


def check_road_ids(ctx: CheckContext) -> Iterator[ValidationIssue]:
    roads = ctx.spec.network.roads
    first: dict[str, int] = {}
    for i, road in enumerate(roads):
        if road.id in first:
            ctx.broken.roads.add(road.id)
            yield issue("E102", ("network", "roads", i, "id"), id=road.id, j=first[road.id])
        else:
            first[road.id] = i
    lane_counts = {r.id: len(r.lanes) for r in reversed(roads)}  # first definition wins
    for i, road in enumerate(roads):
        match = _LANE_ID.fullmatch(road.id)  # lane ids are "{road}_{i}"
        if match and lane_counts.get(match[1], 0) > int(match[2]):
            yield issue(
                "E105", ("network", "roads", i, "id"), id=road.id, lane=road.id, other=match[1]
            )


def check_road_endpoints(ctx: CheckContext) -> Iterator[ValidationIssue]:
    ints = ctx.index.intersections
    for i, road in enumerate(ctx.spec.network.roads):
        bad = False
        for key, ref in (("from", road.from_), ("to", road.to)):
            if ref not in ints:
                bad = True
                yield issue("E103", ("network", "roads", i, key), id=ref, hint=suggest(ref, ints))
        if not bad and road.from_ == road.to:
            bad = True
            yield issue("E104", ("network", "roads", i), id=road.from_)
        if bad:
            ctx.broken.roads.add(road.id)


def _trim(topo: Topology, j: str, end: tuple[float, float]) -> float:
    if topo.kinds[j] is IntersectionKind.boundary:
        return 0.0
    return max(0.0, topo.radii[j] - math.dist(end, topo.intersections[j].point))


def _bends(
    ctx: CheckContext, i: int, road: RoadSpec, pts: Sequence[Any]
) -> Iterator[ValidationIssue]:
    width = road_width(ctx.spec.network, road)
    for k in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[k - 1], pts[k], pts[k + 1]
        phi = abs(wrap_angle(math.atan2(y2 - y1, x2 - x1) - math.atan2(y1 - y0, x1 - x0)))
        shortest = min(math.dist(pts[k - 1], pts[k]), math.dist(pts[k], pts[k + 1]))
        if phi >= math.pi - C.GEOM_EPS or width * math.tan(phi / 2) > shortest:
            yield issue("W802", ("network", "roads", i, "points", k), deg=math.degrees(phi))


def check_road_geometry(ctx: CheckContext) -> Iterator[ValidationIssue]:
    topo = ctx.index
    for i, road in enumerate(ctx.spec.network.roads):
        if not ctx.is_first("road", road) or road.id in ctx.broken.roads:
            continue
        pts = topo.points[road.id]
        if road.points is not None:
            repeats = [k for k in range(1, len(pts)) if pts[k] == pts[k - 1]]
            if repeats:
                ctx.broken.roads.add(road.id)
                for k in repeats:
                    yield issue("E801", ("network", "roads", i, "points", k))
                continue
            yield from _bends(ctx, i, road, pts)
            for j, end in ((road.from_, pts[0]), (road.to, pts[-1])):
                if topo.kinds[j] is IntersectionKind.boundary:
                    continue
                d = math.dist(end, topo.intersections[j].point)
                if d > topo.radii[j] + C.POSITION_EPS:
                    yield issue(
                        "W801", ("network", "roads", i, "points"), d=d, j=j, r=topo.radii[j]
                    )
        length = polyline_length(pts)
        ra, rb = _trim(topo, road.from_, pts[0]), _trim(topo, road.to, pts[-1])
        remaining = length - ra - rb
        ctx.lane_lengths[road.id] = remaining
        if remaining < C.MIN_LANE_LENGTH:
            ctx.broken.roads.add(road.id)
            yield issue(
                "E802",
                ("network", "roads", i),
                L=length,
                a=road.from_,
                b=road.to,
                ra=ra,
                rb=rb,
                rem=remaining,
                min=C.MIN_LANE_LENGTH,
            )


def check_intersections(ctx: CheckContext) -> Iterator[ValidationIssue]:
    topo = ctx.index
    for i, ix in _ints(ctx):
        if not ctx.is_first("intersection", ix):
            continue
        base: Loc = ("network", "intersections", i)
        kind = topo.kinds[ix.id]
        ins = topo.incoming.get(ix.id, ())
        if not ins and not topo.outgoing.get(ix.id):
            yield issue("W101", base, id=ix.id)
        if ix.signal is not None and kind is not IntersectionKind.signalized:
            yield issue("E106", (*base, "signal"), kind=kind)
        if ix.movements is not None and kind is IntersectionKind.boundary:
            yield issue("E107", (*base, "movements"))
        if ix.major_roads is not None:
            if kind is not IntersectionKind.priority:
                yield issue("W102", (*base, "major_roads"), kind=kind)
            else:
                for k, road in enumerate(ix.major_roads):
                    if road not in ins:
                        yield issue("E108", (*base, "major_roads", k), road=road, id=ix.id)
    yield from _overlaps(ctx)


def _overlaps(ctx: CheckContext) -> Iterator[ValidationIssue]:
    """W103 by sort-and-sweep over x (non-boundary intersections)."""
    topo = ctx.index
    position = {ix.id: i for i, ix in reversed(list(_ints(ctx)))}
    nodes = sorted(
        (ix.point[0], ix.id)
        for ix in topo.intersections.values()
        if topo.kinds[ix.id] is not IntersectionKind.boundary and topo.radii[ix.id] > 0
    )
    if not nodes:
        return
    reach = max(topo.radii[j] for _, j in nodes)
    found = []
    for n, (xa, a) in enumerate(nodes):
        for xb, b in nodes[n + 1 :]:
            if xb - xa >= topo.radii[a] + reach:
                break
            ra, rb = topo.radii[a], topo.radii[b]
            d = math.dist(topo.intersections[a].point, topo.intersections[b].point)
            if d < ra + rb - C.POSITION_EPS:
                first, second = sorted((a, b), key=position.__getitem__)
                found.append((position[second], position[first], first, second, d))
    for i2, _, first, second, d in sorted(found):
        ra, rb = topo.radii[first], topo.radii[second]
        yield issue("W103", ("network", "intersections", i2), a=first, b=second, d=d, ra=ra, rb=rb)


def check_derived_ids(ctx: CheckContext) -> Iterator[ValidationIssue]:
    """E006 when a derived id would exceed the id limit (the resolved file could not load).

    Derived movements (``"{from_road}->{to_road}"``) are reported at the incoming road's id,
    explicit movements without an id at the movement (the intersection is then left
    underived), and stops without an id (``"{line}.{k}"``) at the stop.
    """
    topo, spec = ctx.index, ctx.spec
    road_at = {road.id: i for i, road in reversed(list(enumerate(spec.network.roads)))}

    def too_long(a: str, b: str) -> bool:
        return len(a) + len(b) + 2 > C.MAX_ID_LENGTH

    for i, ix in _ints(ctx):
        j = ix.id
        if not ctx.is_first("intersection", ix) or topo.kinds[j] is IntersectionKind.boundary:
            continue
        ins, outs = topo.incoming.get(j, ()), topo.outgoing.get(j, ())
        found: list[ValidationIssue] = []
        if ix.movements is not None:
            found += [
                issue("E006", ("network", "intersections", i, "movements", k), input=_show(mid))
                for k, m in enumerate(ix.movements)
                if m.id is None and too_long(m.from_road, m.to_road)
                for mid in [f"{m.from_road}->{m.to_road}"]
            ]
        elif any(too_long(a, b) for a in ins for b in outs):  # cheap before the turn table
            found += [
                issue("E006", ("network", "roads", road_at[a], "id"), input=_show(f"{a}->{b}"))
                for a, b, _, turn in turn_table(topo, j)
                if too_long(a, b) and (turn is not TurnKind.uturn or spec.network.allow_uturns)
            ]
        if found:
            ctx.broken.intersections.add(j)
            yield from found
    for i, line in enumerate(spec.demand.transit):
        for k, stop in enumerate(line.stops):
            sid = f"{line.id}.{k}"
            if stop.id is None and len(sid) > C.MAX_ID_LENGTH:
                yield issue("E006", ("demand", "transit", i, "stops", k), input=_show(sid))


def check_vehicle_types(ctx: CheckContext) -> Iterator[ValidationIssue]:
    """E401 duplicates; E007 for built-in overrides whose merged values break a rule."""
    seen: set[str] = set()
    for i, vt in enumerate(ctx.spec.vehicle_types):
        if vt.id in seen:
            yield issue("E401", ("vehicle_types", i, "id"), id=vt.id)
            continue
        seen.add(vt.id)
        base = DEFAULT_VEHICLE_TYPES.get(vt.id)
        if base is None:
            continue
        try:
            merge_vehicle_type(base, vt)
        except ValidationError as exc:
            yield from issues_from_pydantic(exc, prefix=("vehicle_types", i), root=VehicleTypeSpec)


# ---- Q checks ------------------------------------------------------------------------------
def _lane_count(ctx: CheckContext, road: str) -> int:
    return len(ctx.index.roads[road].lanes)


def check_movements(ctx: CheckContext) -> Iterator[ValidationIssue]:
    topo, skip, res = ctx.index, ctx.skipped(), ctx.final()
    first_path: dict[str, str] = {}
    for i, ix in enumerate(res.network.intersections):
        if ix.id in skip or ix.movements is None or ix.kind is IntersectionKind.boundary:
            continue
        ins, outs = topo.incoming.get(ix.id, ()), topo.outgoing.get(ix.id, ())
        pairs: dict[tuple[str, str], int] = {}
        for k, mov in enumerate(ix.movements):
            base: Loc = ("network", "intersections", i, "movements", k)
            bad = False
            for key, road, allowed, code in (
                ("from_road", mov.from_road, ins, "E201"),
                ("to_road", mov.to_road, outs, "E202"),
            ):
                if road in ctx.broken.roads:
                    bad = True
                elif road not in topo.roads:
                    bad = True
                    yield issue("E203", (*base, key), road=road, hint=suggest(road, topo.roads))
                elif road not in allowed:
                    bad = True
                    yield issue(code, (*base, key), road=road, id=ix.id)
            mid = str(mov.id)
            if mid in first_path:
                yield issue("E204", (*base, "id"), id=mid, where=first_path[mid])
            else:
                first_path[mid] = format_path(base)
            pair = (mov.from_road, mov.to_road)
            if pair in pairs:
                bad = True
                yield issue("E205", base, a=pair[0], b=pair[1], j=pairs[pair])
            else:
                pairs[pair] = k
            if bad:
                ctx.broken.movements.add(mid)
            else:
                yield from _connections(ctx, base, mov.from_road, mov.to_road, mov.connections)
        for road in ins:
            if road in ctx.broken.roads:
                continue
            movs = [m for m in ix.movements if m.from_road == road]
            base = ("network", "intersections", i)
            if not movs:
                yield issue("W201", base, road=road, id=ix.id)
                continue
            used = {c.from_lane for m in movs for c in m.connections or ()}
            for lane in range(_lane_count(ctx, road)):
                if lane not in used:
                    yield issue("W202", base, lane=f"{road}_{lane}")


def _connections(
    ctx: CheckContext,
    base: Loc,
    a: str,
    b: str,
    connections: Sequence[ConnectionSpec] | None,
) -> Iterator[ValidationIssue]:
    n_in, n_out = _lane_count(ctx, a), _lane_count(ctx, b)
    seen: set[tuple[int, int]] = set()
    for c, conn in enumerate(connections or ()):
        cb: Loc = (*base, "connections", c)
        if conn.from_lane >= n_in:
            yield issue(
                "E206", (*cb, "from_lane"), lane=conn.from_lane, road=a, n=n_in, max=n_in - 1
            )
        if conn.to_lane >= n_out:
            yield issue("E207", (*cb, "to_lane"), lane=conn.to_lane, road=b, n=n_out, max=n_out - 1)
        key = (conn.from_lane, conn.to_lane)
        if key in seen:
            yield issue("E208", cb, a=key[0], b=key[1])
        seen.add(key)


def _follows_reported_error(
    movement: str, roads: Collection[str], ends: Collection[tuple[str, str]]
) -> bool:
    """True if a phase entry ``"{a}->{b}"`` is a consequence of an error already reported.

    That is: it names a broken road, or shares its incoming or outgoing road with a broken
    movement of the intersection (e.g. a movement whose ``from_road`` has a typo), so no
    cascade of E302 errors follows the original one.
    """
    # prefix/suffix tests, not a split: ids may themselves contain "->"
    starts = [*roads, *(x for x, _ in ends)]
    stops = [*roads, *(y for _, y in ends)]
    return any(movement.startswith(f"{r}->") for r in starts) or any(
        movement.endswith(f"->{r}") for r in stops
    )


def _quote(value: float) -> float:
    """A computed value as quoted in a message: rounded, so float noise never shows."""
    return round(value, C.MESSAGE_DECIMALS)


def _quantised(value: float, dt: float) -> float:
    """Realised length of a signal stage: whole steps, ceil(d/dt - 1e-9) (H.2)."""
    return math.ceil(value / dt - C.TIME_EPS) * dt


def check_signals(ctx: CheckContext) -> Iterator[ValidationIssue]:
    skip, res = ctx.skipped(), ctx.final()
    owner = {
        str(m.id): ix.id
        for ix in res.network.intersections
        if ix.id not in skip
        for m in ix.movements or ()
    }
    dt = res.simulation.dt
    for i, ix in enumerate(res.network.intersections):
        sig = ix.signal
        if ix.id in skip or sig is None or ix.kind is not IntersectionKind.signalized:
            continue
        base: Loc = ("network", "intersections", i, "signal")
        own = [str(m.id) for m in ix.movements or ()]
        if not own:
            yield issue("E301", base)
            continue
        ends = {
            (m.from_road, m.to_road)
            for m in ix.movements or ()
            if str(m.id) in ctx.broken.movements
        }
        phases = sig.phases or ()
        seen: set[str | None] = set()
        green: set[str] = set()
        for p, phase in enumerate(phases):
            pb: Loc = (*base, "phases", p)
            if phase.id in seen:
                yield issue("E304", (*pb, "id"), id=phase.id)
            seen.add(phase.id)
            for mid in phase.green:
                if mid in own:
                    green.add(mid)
                elif mid in owner:
                    yield issue("E303", pb, m=mid, other=owner[mid])
                elif not _follows_reported_error(mid, ctx.broken.roads, ends):
                    yield issue("E302", pb, m=mid, hint=suggest(mid, own))
            low = phase.min_green if phase.min_green is not None else sig.min_green
            high = phase.max_green if phase.max_green is not None else sig.max_green
            if low > high:
                yield issue("E306", pb, a=low, b=high)
            elif phase.duration < low:
                yield issue("E306", pb, variant=1, d=phase.duration, g=low)
        if sig.initial_phase >= len(phases):
            yield issue("E305", (*base, "initial_phase"), k=sig.initial_phase, n=len(phases))
        for mid in own:
            if mid not in green and mid not in ctx.broken.movements:
                yield issue("W301", base, m=mid)
        if sig.yellow == 0:
            yield issue("W303", (*base, "yellow"))
        cycle = sum(p.duration + sig.yellow + sig.all_red for p in phases)
        realised = sum(
            _quantised(p.duration, dt) + _quantised(sig.yellow, dt) + _quantised(sig.all_red, dt)
            for p in phases
        )
        if abs(realised - cycle) > C.TIME_EPS * max(1.0, cycle):
            yield issue("W305", base, dt=dt, cq=_quote(realised), c=_quote(cycle))


@dataclass(slots=True)
class _Demand:
    """Per-run lookups shared by the demand and transit checks."""

    vtypes: dict[str, VehicleTypeSpec]
    pairs: set[tuple[str, str]]
    open_nodes: frozenset[str]
    speeds: dict[str, float]
    graph: RoadGraph | None = None
    paths: dict[tuple[str, str, tuple[str, ...]], bool] = field(default_factory=dict)


def _demand_lookups(ctx: CheckContext) -> _Demand:
    res, skip = ctx.final(), ctx.skipped()
    pairs = {
        (m.from_road, m.to_road)
        for ix in res.network.intersections
        if ix.id not in skip
        for m in ix.movements or ()
    }
    # like skipped ones, intersections with an already reported movement connect every road
    # pair, so a movement typo gives no E503/E504 cascade on the routes through it
    broken_at = {
        ix.id
        for ix in res.network.intersections
        for m in ix.movements or ()
        if str(m.id) in ctx.broken.movements
    }
    speeds: dict[str, float] = {}
    for road in res.network.roads:
        speeds.setdefault(road.id, road.speed_limit or res.network.speed_limit)
    return _Demand({vt.id: vt for vt in res.vehicle_types}, pairs, skip | broken_at, speeds)


def _vehicle_type(d: _Demand, path: str | Loc, t: str) -> Iterator[ValidationIssue]:
    if t not in d.vtypes:
        yield issue("E402", path, t=t, hint=suggest(t, d.vtypes), list=", ".join(sorted(d.vtypes)))


def _known(ctx: CheckContext, path: Loc, road: str, bad: list[bool]) -> Iterator[ValidationIssue]:
    """E502 for an unknown road; ``bad[0]`` is set for unknown or broken roads."""
    if road in ctx.broken.roads:
        bad[0] = True
    elif road not in ctx.index.roads:
        bad[0] = True
        yield issue("E502", path, road=road, hint=suggest(road, ctx.index.roads))


def _route(
    ctx: CheckContext, d: _Demand, base: Loc, roads: Sequence[str]
) -> Iterator[ValidationIssue]:
    bad = [False]
    for k, road in enumerate(roads):
        yield from _known(ctx, (*base, k), road, bad)
    if bad[0]:
        return
    topo = ctx.index
    for k in range(1, len(roads)):
        a, b = roads[k - 1], roads[k]
        ja, jb = topo.roads[a].to, topo.roads[b].from_
        if ja != jb:
            yield issue("E503", (*base, k), variant=1, a=a, ja=ja, b=b, jb=jb)
        elif ja not in d.open_nodes and (a, b) not in d.pairs:
            yield issue("E503", (*base, k), a=a, b=b, j=ja)


def _od(
    ctx: CheckContext, d: _Demand, base: Loc, item: FlowSpec | TripSpec
) -> Iterator[ValidationIssue]:
    origin, dest = str(item.origin), str(item.destination)
    bad = [False]
    yield from _known(ctx, (*base, "origin"), origin, bad)
    for k, road in enumerate(item.via):
        yield from _known(ctx, (*base, "via", k), road, bad)
    yield from _known(ctx, (*base, "destination"), dest, bad)
    if bad[0]:
        return
    key = (origin, dest, tuple(item.via))
    if key not in d.paths:
        if d.graph is None:
            d.graph = RoadGraph(ctx.final(), open_intersections=d.open_nodes)
        d.paths[key] = d.graph.path(origin, dest, item.via) is not None
    if not d.paths[key]:
        via = f" via {', '.join(json.dumps(v) for v in item.via)}" if item.via else ""
        yield issue("E504", (*base, "destination"), d=dest, o=origin, via_text=via)


def _first_roads(item: FlowSpec | TripSpec | TransitLineSpec) -> list[tuple[str, float]]:
    """(first road, weight share) of every way a vehicle of ``item`` can start."""
    if isinstance(item, FlowSpec) and item.routes is not None:
        total = sum(r.weight for r in item.routes)
        shares: dict[str, float] = {}
        for choice in item.routes:
            shares[choice.roads[0]] = shares.get(choice.roads[0], 0.0) + choice.weight / total
        return list(shares.items())
    if item.route is not None:
        return [(item.route[0], 1.0)]
    return [(str(getattr(item, "origin", "")), 1.0)]


def _depart_lane(
    ctx: CheckContext, base: Loc, item: FlowSpec | TripSpec | TransitLineSpec
) -> Iterator[ValidationIssue]:
    lane = item.depart_lane
    if isinstance(lane, str):
        return
    for road, _ in _first_roads(item):
        if road in ctx.index.roads and road not in ctx.broken.roads:
            n = _lane_count(ctx, road)
            if lane >= n:
                yield issue("E505", (*base, "depart_lane"), k=lane, road=road, n=n)
                return


def _flow_rate(flow: FlowSpec) -> tuple[float, str]:
    if flow.rate is not None:
        return flow.rate, "rate"
    return C.SECONDS_PER_HOUR / float(flow.period or 1), "period"


def _capacity(ctx: CheckContext, d: _Demand, flow: FlowSpec, road: str) -> float | None:
    """W501 entry capacity n * 3600 / (T + (l + s0) / v) of ``road`` for the flow's types."""
    mix = dict(flow.type_mix) if flow.type_mix is not None else {str(flow.vehicle_type): 1.0}
    if any(t not in d.vtypes for t in mix):
        return None
    speed = d.speeds[road]
    total = sum(mix.values())
    tau = sum(
        w / total * (d.vtypes[t].headway + (d.vtypes[t].length + d.vtypes[t].min_gap) / speed)
        for t, w in mix.items()
    )
    return _lane_count(ctx, road) * C.SECONDS_PER_HOUR / tau


def _flow_checks(
    ctx: CheckContext, d: _Demand, base: Loc, flow: FlowSpec
) -> Iterator[ValidationIssue]:
    sim = ctx.final().simulation
    rate, rate_key = _flow_rate(flow)
    if flow.arrival == "binomial" and rate * sim.dt > C.SECONDS_PER_HOUR * (1 + C.GEOM_EPS):
        yield issue(
            "E506", (*base, rate_key), rate=rate, max=C.SECONDS_PER_HOUR / sim.dt, dt=sim.dt
        )
    if flow.end is None and flow.count is None and sim.duration is None:
        yield issue("E507", base)
    for road, share in _first_roads(flow):
        if road not in ctx.index.roads or road in ctx.broken.roads:
            continue
        cap = _capacity(ctx, d, flow, road)
        if cap is not None and rate * share > cap:
            yield issue("W501", (*base, rate_key), rate=_quote(rate * share), road=road, cap=cap)
            break
    if sim.duration is not None and flow.begin >= sim.duration:
        yield issue("W502", (*base, "begin"), t=flow.begin, d=sim.duration)


def check_demand(ctx: CheckContext) -> Iterator[ValidationIssue]:
    res = ctx.final()
    d = _demand_lookups(ctx)
    demand = res.demand
    generating = {f.id: "flow" for f in demand.flows} | {
        t.id: "transit line" for t in demand.transit
    }
    ids: dict[str, tuple[str, int]] = {}
    duration = res.simulation.duration
    groups: tuple[tuple[str, Sequence[FlowSpec | TripSpec | TransitLineSpec]], ...] = (
        ("flows", demand.flows),
        ("trips", demand.trips),
        ("transit", demand.transit),
    )
    for kind, items in groups:
        for i, item in enumerate(items):
            base: Loc = ("demand", kind, i)
            if item.id in ids:
                other_kind, j = ids[item.id]
                yield issue("E501", (*base, "id"), id=item.id, kind2=other_kind, j=j)
            else:
                ids[item.id] = (kind, i)
            match = re.fullmatch(r"(.+)\.\d+", item.id)
            if isinstance(item, TripSpec) and match and match.group(1) in generating:
                owner = match.group(1)
                yield issue(
                    "E501",
                    (*base, "id"),
                    variant=1,
                    id=item.id,
                    kind=generating[owner],
                    other=owner,
                )
            if isinstance(item, FlowSpec) and item.type_mix is not None:
                for t in item.type_mix:
                    yield from _vehicle_type(d, _key_path((*base, "type_mix"), t), t)
            else:
                yield from _vehicle_type(d, (*base, "vehicle_type"), str(item.vehicle_type))
            if isinstance(item, FlowSpec) and item.routes is not None:
                for r, choice in enumerate(item.routes):
                    yield from _route(ctx, d, (*base, "routes", r, "roads"), choice.roads)
            elif item.route is not None:
                yield from _route(ctx, d, (*base, "route"), item.route)
            elif not isinstance(item, TransitLineSpec):
                yield from _od(ctx, d, base, item)
            yield from _depart_lane(ctx, base, item)
            if isinstance(item, FlowSpec):
                yield from _flow_checks(ctx, d, base, item)
            elif duration is not None:
                start, key = _start(item)
                if start >= duration:
                    yield issue("W502", (*base, *key), t=start, d=duration)


def _start(item: TripSpec | TransitLineSpec) -> tuple[float, Loc]:
    """First departure time of a trip or transit line and the path of the field holding it."""
    if isinstance(item, TripSpec):
        return item.depart, ("depart",)
    if item.departures:
        return item.departures[0], ("departures", 0)
    return item.begin, ("begin",)


def check_transit(ctx: CheckContext) -> Iterator[ValidationIssue]:
    res, topo = ctx.final(), ctx.index
    d = _demand_lookups(ctx)
    connections: dict[tuple[str, str], set[int]] = {}
    for ix in res.network.intersections:
        for m in ix.movements or ():
            lanes = connections.setdefault((m.from_road, m.to_road), set())
            lanes.update(c.from_lane for c in m.connections or ())
    stop_ids: set[str | None] = set()
    for i, line in enumerate(res.demand.transit):
        base: Loc = ("demand", "transit", i)
        if any(r not in topo.roads or r in ctx.broken.roads for r in line.route):
            continue  # E502 already reported, or the road is broken
        vt = d.vtypes.get(line.vehicle_type)
        prev = (0, -math.inf)
        for k, stop in enumerate(line.stops):
            sb: Loc = (*base, "stops", k)
            if stop.id in stop_ids:
                yield issue("E606", (*sb, "id"), id=stop.id)
            stop_ids.add(stop.id)
            if stop.road not in line.route:
                yield issue("E601", (*sb, "road"), road=stop.road)
                continue
            at = next(
                (n for n in range(prev[0], len(line.route)) if line.route[n] == stop.road), None
            )
            if at is None or (at, stop.position) < prev:
                yield issue("E602", sb, k=k, k1=k - 1)
                continue
            prev = (at, stop.position)
            n = _lane_count(ctx, stop.road)
            lane = stop.lane if stop.lane is not None else n - 1
            if lane >= n:
                yield issue("E604", (*sb, "lane"), k=lane, road=stop.road, n=n)
                continue
            length = ctx.lane_lengths.get(stop.road)
            if length is None or vt is None:
                continue
            high = length - C.STOP_END_CLEARANCE
            if not vt.length <= stop.position <= high:
                yield issue("E807", (*sb, "position"), p=stop.position, len=vt.length, Lm1=high)
                continue
            if at + 1 < len(line.route):
                reachable = connections.get((stop.road, line.route[at + 1]), set())
                to_line = length - stop.position
                if lane not in reachable and to_line < C.STOP_NEAR_LINE_DISTANCE:
                    yield issue("W306", sb, d=to_line)


def check_simulation(ctx: CheckContext) -> Iterator[ValidationIssue]:
    sim = ctx.final().simulation
    if not C.DT_RECOMMENDED_MIN <= sim.dt <= C.DT_RECOMMENDED_MAX:
        yield issue("W701", ("simulation", "dt"), dt=sim.dt)
    if sim.duration is not None:
        rounded = math.floor(sim.duration / sim.dt + C.TIME_EPS) * sim.dt
        if abs(rounded - sim.duration) > C.TIME_EPS * max(1.0, sim.duration):
            yield issue(
                "W702", ("simulation", "duration"), d=sim.duration, dt=sim.dt, r=_quote(rounded)
            )


PRE_CHECKS: Final[tuple[Check, ...]] = (
    check_intersection_ids,
    check_road_ids,
    check_road_endpoints,
    check_road_geometry,
    check_intersections,
    check_derived_ids,
    check_vehicle_types,
)
POST_CHECKS: Final[tuple[Check, ...]] = (
    check_movements,
    check_signals,
    check_demand,
    check_transit,
    check_simulation,
)


def validate_spec(spec: ScenarioSpec) -> ValidationReport:
    """Run the semantic stages (P, D, Q) on a structurally valid spec.

    The version check of the load stage (E011/E012) runs too, for specs that were built in
    Python rather than loaded (loaded documents are migrated to the current version first).
    """
    ctx = CheckContext(spec, build_topology(spec))
    version = version_issue(spec.version)
    issues = [version] if version is not None else []
    issues += [found for check in PRE_CHECKS for found in check(ctx)]
    ctx.resolved = resolve(spec, skip=ctx.skipped())
    issues += [found for check in POST_CHECKS for found in check(ctx)]
    return ValidationReport(tuple(issues), spec, ctx.resolved)


def validate_data(data: Any) -> ValidationReport:
    """Validate a parsed JSON document through every stage; never raises."""
    if not isinstance(data, Mapping):
        return ValidationReport((issue("E010", "format", got=f"a JSON {type(data).__name__}"),))
    try:
        doc = migrate(dict(data))
    except ScenarioValidationError as exc:
        return ValidationReport(exc.issues)
    try:
        spec = ScenarioSpec.model_validate(doc)
    except ValidationError as exc:
        return ValidationReport(tuple(issues_from_pydantic(exc)))
    return validate_spec(spec)
