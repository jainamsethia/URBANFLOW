"""Editor operations: a small, whitelisted vocabulary of scenario edits (plan P, B.2 #15).

An editor sends ``{"op": name, ...fields}`` objects; :func:`apply_ops` applies a list of
them to a scenario document **atomically** (all or nothing) through
:class:`~urbanflow.scenario.builder.ScenarioBuilder` and returns the new document plus the
ids it created. Documents may be drafts (a map with no road yet): only each entity is
checked here; whole-document validation is the caller's (``validate_data``). Field names
are the scenario schema's (``rate``, ``speed_limit``, ``lanes``...); an invalid operation
raises :class:`~urbanflow.core.errors.ScenarioValidationError` with paths such as
``ops[3].lanes``. Only these operations exist: nothing is looked up by name or executed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from urbanflow.core import constants as C
from urbanflow.core.errors import ScenarioValidationError, ValidationIssue
from urbanflow.core.types import IntersectionKind
from urbanflow.scenario.builder import ScenarioBuilder
from urbanflow.scenario.schema import Arrival, Id
from urbanflow.scenario.validate import issues_from_pydantic

__all__ = ["BuilderOp", "OpsResult", "apply_ops"]

Point = tuple[float, float]
MAX_OPS = 1000


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class AddIntersection(_Op):
    op: Literal["add_intersection"]
    id: Id | None = None
    """Default: the first free ``J{n}``."""
    point: Point
    kind: IntersectionKind | None = None
    """Default: derived (one neighbour = map edge, ...)."""


class MoveIntersection(_Op):
    op: Literal["move_intersection"]
    id: Id
    point: Point


class UpdateIntersection(_Op):
    op: Literal["update_intersection"]
    id: Id
    kind: IntersectionKind | Literal["auto"]
    """``auto`` derives the kind again; leaving ``signalized`` drops the signal."""


class DeleteIntersection(_Op):
    op: Literal["delete_intersection"]
    id: Id
    """Its roads go too; flows that used them are reported by validation."""


class AddRoad(_Op):
    op: Literal["add_road"]
    id: Id | None = None
    """Default ``{from}_{to}`` (``.2``, ``.3``... if taken)."""
    from_: Id = Field(alias="from")
    to: Id
    lanes: int = Field(default=1, ge=1, le=C.MAX_LANES_PER_ROAD)
    speed_limit: float | None = Field(default=None, ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX)
    bidirectional: bool = True
    """Also add the road back (``{to}_{from}``)."""


class UpdateRoad(_Op):
    op: Literal["update_road"]
    id: Id
    lanes: int | None = Field(default=None, ge=1, le=C.MAX_LANES_PER_ROAD)
    speed_limit: float | None = Field(default=None, ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX)
    name: str | None = Field(default=None, max_length=200)


class DeleteRoad(_Op):
    op: Literal["delete_road"]
    id: Id


class SetController(_Op):
    op: Literal["set_controller"]
    intersection: Id
    controller: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    """Makes the intersection signalised if it is not."""


class SetSignalTiming(_Op):
    op: Literal["set_signal_timing"]
    intersection: Id
    green: float | None = Field(default=None, gt=0, le=600)
    """Green of every (derived) phase, s."""
    yellow: float | None = Field(default=None, ge=0, le=10)
    all_red: float | None = Field(default=None, ge=0, le=10)


class AddFlow(_Op):
    op: Literal["add_flow"]
    id: Id | None = None
    """Default: the first free ``flow{n}``."""
    origin: Id
    destination: Id
    rate: float = Field(gt=0, le=20_000)
    """veh/h."""
    arrival: Arrival = "poisson"
    begin: float = Field(default=0.0, ge=0)
    end: float | None = Field(default=None, gt=0)
    vehicle_type: Id = "car"


class UpdateFlow(_Op):
    op: Literal["update_flow"]
    id: Id
    rate: float | None = Field(default=None, gt=0, le=20_000)
    origin: Id | None = None
    destination: Id | None = None
    arrival: Arrival | None = None
    vehicle_type: Id | None = None


class DeleteFlow(_Op):
    op: Literal["delete_flow"]
    id: Id


class SetSimulation(_Op):
    op: Literal["set_simulation"]
    duration: float | None = Field(default=None, gt=0, le=7 * 86400)
    dt: float | None = Field(default=None, gt=0, le=2)
    seed: int | None = Field(default=None, ge=0)


class SetMetadata(_Op):
    op: Literal["set_metadata"]
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=10_000)


BuilderOp = Annotated[
    AddIntersection
    | MoveIntersection
    | UpdateIntersection
    | DeleteIntersection
    | AddRoad
    | UpdateRoad
    | DeleteRoad
    | SetController
    | SetSignalTiming
    | AddFlow
    | UpdateFlow
    | DeleteFlow
    | SetSimulation
    | SetMetadata,
    Field(discriminator="op"),
]
_ADAPTER: TypeAdapter[BuilderOp] = TypeAdapter(BuilderOp)


@dataclasses.dataclass(frozen=True, slots=True)
class OpsResult:
    data: dict[str, Any]
    """The edited scenario document (possibly still a draft)."""
    created: list[dict[str, Any]]
    """``{"op_index", "kind", "id"}`` of every entity an operation created."""


def _free(table: Mapping[str, Any], stem: str, first: int = 1) -> str:
    n = first
    while f"{stem}{n}" in table:
        n += 1
    return f"{stem}{n}"


def _road_id(table: Mapping[str, Any], a: str, z: str) -> str:
    rid, n = f"{a}_{z}", 2
    while rid in table:
        rid, n = f"{a}_{z}.{n}", n + 1
    return rid


def _apply(b: ScenarioBuilder, op: Any) -> list[tuple[str, str]]:
    """Apply one validated operation; ``(kind, id)`` of what it created."""
    roads = b._table("road")
    match op:
        case AddIntersection():
            jid = op.id or _free(b._table("intersection"), "J")
            b.intersection(jid, op.point, kind=op.kind)
            return [("intersection", jid)]
        case MoveIntersection():
            b.move(op.id, op.point)
        case UpdateIntersection():
            current = b._get("intersection", op.id)
            if op.kind == "auto":
                b.unset("intersection", op.id, "kind", "signal")
            else:
                b.update("intersection", op.id, kind=op.kind)
                if op.kind is not IntersectionKind.signalized and current.signal is not None:
                    b.unset("intersection", op.id, "signal")
        case DeleteIntersection():
            b.remove("intersection", op.id, cascade=True)
        case AddRoad():
            forward = op.id or _road_id(roads, op.from_, op.to)
            b.road(forward, op.from_, op.to, lanes=op.lanes, speed_limit=op.speed_limit)
            made = [("road", forward)]
            if op.bidirectional:
                back = _road_id(roads, op.to, op.from_)
                b.road(back, op.to, op.from_, lanes=op.lanes, speed_limit=op.speed_limit)
                made.append(("road", back))
            return made
        case UpdateRoad():
            fields: dict[str, Any] = {}
            if op.lanes is not None:
                fields["lanes"] = [{} for _ in range(op.lanes)]
            if op.speed_limit is not None:
                fields["speed_limit"] = op.speed_limit
            if op.name is not None:
                fields["name"] = op.name
            b.update("road", op.id, **fields)
        case DeleteRoad():
            b.remove("road", op.id, cascade=True)
        case SetController():
            ix = b._get("intersection", op.intersection)
            signal = ix.signal.model_dump(exclude_unset=True) if ix.signal is not None else {}
            signal["controller"] = {"type": op.controller, "params": op.params}
            b.update("intersection", op.intersection, kind="signalized", signal=signal)
        case SetSignalTiming():
            ix = b._get("intersection", op.intersection)
            controller = ix.signal.controller if ix.signal is not None else None
            b.signal(
                op.intersection,
                controller=controller.type if controller is not None else "fixed_time",
                green=op.green,
                yellow=op.yellow,
                all_red=op.all_red,
                **(controller.params if controller is not None else {}),
            )
        case AddFlow():
            fid = op.id or _free(b._table("flow"), "flow")
            b.flow(
                fid,
                origin=op.origin,
                destination=op.destination,
                rate=op.rate,
                arrival=op.arrival,
                begin=op.begin,
                end=op.end,
                vehicle_type=op.vehicle_type,
            )
            return [("flow", fid)]
        case UpdateFlow():
            changes = op.model_dump(exclude={"op", "id"}, exclude_none=True)
            b.update("flow", op.id, **changes)
        case DeleteFlow():
            b.remove("flow", op.id)
        case SetSimulation():
            b.simulation(**op.model_dump(exclude={"op"}, exclude_none=True))
        case SetMetadata():
            b.metadata(**op.model_dump(exclude={"op"}, exclude_none=True))
    return []


def apply_ops(data: Mapping[str, Any], ops: Sequence[Mapping[str, Any] | BaseModel]) -> OpsResult:
    """Apply ``ops`` in order to the scenario document ``data``, all or nothing."""
    if len(ops) > MAX_OPS:
        raise ScenarioValidationError(
            [ValidationIssue("ops", f"at most {MAX_OPS} operations per request", "E004")]
        )
    builder = ScenarioBuilder.from_data(data)
    created: list[dict[str, Any]] = []
    for i, raw in enumerate(ops):
        where = f"ops[{i}]"
        try:
            op = raw if isinstance(raw, BaseModel) else _ADAPTER.validate_python(raw)
        except ValidationError as exc:
            tag = str(raw.get("op", "")) if isinstance(raw, Mapping) else ""
            issues = [_untag(x, tag) for x in issues_from_pydantic(exc, root=None)]
            raise ScenarioValidationError([_at(where, x) for x in issues]) from None
        try:
            made = _apply(builder, op)
        except ScenarioValidationError as exc:
            raise ScenarioValidationError([_at(where, x) for x in exc.issues]) from None
        created += [{"op_index": i, "kind": kind, "id": eid} for kind, eid in made]
    return OpsResult(builder.to_data(), created)


def _untag(issue: ValidationIssue, tag: str) -> ValidationIssue:
    """Drop the operation name pydantic puts first in paths of a tagged union."""
    if tag and (issue.path == tag or issue.path.startswith(f"{tag}.")):
        return dataclasses.replace(issue, path=issue.path[len(tag) + 1 :])
    return issue


def _at(where: str, issue: ValidationIssue) -> ValidationIssue:
    inner = ".".join("from" if part == "from_" else part for part in issue.path.split("."))
    path = where if inner in ("", "$") else f"{where}.{inner}"
    return dataclasses.replace(issue, path=path)
