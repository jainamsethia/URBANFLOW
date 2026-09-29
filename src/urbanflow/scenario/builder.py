"""Fluent construction and editing of scenarios (plan E.9).

Entities live in insertion-ordered dicts keyed by id. Methods return ``self``, validate
their arguments with ``pydantic.validate_call`` and do only local checks (duplicate or
unknown ids); each problem raises a single-issue
:class:`~urbanflow.core.errors.ScenarioValidationError`. :meth:`ScenarioBuilder.build`
runs the full validation. ponytail: the editor-op surface (``apply``, ``OPS`` and the
op-only methods) lands with ``scenario/ops.py`` in P14 (plan U.3).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationError, validate_call

from urbanflow.core import constants as C
from urbanflow.core.errors import ScenarioValidationError, suggest
from urbanflow.core.types import DriveSide, IntersectionKind, TurnKind
from urbanflow.scenario.derive import resolve
from urbanflow.scenario.io import CURRENT_VERSION
from urbanflow.scenario.schema import (
    FORMAT,
    Arrival,
    DepartLane,
    DepartSpeed,
    FlowSpec,
    IntersectionSpec,
    LaneSpec,
    MetaSpec,
    MovementSpec,
    PhaseSpec,
    RoadSpec,
    ScenarioSpec,
    SignalSpec,
    SignalTemplate,
    SimulationSpec,
    StopSpec,
    TransitLineSpec,
    TripSpec,
    VehicleTypeSpec,
)
from urbanflow.scenario.validate import issue, issues_from_pydantic

if TYPE_CHECKING:
    from urbanflow.scenario.scenario import Scenario

__all__ = ["EntityKind", "ScenarioBuilder"]

EntityKind = Literal["intersection", "road", "vehicle_type", "flow", "trip", "transit"]
_LABELS: dict[str, str] = {
    "intersection": "intersection",
    "road": "road",
    "vehicle_type": "vehicle type",
    "flow": "flow",
    "trip": "trip",
    "transit": "transit line",
}
_MODELS: dict[str, type[BaseModel]] = {
    "intersection": IntersectionSpec,
    "road": RoadSpec,
    "vehicle_type": VehicleTypeSpec,
    "flow": FlowSpec,
    "trip": TripSpec,
    "transit": TransitLineSpec,
}
_MAX_LISTED_REFS = 5
_CALL_CONFIG = ConfigDict(arbitrary_types_allowed=True)


def _fail(code: str, path: str, **fmt: Any) -> ScenarioValidationError:
    return ScenarioValidationError([issue(code, path, **fmt)])


def _entity[M: BaseModel](model: type[M], data: Mapping[str, Any]) -> M:
    """Validate one entity; errors carry paths relative to it (= the argument names)."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ScenarioValidationError(issues_from_pydantic(exc, root=model)) from None


def _set(**values: Any) -> dict[str, Any]:
    """Only the given values that are not None (so unset fields stay unset in the spec)."""
    return {k: v for k, v in values.items() if v is not None}


def _op[F: Callable[..., Any]](fn: F) -> F:
    """Validate arguments with ``pydantic.validate_call``; report errors as issues."""
    validated = validate_call(config=_CALL_CONFIG)(fn)
    signature = inspect.signature(fn)
    names = [n for n in signature.parameters if n != "self"]
    open_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )

    @functools.wraps(fn)
    def wrapper(self: ScenarioBuilder, /, *args: Any, **kwargs: Any) -> Any:
        if not open_kwargs:
            unknown = next((k for k in kwargs if k not in signature.parameters), None)
            if unknown is not None:
                raise _fail("E002", unknown, field=unknown, hint=suggest(unknown, names))
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            raise _fail("E009", "$", message=str(exc)) from None
        positional: list[Any] = []
        call: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            kind = signature.parameters[name].kind
            if name == "self":
                continue
            if kind is inspect.Parameter.VAR_KEYWORD:
                call.update(value)
            elif kind is inspect.Parameter.POSITIONAL_ONLY:
                positional.append(value)
            else:
                call[name] = value
        try:
            return validated(self, *positional, **call)
        except ValidationError as exc:
            raise ScenarioValidationError(
                issues_from_pydantic(exc, root=None, names=names)
            ) from None

    return cast(F, wrapper)


class ScenarioBuilder:
    """Build or edit a scenario step by step (see plan E.9 for the 2x2 grid example)."""

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        drive_side: DriveSide | Literal["right", "left"] = "right",
        lane_width: float = C.LANE_WIDTH,
        speed_limit: float = C.SPEED_LIMIT,
        **simulation: Any,
    ) -> None:
        self._meta = _entity(MetaSpec, _set(name=name, description=description or None))
        _entity(SimulationSpec, simulation)
        self._simulation: dict[str, Any] = dict(simulation)
        self._network: dict[str, Any] = {
            "drive_side": drive_side,
            "lane_width": lane_width,
            "speed_limit": speed_limit,
        }
        self._intersections: dict[str, IntersectionSpec] = {}
        self._roads: dict[str, RoadSpec] = {}
        self._vehicle_types: dict[str, VehicleTypeSpec] = {}
        self._flows: dict[str, FlowSpec] = {}
        self._trips: dict[str, TripSpec] = {}
        self._transit: dict[str, TransitLineSpec] = {}

    @classmethod
    def from_scenario(cls, scenario: Scenario | ScenarioSpec) -> Self:
        """A builder with the *declared* entities of ``scenario`` (derived parts stay derived)."""
        spec = scenario if isinstance(scenario, ScenarioSpec) else scenario.spec
        builder = cls(spec.meta.name)
        builder._meta = spec.meta
        builder._simulation = spec.simulation.model_dump(exclude_unset=True)
        builder._network = spec.network.model_dump(
            exclude_unset=True, exclude={"intersections", "roads"}
        )
        builder._intersections = {ix.id: ix for ix in spec.network.intersections}
        builder._roads = {r.id: r for r in spec.network.roads}
        builder._vehicle_types = {vt.id: vt for vt in spec.vehicle_types}
        builder._flows = {f.id: f for f in spec.demand.flows}
        builder._trips = {t.id: t for t in spec.demand.trips}
        builder._transit = {t.id: t for t in spec.demand.transit}
        return builder

    # ------------------------------------------------------------------ helpers
    def _table(self, kind: str) -> dict[str, Any]:
        tables: dict[str, dict[str, Any]] = {
            "intersection": self._intersections,
            "road": self._roads,
            "vehicle_type": self._vehicle_types,
            "flow": self._flows,
            "trip": self._trips,
            "transit": self._transit,
        }
        return tables[kind]

    def _get(self, kind: str, id: str, path: str = "id") -> Any:
        table = self._table(kind)
        if id not in table:
            raise _fail("E021", path, kind=_LABELS[kind], id=id, hint=suggest(id, table))
        return table[id]

    def _state(self) -> tuple[Any, ...]:
        tables = tuple(dict(self._table(k)) for k in _LABELS)
        return (self._meta, dict(self._simulation), dict(self._network), *tables)

    def _restore(self, state: tuple[Any, ...]) -> None:
        self._meta, self._simulation, self._network = state[0], state[1], state[2]
        for kind, table in zip(_LABELS, state[3:], strict=True):
            current = self._table(kind)
            current.clear()
            current.update(table)

    @contextmanager
    def _atomic(self) -> Iterator[None]:
        """Undo every change made inside the block if it raises."""
        saved = self._state()
        try:
            yield
        except BaseException:
            self._restore(saved)
            raise

    def _replace(self, id: str, **update: Any) -> IntersectionSpec:
        ix = self._intersections[id].model_copy(update=update)
        self._intersections[id] = ix
        return ix

    def _demand_id_free(self, id: str) -> None:
        for kind, table in (
            ("flows", self._flows),
            ("trips", self._trips),
            ("transit", self._transit),
        ):
            if id in table:
                raise _fail("E501", "id", id=id, kind2=kind, j=list(table).index(id))

    def _resolved_intersection(self, id: str) -> IntersectionSpec:
        resolved = resolve(self.to_spec())
        return next(ix for ix in resolved.network.intersections if ix.id == id)

    # ------------------------------------------------------------------ simulation
    @_op
    def simulation(self, **fields: Any) -> Self:
        """Set scenario-level simulation defaults (dt, duration, seed, ...)."""
        merged = {**self._simulation, **fields}
        _entity(SimulationSpec, merged)
        self._simulation = merged
        return self

    # ------------------------------------------------------------------ network
    @_op
    def intersection(
        self,
        id: str,
        point: tuple[float, float],
        *,
        kind: IntersectionKind | str | None = None,
        radius: float | None = None,
        major_roads: Sequence[str] | None = None,
    ) -> Self:
        """Add an intersection; omitted fields are derived."""
        if id in self._intersections:
            raise _fail("E101", "id", id=id, j=list(self._intersections).index(id))
        data = _set(id=id, point=point, kind=kind, radius=radius, major_roads=major_roads)
        self._intersections[id] = _entity(IntersectionSpec, data)
        return self

    @_op
    def boundary(self, id: str, point: tuple[float, float]) -> Self:
        """Add a boundary node (where vehicles enter and leave the network)."""
        return self.intersection(id, point, kind=IntersectionKind.boundary)

    @_op
    def road(
        self,
        id: str,
        from_: str,
        to: str,
        *,
        lanes: int | Sequence[LaneSpec | Mapping[str, Any]] = 1,
        speed_limit: float | None = None,
        lane_width: float | None = None,
        points: Sequence[tuple[float, float]] | None = None,
        name: str = "",
    ) -> Self:
        """Add a one-way road; ``lanes`` is a count or per-lane specs (index 0 = median)."""
        if id in self._roads:
            raise _fail("E102", "id", id=id, j=list(self._roads).index(id))
        for path, ref in (("from_", from_), ("to", to)):
            if ref not in self._intersections:
                raise _fail("E103", path, id=ref, hint=suggest(ref, self._intersections))
        lane_specs = [{} for _ in range(lanes)] if isinstance(lanes, int) else list(lanes)
        data = _set(
            id=id,
            to=to,
            lanes=lane_specs,
            speed_limit=speed_limit,
            lane_width=lane_width,
            points=points,
            name=name or None,
        )
        self._roads[id] = _entity(RoadSpec, {"from": from_, **data})
        return self

    @_op
    def two_way(
        self,
        a: str,
        b: str,
        *,
        lanes: int = 1,
        lanes_back: int | None = None,
        speed_limit: float | None = None,
        points: Sequence[tuple[float, float]] | None = None,
        ids: tuple[str, str] | None = None,
    ) -> Self:
        """Add roads ``a -> b`` and ``b -> a`` (ids default to ``"{a}_{b}"`` and ``"{b}_{a}"``)."""
        forward, backward = ids or (f"{a}_{b}", f"{b}_{a}")
        with self._atomic():
            self.road(forward, a, b, lanes=lanes, speed_limit=speed_limit, points=points)
            self.road(
                backward,
                b,
                a,
                lanes=lanes if lanes_back is None else lanes_back,
                speed_limit=speed_limit,
                points=None if points is None else list(points)[::-1],
            )
        return self

    @_op
    def move(self, intersection: str, point: tuple[float, float]) -> Self:
        """Move an intersection; roads with default (straight) geometry follow it."""
        ix: IntersectionSpec = self._get("intersection", intersection, "intersection")
        self._intersections[intersection] = _entity(
            IntersectionSpec, {**ix.model_dump(exclude_unset=True), "point": point}
        )
        return self

    # ------------------------------------------------------------------ movements
    @_op
    def movement(
        self,
        intersection: str,
        from_road: str,
        to_road: str,
        *,
        connections: Sequence[tuple[int, int]] | None = None,
        turn: TurnKind | str | None = None,
        priority: Literal["major", "minor"] | None = None,
        id: str | None = None,
    ) -> Self:
        """Add an explicit movement (the intersection's movement list becomes authoritative)."""
        ix: IntersectionSpec = self._get("intersection", intersection, "intersection")
        for path, road in (("from_road", from_road), ("to_road", to_road)):
            if road not in self._roads:
                raise _fail("E203", path, road=road, hint=suggest(road, self._roads))
        existing = list(ix.movements or ())
        mid = id or f"{from_road}->{to_road}"
        i = list(self._intersections).index(intersection)
        for k, mov in enumerate(existing):
            if (mov.from_road, mov.to_road) == (from_road, to_road):
                raise _fail("E205", "to_road", a=from_road, b=to_road, j=k)
            if (mov.id or f"{mov.from_road}->{mov.to_road}") == mid:
                where = f"network.intersections[{i}].movements[{k}]"
                raise _fail("E204", "id", id=mid, where=where)
        conns = (
            None
            if connections is None
            else [{"from_lane": a, "to_lane": b} for a, b in connections]
        )
        data = _set(
            id=id,
            from_road=from_road,
            to_road=to_road,
            turn=turn,
            priority=priority,
            connections=conns,
        )
        self._replace(intersection, movements=(*existing, _entity(MovementSpec, data)))
        return self

    @_op
    def materialize(self, intersection: str) -> Self:
        """Copy the derived movements and signal phases of ``intersection`` into explicit form."""
        ix: IntersectionSpec = self._get("intersection", intersection, "intersection")
        derived = self._resolved_intersection(intersection)
        update: dict[str, Any] = {}
        if ix.movements is None and derived.movements is not None:
            update["movements"] = derived.movements
        if derived.kind is IntersectionKind.signalized and derived.signal is not None:
            signal = ix.signal or SignalSpec()
            if signal.phases is None and derived.signal.phases is not None:
                update["signal"] = signal.model_copy(update={"phases": derived.signal.phases})
        if update:
            self._replace(intersection, **update)
        return self

    # ------------------------------------------------------------------ signals
    @_op
    def signal(
        self,
        intersection: str,
        *,
        controller: str = "fixed_time",
        phases: Sequence[PhaseSpec | Mapping[str, Any]] | None = None,
        template: SignalTemplate = "auto",
        green: float | None = None,
        yellow: float | None = None,
        all_red: float | None = None,
        min_green: float | None = None,
        max_green: float | None = None,
        **controller_params: Any,
    ) -> Self:
        """Signalise an intersection.

        Without ``phases`` they are derived from ``template``; ``green`` then fixes every
        derived phase's duration (materialising the phases).
        """
        ix: IntersectionSpec = self._get("intersection", intersection, "intersection")
        ctrl: dict[str, Any] = {"type": controller}
        if controller_params:
            ctrl["params"] = controller_params
        data = _set(
            controller=ctrl,
            phases=None if phases is None else list(phases),
            template=None if template == "auto" else template,
            yellow=yellow,
            all_red=all_red,
            min_green=min_green,
            max_green=max_green,
        )
        with self._atomic():
            self._replace(
                intersection,
                kind=ix.kind or IntersectionKind.signalized,
                signal=_entity(SignalSpec, data),
            )
            if green is not None and phases is None:
                self._retime([intersection], lambda _: green)
        return self

    def _retime(self, ids: Sequence[str], duration: Callable[[SignalSpec], float]) -> None:
        """Materialise the derived phases of ``ids`` with ``duration(signal)`` s of green each."""
        wanted = set(ids)
        for derived in resolve(self.to_spec()).network.intersections:
            sig = derived.signal
            if derived.id not in wanted or sig is None or not sig.phases:
                continue
            green = duration(sig)
            phases = tuple(
                _entity(PhaseSpec, {**p.model_dump(exclude_unset=True), "duration": green})
                for p in sig.phases
            )
            signal = self._intersections[derived.id].signal or SignalSpec()
            self._replace(derived.id, signal=signal.model_copy(update={"phases": phases}))

    @_op
    def signal_all(self, **kwargs: Any) -> Self:
        """:meth:`signal` every (derived or declared) signalized intersection without a signal."""
        green = kwargs.pop("green", None)
        targets = [
            ix.id
            for ix in resolve(self.to_spec()).network.intersections
            if ix.kind is IntersectionKind.signalized and self._intersections[ix.id].signal is None
        ]
        with self._atomic():
            for j in targets:
                self.signal(j, **kwargs)
            if green is not None and kwargs.get("phases") is None:
                self._retime(targets, lambda _: float(green))
        return self

    # ------------------------------------------------------------------ vehicle types, demand
    @_op
    def vehicle_type(self, id: str, **fields: Any) -> Self:
        """Add a vehicle type (fields override the built-in type of the same id)."""
        if id in self._vehicle_types:
            raise _fail("E401", "id", id=id)
        self._vehicle_types[id] = _entity(VehicleTypeSpec, {"id": id, **fields})
        return self

    @_op
    def flow(
        self,
        id: str,
        *,
        route: Sequence[str] | None = None,
        routes: Sequence[tuple[Sequence[str], float] | Mapping[str, Any]] | None = None,
        origin: str | None = None,
        destination: str | None = None,
        via: Sequence[str] = (),
        rate: float | None = None,
        period: float | None = None,
        arrival: Arrival = "uniform",
        begin: float = 0.0,
        end: float | None = None,
        count: int | None = None,
        vehicle_type: str = "car",
        type_mix: Mapping[str, float] | None = None,
        depart_lane: DepartLane = "best",
        depart_speed: DepartSpeed = "max",
    ) -> Self:
        """Add a flow: ``route``/``routes``/``origin``+``destination``; ``rate`` or ``period``."""
        self._demand_id_free(id)
        choices = None
        if routes is not None:
            choices = [
                dict(r) if isinstance(r, Mapping) else {"roads": list(r[0]), "weight": r[1]}
                for r in routes
            ]
        data = _set(
            id=id,
            route=route,
            routes=choices,
            origin=origin,
            destination=destination,
            via=list(via) or None,
            rate=rate,
            period=period,
            arrival=None if arrival == "uniform" else arrival,
            begin=begin or None,
            end=end,
            count=count,
            vehicle_type=None if vehicle_type == "car" and type_mix is not None else vehicle_type,
            type_mix=type_mix,
            depart_lane=None if depart_lane == "best" else depart_lane,
            depart_speed=None if depart_speed == "max" else depart_speed,
        )
        if data.get("vehicle_type") == "car":
            del data["vehicle_type"]  # the default; keeps the declared file short
        self._flows[id] = _entity(FlowSpec, data)
        return self

    @_op
    def trip(
        self,
        id: str,
        depart: float,
        *,
        route: Sequence[str] | None = None,
        origin: str | None = None,
        destination: str | None = None,
        via: Sequence[str] = (),
        vehicle_type: str = "car",
        depart_lane: DepartLane = "best",
        depart_speed: DepartSpeed = "max",
    ) -> Self:
        """Add a single vehicle departing at ``depart`` seconds."""
        self._demand_id_free(id)
        data = _set(
            id=id,
            depart=depart,
            route=route,
            origin=origin,
            destination=destination,
            via=list(via) or None,
            vehicle_type=None if vehicle_type == "car" else vehicle_type,
            depart_lane=None if depart_lane == "best" else depart_lane,
            depart_speed=None if depart_speed == "max" else depart_speed,
        )
        self._trips[id] = _entity(TripSpec, data)
        return self

    @_op
    def transit_line(
        self,
        id: str,
        route: Sequence[str],
        stops: Sequence[StopSpec | Mapping[str, Any] | tuple[str, float]],
        *,
        headway: float | None = None,
        departures: Sequence[float] | None = None,
        begin: float = 0.0,
        end: float | None = None,
        vehicle_type: str = "bus",
        dwell: float = C.STOP_DWELL,
    ) -> Self:
        """Add a bus line; stops are ``(road, position)`` pairs, mappings or StopSpecs."""
        self._demand_id_free(id)
        stop_data: list[Any] = []
        for stop in stops:
            if isinstance(stop, StopSpec):
                fixed = "dwell" in stop.model_fields_set
                stop_data.append(stop if fixed else stop.model_copy(update={"dwell": dwell}))
            elif isinstance(stop, Mapping):
                stop_data.append({"dwell": dwell, **stop})
            else:
                stop_data.append({"road": stop[0], "position": stop[1], "dwell": dwell})
        data = _set(
            id=id,
            route=route,
            stops=stop_data,
            headway=headway,
            departures=departures,
            begin=begin or None,
            end=end,
            vehicle_type=None if vehicle_type == "bus" else vehicle_type,
        )
        self._transit[id] = _entity(TransitLineSpec, data)
        return self

    # ------------------------------------------------------------------ generic edits
    @_op
    def update(self, kind: EntityKind, id: str, /, **fields: Any) -> Self:
        """Change fields of an existing entity (re-validated; the id cannot change)."""
        current: BaseModel = self._get(kind, id)
        if "id" in fields and fields["id"] != id:
            raise _fail("E024", "id", kind=_LABELS[kind])
        changes = {("from_" if k == "from" else k): v for k, v in fields.items()}
        data = {**current.model_dump(exclude_unset=True, by_alias=False), **changes}
        self._table(kind)[id] = _entity(_MODELS[kind], data)
        return self

    def _references(self, kind: str, id: str) -> list[str]:
        """Human-readable descriptions of everything that refers to ``kind`` ``id``."""
        refs: list[str] = []
        if kind == "intersection":
            refs += [f'road "{r.id}"' for r in self._roads.values() if id in (r.from_, r.to)]
        if kind == "road":
            for ix in self._intersections.values():
                refs += [
                    f'movement "{m.from_road}->{m.to_road}" of "{ix.id}"'
                    for m in ix.movements or ()
                    if id in (m.from_road, m.to_road)
                ]
                if id in (ix.major_roads or ()):
                    refs.append(f'major_roads of "{ix.id}"')
        if kind in {"road", "vehicle_type"}:
            demand: list[tuple[str, BaseModel]] = [
                *(("flow", f) for f in self._flows.values()),
                *(("trip", t) for t in self._trips.values()),
                *(("transit line", t) for t in self._transit.values()),
            ]
            refs += [
                f'{label} "{getattr(item, "id", "")}"'
                for label, item in demand
                if id in _demand_refs(item, kind)
            ]
        return refs

    @_op
    def remove(self, kind: EntityKind, id: str, *, cascade: bool = False) -> Self:
        """Remove an entity. Without ``cascade`` anything referencing it is an error.

        ``cascade`` removes an intersection's roads, and a road's explicit movements and
        major-road entries; demand is never deleted (dangling references become
        validation issues).
        """
        self._get(kind, id)
        refs = self._references(kind, id)
        if refs and not cascade:
            listed = ", ".join(refs[:_MAX_LISTED_REFS])
            more = len(refs) - _MAX_LISTED_REFS
            text = listed + (f" and {more} more" if more > 0 else "")
            raise _fail("E022", "id", kind=_LABELS[kind], id=id, refs=text)
        with self._atomic():
            if kind == "intersection":
                for road in [r.id for r in self._roads.values() if id in (r.from_, r.to)]:
                    self.remove("road", road, cascade=True)
            if kind == "road":
                self._drop_road_references(id)
            del self._table(kind)[id]
        return self

    def _drop_road_references(self, road: str) -> None:
        """Strip ``road`` from explicit movements, major roads and phases (cascade removal)."""
        derived = {ix.id: ix for ix in resolve(self.to_spec()).network.intersections}
        for j, ix in list(self._intersections.items()):
            update: dict[str, Any] = {}
            if ix.major_roads is not None and road in ix.major_roads:
                update["major_roads"] = tuple(r for r in ix.major_roads if r != road)
            if ix.movements is not None:
                kept = tuple(m for m in ix.movements if road not in (m.from_road, m.to_road))
                if len(kept) != len(ix.movements):
                    update["movements"] = kept
            gone = {
                str(m.id) for m in derived[j].movements or () if road in (m.from_road, m.to_road)
            }
            if gone and ix.signal is not None:
                update["signal"] = _without_green(ix.signal, gone)
            if update:
                self._replace(j, **update)

    # ------------------------------------------------------------------ output
    def to_spec(self) -> ScenarioSpec:
        """The declared spec (structural validation only)."""
        major, minor = CURRENT_VERSION
        data: dict[str, Any] = {
            "format": FORMAT,
            "version": f"{major}.{minor}",
            "meta": self._meta,
            "network": {
                **self._network,
                "intersections": tuple(self._intersections.values()),
                "roads": tuple(self._roads.values()),
            },
        }
        if self._simulation:
            data["simulation"] = self._simulation
        if self._vehicle_types:
            data["vehicle_types"] = tuple(self._vehicle_types.values())
        demand = {
            key: tuple(table.values())
            for key, table in (
                ("flows", self._flows),
                ("trips", self._trips),
                ("transit", self._transit),
            )
            if table
        }
        if demand:
            data["demand"] = demand
        return _entity(ScenarioSpec, data)

    def build(self, *, validate: bool = True) -> Scenario:
        """The finished :class:`Scenario`; with ``validate`` every error raises."""
        from urbanflow.scenario.scenario import Scenario

        spec = self.to_spec()
        return Scenario.from_spec(spec) if validate else Scenario(spec)


def _without_green(signal: SignalSpec, gone: set[str]) -> SignalSpec:
    """``signal`` without green entries for the ``gone`` movements (empty phases dropped)."""
    if signal.phases is None:
        return signal
    phases = [
        p.model_copy(update={"green": {m: s for m, s in p.green.items() if m not in gone}})
        for p in signal.phases
    ]
    return signal.model_copy(update={"phases": tuple(p for p in phases if p.green) or None})


def _demand_refs(item: BaseModel, kind: str) -> set[str]:
    """Road ids (``kind="road"``) or vehicle type ids referenced by a demand entity."""
    if kind == "vehicle_type":
        types = {getattr(item, "vehicle_type", None)}
        types.update(getattr(item, "type_mix", None) or ())
        return {t for t in types if t}
    roads: set[str] = set(getattr(item, "route", None) or ())
    roads.update(getattr(item, "via", ()))
    roads.update(
        r for r in (getattr(item, "origin", None), getattr(item, "destination", None)) if r
    )
    for choice in getattr(item, "routes", None) or ():
        roads.update(choice.roads)
    roads.update(stop.road for stop in getattr(item, "stops", ()))
    return roads
