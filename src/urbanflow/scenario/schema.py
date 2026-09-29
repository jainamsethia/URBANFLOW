"""Scenario file format v1.0: frozen pydantic spec models and the JSON Schema export (plan E.7).

Every model forbids unknown keys, rejects NaN/inf and is immutable; collections are tuples
and mappings are typed ``Mapping`` so nothing can be mutated in place. Fields marked
"derived" may be omitted: :func:`urbanflow.scenario.derive.resolve` fills them. Attribute
docstrings (units included) become JSON Schema ``description`` values, which editors show
on hover.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from urbanflow.core import constants as C
from urbanflow.core.config import RoutingWeight
from urbanflow.core.types import DriveSide, IntersectionKind, TurnKind, VehicleClass

__all__ = [
    "DEFAULT_VEHICLE_TYPES",
    "EXCLUSIVE_MESSAGES",
    "FORMAT",
    "ID_PATTERN",
    "SCHEMA_ID",
    "Arrival",
    "ConnectionSpec",
    "ControllerSpec",
    "DemandSpec",
    "DepartLane",
    "DepartSpeed",
    "FlowSpec",
    "GeneratorInfo",
    "Id",
    "IntersectionSpec",
    "LaneSpec",
    "MetaSpec",
    "Metres",
    "MovementSpec",
    "NetworkSpec",
    "PhaseSpec",
    "Point",
    "RoadSpec",
    "RouteChoiceSpec",
    "ScenarioSpec",
    "Seconds",
    "SignalSpec",
    "SignalTemplate",
    "SimulationSpec",
    "Speed",
    "SpeedFactorSpec",
    "StopSpec",
    "TransitLineSpec",
    "TripSpec",
    "VehicleTypeSpec",
    "scenario_json_schema",
]

FORMAT: Final = "urbanflow.scenario"
SCHEMA_ID: Final = "urn:urbanflow:scenario:1.0"
ID_PATTERN: Final = rf"^[A-Za-z0-9_.:>\-]{{1,{C.MAX_ID_LENGTH}}}$"

Id = Annotated[str, StringConstraints(pattern=ID_PATTERN)]
Coordinate = Annotated[float, Field(ge=-C.MAX_COORDINATE, le=C.MAX_COORDINATE)]
Point = tuple[Coordinate, Coordinate]
Seconds = Annotated[float, Field(ge=0)]
Speed = Annotated[float, Field(ge=C.SPEED_LIMIT_MIN, le=C.SPEED_LIMIT_MAX)]
Metres = Annotated[float, Field(ge=0)]
LaneWidth = Annotated[float, Field(ge=C.LANE_WIDTH_MIN, le=C.LANE_WIDTH_MAX)]
DepartLane = Literal["best", "random", "first"] | Annotated[int, Field(ge=0)]
DepartSpeed = Literal["max"] | Annotated[float, Field(ge=0)]
SignalTemplate = Literal["auto", "two_phase", "protected_left", "split"]
Arrival = Literal["uniform", "poisson", "binomial"]

# E007 messages raised by the cross-field validators below (`uf_exclusive`, plan E.8).
EXCLUSIVE_MESSAGES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "rate_period": 'specify exactly one of "rate" or "period"',
        "flow_route": 'specify exactly one of "route", "routes" or "origin" and "destination"',
        "type_mix": '"vehicle_type" and "type_mix" are mutually exclusive',
        "headway": 'specify exactly one of "headway" or "departures"',
        "end_begin": '"end" ({end}) must be greater than "begin" ({begin})',
        "green": "max_green ({a}) must exceed min_green ({b})",
        "decel": "emergency_decel ({e}) must be >= decel ({d})",
        "speed_factor": "speed_factor requires min <= mean <= max",
        "trip_route": 'specify exactly one of "route" or "origin" and "destination"',
        "via": '"via" requires "origin" and "destination"',
        "departures": '"departures" must be sorted in ascending order',
    }
)


def _exclusive(rule: str, **ctx: Any) -> PydanticCustomError:
    return PydanticCustomError("uf_exclusive", EXCLUSIVE_MESSAGES[rule], ctx)


def _check_window(begin: float, end: float | None) -> None:
    if end is not None and end <= begin:
        raise _exclusive("end_begin", end=end, begin=begin)


def _partial_override(type_id: object, set_fields: set[str], related: frozenset[str]) -> bool:
    """True when a built-in type is overridden in only some of ``related`` fields.

    Its cross-field rule then depends on the built-in values, so it is checked on the
    merged type (``validate.check_vehicle_types``), not on the override alone.
    """
    return type_id in C.BUILTIN_VEHICLE_TYPES and not related <= set_fields


_FACTOR_FIELDS: Final = frozenset({"mean", "min", "max"})
_BRAKING_FIELDS: Final = frozenset({"decel", "emergency_decel"})


class Spec(BaseModel):
    """Base of every scenario model: frozen, strict about keys, finite numbers only."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        use_attribute_docstrings=True,
    )


# --------------------------------------------------------------------------- meta
class GeneratorInfo(Spec):
    """Provenance written by generators and importers."""

    name: str
    """Generator or importer name."""
    params: Mapping[str, Any] = Field(default_factory=dict)
    """Parameters it was called with."""
    urbanflow_version: str
    """UrbanFlow version that produced the file."""


class MetaSpec(Spec):
    """Descriptive metadata; excluded from the content hash."""

    name: str = Field(min_length=1, max_length=C.MAX_NAME_LENGTH)
    """Scenario name."""
    description: str = Field(default="", max_length=C.MAX_DESCRIPTION_LENGTH)
    """Free-text description."""
    tags: tuple[str, ...] = Field(default=(), max_length=C.MAX_TAGS)
    """Free-form tags."""
    authors: tuple[str, ...] = ()
    """Authors."""
    generator: GeneratorInfo | None = None
    """Provenance (set by generators and importers)."""


class SimulationSpec(Spec):
    """Scenario-level defaults of the run configuration (same fields as SimulationConfig)."""

    dt: float = Field(default=C.DT, ge=C.DT_MIN, le=C.DT_MAX)
    """Step length, s (W701 outside 0.2-1.0 s)."""
    duration: float | None = Field(default=C.DURATION, gt=0)
    """Simulated time, s; null runs until demand is exhausted and the network is empty."""
    seed: int = Field(default=0, ge=0, le=C.SEED_MAX)
    """Root seed of every random stream."""
    car_following: str = "idm"
    """Registered car-following model."""
    lane_change_model: str = "mobil"
    """Registered lane-change model."""
    lane_changing: bool = True
    """Enable lane changes."""
    router: str = "shortest"
    """Registered router."""
    routing_weight: RoutingWeight = "freeflow_time"
    """Edge weight of the shortest-path router."""
    halting_speed: float = Field(default=C.HALTING_SPEED, ge=0)
    """Vehicles slower than this are halting, m/s."""
    deadlock_timeout: float = Field(default=C.DEADLOCK_TIMEOUT, ge=0)
    """Teleport vehicles stuck this long, s; 0 turns the watchdog off."""
    max_vehicles: int | None = Field(default=None, ge=1)
    """Cap on running vehicles; extra demand queues."""


# --------------------------------------------------------------------------- network
class LaneSpec(Spec):
    """One lane; lane 0 is the median (innermost) lane."""

    width: LaneWidth | None = None
    """Lane width, m (2-5); derived: the road's lane_width."""
    speed_limit: Speed | None = None
    """Speed limit, m/s (1-70); derived: the road's speed_limit."""


class ConnectionSpec(Spec):
    """A lane-to-lane connector through an intersection."""

    from_lane: int = Field(ge=0)
    """Lane index on the incoming road."""
    to_lane: int = Field(ge=0)
    """Lane index on the outgoing road."""
    shape: tuple[Point, ...] | None = Field(default=None, min_length=2)
    """Explicit connector polyline [[x, y], ...]; null = cubic Bezier from the lane ends."""


class MovementSpec(Spec):
    """A road-to-road turn at an intersection."""

    id: Id | None = None
    """Movement id; derived: "{from_road}->{to_road}"."""
    from_road: Id
    """Incoming road (ends at this intersection)."""
    to_road: Id
    """Outgoing road (starts at this intersection)."""
    turn: TurnKind | None = None
    """straight, left, right or uturn; derived from geometry."""
    priority: Literal["major", "minor"] | None = None
    """Right of way at priority intersections; derived from major_roads."""
    connections: tuple[ConnectionSpec, ...] | None = Field(default=None, min_length=1)
    """Lane connections; derived from the lane mapping rules."""


class ControllerSpec(Spec):
    """Signal controller: a registry name plus its parameters."""

    type: str = Field(default="fixed_time", min_length=1)
    """Registered controller (fixed_time, actuated, max_pressure, webster, external, ...)."""
    params: Mapping[str, Any] = Field(default_factory=dict)
    """Controller parameters, checked against the controller's Params model (deep check)."""


class PhaseSpec(Spec):
    """A signal phase: the movements that have green."""

    id: Id | None = None
    """Phase id; derived: "p{k}"."""
    green: Mapping[Id, Literal["G", "g"]] = Field(min_length=1)
    """movement id -> "G" (protected) or "g" (permissive); unlisted movements are red."""
    duration: float = Field(default=C.PHASE_DURATION, gt=0, le=C.PHASE_DURATION_MAX)
    """Fixed-time green, s."""
    min_green: float | None = Field(default=None, ge=0, le=C.MIN_GREEN_MAX)
    """Minimum green, s; inherits the signal's min_green."""
    max_green: float | None = Field(default=None, gt=0, le=C.MAX_GREEN_MAX)
    """Maximum green, s; inherits the signal's max_green."""


class SignalSpec(Spec):
    """Traffic signal of a signalized intersection."""

    controller: ControllerSpec = Field(default_factory=ControllerSpec)
    """Controller type and parameters."""
    phases: tuple[PhaseSpec, ...] | None = Field(
        default=None, min_length=1, max_length=C.MAX_PHASES
    )
    """Phases in cycle order; derived from template when omitted."""
    template: SignalTemplate = "auto"
    """Phase template used when phases are omitted."""
    yellow: float = Field(default=C.SIGNAL_YELLOW, ge=0, le=C.SIGNAL_INTERGREEN_MAX)
    """Yellow interval, s (inserted automatically for movements losing green)."""
    all_red: float = Field(default=C.SIGNAL_ALL_RED, ge=0, le=C.SIGNAL_INTERGREEN_MAX)
    """All-red interval after yellow, s."""
    min_green: float = Field(default=C.DEFAULT_MIN_GREEN_S, ge=0, le=C.MIN_GREEN_MAX)
    """Minimum green, s."""
    max_green: float = Field(default=C.SIGNAL_MAX_GREEN, gt=0, le=C.MAX_GREEN_MAX)
    """Maximum green, s (> min_green)."""
    initial_phase: int = Field(default=0, ge=0)
    """Index of the phase active at t = 0."""

    @model_validator(mode="after")
    def _green_bounds(self) -> Self:
        if self.max_green <= self.min_green:
            raise _exclusive("green", a=self.max_green, b=self.min_green)
        return self


class IntersectionSpec(Spec):
    """A node of the road network."""

    id: Id
    """Intersection id."""
    point: Point
    """Centre [x, y], m."""
    kind: IntersectionKind | None = None
    """signalized, priority, uncontrolled or boundary; derived from topology."""
    radius: float | None = Field(default=None, ge=0, le=C.INTERSECTION_RADIUS_MAX)
    """Stop-line distance from point, m; derived: widest incident road + 2 m."""
    major_roads: tuple[Id, ...] | None = None
    """Incoming roads with right of way (priority kind); derived."""
    movements: tuple[MovementSpec, ...] | None = None
    """Movements; when present the list is authoritative (all or nothing)."""
    signal: SignalSpec | None = None
    """Signal (signalized only); derived for signalized intersections."""


class RoadSpec(Spec):
    """A one-way road from one intersection to another."""

    id: Id
    """Road id."""
    from_: Id = Field(alias="from")
    """Start intersection."""
    to: Id
    """End intersection."""
    points: tuple[Point, ...] | None = Field(
        default=None, min_length=2, max_length=C.MAX_ROAD_POINTS
    )
    """Median-edge polyline [[x, y], ...], m; derived: straight between the intersections."""
    lanes: tuple[LaneSpec, ...] = Field(min_length=1, max_length=C.MAX_LANES_PER_ROAD)
    """Lanes from the median outward; [{}, {}] is two default lanes."""
    speed_limit: Speed | None = None
    """Default lane speed limit, m/s; derived: network.speed_limit."""
    lane_width: LaneWidth | None = None
    """Default lane width, m; derived: network.lane_width."""
    name: str = Field(default="", max_length=C.MAX_STREET_NAME_LENGTH)
    """Street name for display."""


class NetworkSpec(Spec):
    """The road network."""

    drive_side: DriveSide = DriveSide.right
    """Side of the road vehicles drive on (right or left)."""
    lane_width: LaneWidth = C.LANE_WIDTH
    """Default lane width, m."""
    speed_limit: Speed = C.SPEED_LIMIT
    """Default speed limit, m/s."""
    allow_uturns: bool = False
    """Also derive U-turn movements."""
    intersections: tuple[IntersectionSpec, ...] = Field(
        min_length=1, max_length=C.MAX_INTERSECTIONS
    )
    """Intersections (nodes)."""
    roads: tuple[RoadSpec, ...] = Field(min_length=1, max_length=C.MAX_ROADS)
    """Roads (directed edges)."""


# --------------------------------------------------------------------------- vehicle types
class SpeedFactorSpec(Spec):
    """Per-vehicle speed factor f ~ N(mean, std) clipped to [min, max] (min <= mean <= max)."""

    mean: float = Field(default=C.SPEED_FACTOR_MEAN, gt=0)
    """Mean speed factor."""
    std: float = Field(default=C.SPEED_FACTOR_STD, ge=0)
    """Standard deviation."""
    min: float = Field(default=C.SPEED_FACTOR_MIN, gt=0)
    """Lower clip."""
    max: float = Field(default=C.SPEED_FACTOR_MAX, gt=0)
    """Upper clip."""


class VehicleTypeSpec(Spec):
    """A vehicle type; fields you set override the built-in type with the same id."""

    id: Id
    """Type id (car, bus, truck and emergency are built in)."""
    vclass: VehicleClass = VehicleClass.car
    """car, bus, truck or emergency."""
    length: float = Field(default=C.VEHICLE_LENGTH, gt=0, le=C.VEHICLE_LENGTH_MAX)
    """Length, m."""
    width: float = Field(default=C.VEHICLE_WIDTH, gt=0, le=C.VEHICLE_WIDTH_MAX)
    """Width, m."""
    max_speed: float = Field(default=C.VEHICLE_MAX_SPEED, gt=0, le=C.SPEED_LIMIT_MAX)
    """Maximum speed, m/s; effective v0 = min(max_speed, limit x speed_factor)."""
    accel: float = Field(default=C.IDM_ACCEL, gt=0, le=C.VEHICLE_ACCEL_MAX)
    """IDM maximum acceleration a, m/s^2."""
    decel: float = Field(default=C.IDM_DECEL, gt=0, le=C.VEHICLE_ACCEL_MAX)
    """IDM comfortable deceleration b, m/s^2."""
    emergency_decel: float = Field(default=C.IDM_EMERGENCY_DECEL, gt=0, le=C.EMERGENCY_DECEL_MAX)
    """Physical braking bound, m/s^2 (>= decel)."""
    min_gap: float = Field(default=C.IDM_MIN_GAP, ge=0, le=C.MIN_GAP_MAX)
    """Jam distance s0, m."""
    headway: float = Field(default=C.IDM_HEADWAY, gt=0, le=C.HEADWAY_MAX)
    """Desired time headway T, s."""
    speed_factor: SpeedFactorSpec = Field(default_factory=SpeedFactorSpec)
    """Speed factor distribution, sampled per vehicle from the spawning flow's stream."""
    politeness: float = Field(default=C.MOBIL_POLITENESS, ge=0, le=1)
    """MOBIL politeness p."""
    lc_threshold: float = Field(default=C.MOBIL_THRESHOLD, ge=0, le=C.LC_THRESHOLD_MAX)
    """MOBIL threshold a_th, m/s^2."""
    lc_safe_decel: float = Field(default=C.MOBIL_SAFE_DECEL, gt=0, le=C.LC_SAFE_DECEL_MAX)
    """MOBIL safe deceleration b_safe, m/s^2."""
    model_params: Mapping[str, float] = Field(default_factory=dict)
    """Model-specific extras such as delta (checked by the deep validation)."""
    color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    """#RRGGBB; null uses the per-class palette."""

    @field_validator("speed_factor")
    @classmethod
    def _ordered(cls, factor: SpeedFactorSpec, info: ValidationInfo) -> SpeedFactorSpec:
        # the E007 path is vehicle_types[i].speed_factor; `id` is validated before this field
        if _partial_override(info.data.get("id"), factor.model_fields_set, _FACTOR_FIELDS):
            return factor
        if not factor.min <= factor.mean <= factor.max:
            raise _exclusive("speed_factor")
        return factor

    @model_validator(mode="after")
    def _braking(self) -> Self:
        if _partial_override(self.id, self.model_fields_set, _BRAKING_FIELDS):
            return self
        if self.emergency_decel < self.decel:
            raise _exclusive("decel", e=self.emergency_decel, d=self.decel)
        return self


def _builtin_types() -> Mapping[str, VehicleTypeSpec]:
    return MappingProxyType(
        {
            name: VehicleTypeSpec.model_validate({"id": name, **overrides})
            for name, overrides in C.BUILTIN_VEHICLE_TYPES.items()
        }
    )


DEFAULT_VEHICLE_TYPES: Final[Mapping[str, VehicleTypeSpec]] = _builtin_types()
"""Built-in vehicle types (car, bus, truck, emergency) by id."""


# --------------------------------------------------------------------------- demand
class RouteChoiceSpec(Spec):
    """One entry of a flow's route distribution."""

    roads: tuple[Id, ...] = Field(min_length=1)
    """Fully connected road sequence."""
    weight: float = Field(gt=0)
    """Relative weight (normalised over the distribution)."""


class FlowSpec(Spec):
    """A stream of vehicles."""

    id: Id
    """Flow id; vehicles are "{flow}.{k}"."""
    vehicle_type: Id | None = None
    """Vehicle type; "car" when neither this nor type_mix is given."""
    type_mix: Mapping[Id, Annotated[float, Field(gt=0)]] | None = Field(default=None, min_length=1)
    """type id -> weight, sampled per vehicle (mutually exclusive with vehicle_type)."""
    route: tuple[Id, ...] | None = Field(default=None, min_length=1)
    """Fixed road sequence."""
    routes: tuple[RouteChoiceSpec, ...] | None = Field(
        default=None, min_length=1, max_length=C.MAX_ROUTE_CHOICES
    )
    """Route distribution; one entry is drawn per vehicle at spawn."""
    origin: Id | None = None
    """Origin road (the router resolves the path at departure)."""
    destination: Id | None = None
    """Destination road."""
    via: tuple[Id, ...] = ()
    """Ordered waypoint roads (origin/destination mode only)."""
    rate: float | None = Field(default=None, gt=0)
    """Vehicles per hour."""
    period: float | None = Field(default=None, gt=0)
    """Seconds between vehicles (instead of rate)."""
    arrival: Arrival = "uniform"
    """Arrival process: uniform, poisson or binomial."""
    begin: Seconds = 0.0
    """First possible departure, s."""
    end: float | None = None
    """End of the half-open window [begin, end), s; null = until the simulation ends."""
    count: int | None = Field(default=None, ge=1)
    """Maximum number of vehicles."""
    depart_lane: DepartLane = "best"
    """best, random, first or a lane index."""
    depart_speed: DepartSpeed = "max"
    """max (insertion-safe) or a speed in m/s."""

    @model_validator(mode="after")
    def _exclusive_fields(self) -> Self:
        if (self.rate is None) == (self.period is None):
            raise _exclusive("rate_period")
        od = self.origin is not None and self.destination is not None
        partial_od = (self.origin is None) != (self.destination is None)
        modes = (self.route is not None) + (self.routes is not None) + od
        if modes != 1 or partial_od:
            raise _exclusive("flow_route")
        if self.via and not od:
            raise _exclusive("via")
        if self.vehicle_type is not None and self.type_mix is not None:
            raise _exclusive("type_mix")
        _check_window(self.begin, self.end)
        return self


class TripSpec(Spec):
    """A single vehicle with a fixed departure time."""

    id: Id
    """Trip id (also the vehicle id)."""
    depart: Seconds
    """Departure time, s."""
    vehicle_type: Id = "car"
    """Vehicle type."""
    route: tuple[Id, ...] | None = Field(default=None, min_length=1)
    """Fixed road sequence."""
    origin: Id | None = None
    """Origin road."""
    destination: Id | None = None
    """Destination road."""
    via: tuple[Id, ...] = ()
    """Ordered waypoint roads (origin/destination mode only)."""
    depart_lane: DepartLane = "best"
    """best, random, first or a lane index."""
    depart_speed: DepartSpeed = "max"
    """max (insertion-safe) or a speed in m/s."""

    @model_validator(mode="after")
    def _exclusive_fields(self) -> Self:
        od = self.origin is not None and self.destination is not None
        partial_od = (self.origin is None) != (self.destination is None)
        if (self.route is not None) + od != 1 or partial_od:
            raise _exclusive("trip_route")
        if self.via and not od:
            raise _exclusive("via")
        return self


class StopSpec(Spec):
    """A transit stop on one of the line's roads."""

    id: Id | None = None
    """Stop id; derived: "{line}.{k}"."""
    road: Id
    """Road on the line's route."""
    position: Metres
    """Where the bus front stops, m along the trimmed lane."""
    lane: int | None = Field(default=None, ge=0)
    """Lane index; derived: the curb lane (n-1)."""
    dwell: Seconds = C.STOP_DWELL
    """Dwell time, s."""


class TransitLineSpec(Spec):
    """A bus line with scheduled departures and stops."""

    id: Id
    """Line id; vehicles are "{line}.{k}"."""
    vehicle_type: Id = "bus"
    """Vehicle type."""
    route: tuple[Id, ...] = Field(min_length=1)
    """Road sequence (always explicit)."""
    stops: tuple[StopSpec, ...] = Field(min_length=1)
    """Stops in route order."""
    headway: float | None = Field(default=None, gt=0)
    """Seconds between departures."""
    departures: tuple[Seconds, ...] | None = Field(default=None, min_length=1)
    """Explicit departure times, s (sorted)."""
    begin: Seconds = 0.0
    """First departure, s."""
    end: float | None = None
    """No departures at or after this time, s."""
    depart_lane: DepartLane = "best"
    """best, random, first or a lane index."""

    @model_validator(mode="after")
    def _exclusive_fields(self) -> Self:
        if (self.headway is None) == (self.departures is None):
            raise _exclusive("headway")
        if self.departures is not None and list(self.departures) != sorted(self.departures):
            raise _exclusive("departures")
        _check_window(self.begin, self.end)
        return self


class DemandSpec(Spec):
    """Traffic demand. Flow, trip and transit ids share one namespace."""

    flows: tuple[FlowSpec, ...] = ()
    """Vehicle flows."""
    trips: tuple[TripSpec, ...] = ()
    """Individual trips."""
    transit: tuple[TransitLineSpec, ...] = ()
    """Transit lines."""


# --------------------------------------------------------------------------- root
class ScenarioSpec(Spec):
    """An UrbanFlow scenario (format urbanflow.scenario, version 1.0)."""

    schema_: str | None = Field(default=None, alias="$schema", exclude_if=lambda v: v is None)
    """Ignored; an editor hint only."""
    format: Literal["urbanflow.scenario"]
    """Always "urbanflow.scenario"."""
    version: str = Field(pattern=r"^\d+\.\d+$")
    """Format version "MAJOR.MINOR"."""
    meta: MetaSpec
    """Name, description, tags and provenance."""
    simulation: SimulationSpec = Field(default_factory=SimulationSpec)
    """Scenario-level simulation defaults."""
    network: NetworkSpec
    """Road network."""
    vehicle_types: tuple[VehicleTypeSpec, ...] = Field(default=(), max_length=C.MAX_VEHICLE_TYPES)
    """Vehicle types, merged over the built-ins by id."""
    demand: DemandSpec = Field(default_factory=DemandSpec)
    """Flows, trips and transit lines."""


def scenario_json_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of the scenario file format.

    Cross-references (road ids, movement ids...) cannot be expressed in JSON Schema, so
    semantic validation always runs in :mod:`urbanflow.scenario.validate` as well.
    """
    schema = ScenarioSpec.model_json_schema(by_alias=True, mode="validation")
    schema.pop("title", None)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "UrbanFlow scenario 1.0",
        **schema,
    }
