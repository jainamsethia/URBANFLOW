"""Convert CityFlow road networks and flows into an UrbanFlow scenario (plan E.10 §4.2).

Input files are parsed as data only (minimal pydantic models; unknown keys ignored), and
input errors carry input-rooted paths such as ``roadnet.intersections[3].trafficLight``.
The mapping is lossy in documented ways; every loss is reported as a warning:

* ``virtual`` intersections become boundaries; others become ``signalized`` when their
  traffic light has more than one phase, else ``uncontrolled``; ``width`` is the radius.
* roads, lanes (lane 0 = innermost in both) and lane links map one to one; CityFlow's full
  lane fan-out is kept unless ``derive_connections`` (then UrbanFlow derives the mapping).
* phases no longer than ``yellow_threshold`` whose links are all right turns are CityFlow's
  intergreens: they are dropped and UrbanFlow inserts its own (yellow = their shortest
  time, no all-red); ``keep_all_phases`` keeps them as ordinary phases. Left turns are
  permissive (g) in phases that also release a straight from another road, right turns
  when they also release a straight or left from another road; everything else is G.
* vehicle parameters become types ``cf_type_{k}`` (``maxPosAcc`` -> accel, ``usualNegAcc``
  -> decel, ``maxNegAcc`` -> emergency_decel, no speed spread); CityFlow's car-following
  model differs, so trajectories differ.
* a flow's route is used as is when consecutive roads connect, else as
  origin / via / destination anchors; ``interval`` is the period and the inclusive
  ``endTime`` becomes ``end = endTime + dt/2`` (-1: until the simulation ends).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from urbanflow._version import __version__
from urbanflow.core import constants as C
from urbanflow.core.errors import NotFoundError, ScenarioValidationError, Severity, ValidationIssue
from urbanflow.scenario.io import read_text
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.validate import issues_from_pydantic

__all__ = ["CityFlowOptions", "ImportResult", "convert", "load"]

_TURNS = {"go_straight": "straight", "turn_left": "left", "turn_right": "right"}
_BAD_ID = re.compile(r"[^A-Za-z0-9_.:\-]")


class _Cf(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class CfPoint(_Cf):
    x: float
    y: float


class CfLane(_Cf):
    width: float = Field(gt=0)
    maxSpeed: float = Field(gt=0)


class CfRoad(_Cf):
    id: str = Field(min_length=1)
    points: list[CfPoint] = Field(min_length=2)
    lanes: list[CfLane] = Field(min_length=1)
    startIntersection: str
    endIntersection: str


class CfLaneLink(_Cf):
    startLaneIndex: int = Field(ge=0)
    endLaneIndex: int = Field(ge=0)
    points: list[CfPoint] = Field(default_factory=list)


class CfRoadLink(_Cf):
    type: str
    startRoad: str
    endRoad: str
    laneLinks: list[CfLaneLink] = Field(default_factory=list)


class CfPhase(_Cf):
    time: float = Field(ge=0)
    availableRoadLinks: list[int] = Field(default_factory=list)


class CfLight(_Cf):
    lightphases: list[CfPhase] = Field(default_factory=list)


class CfIntersection(_Cf):
    id: str = Field(min_length=1)
    point: CfPoint
    width: float = Field(default=0.0, ge=0)
    roadLinks: list[CfRoadLink] = Field(default_factory=list)
    trafficLight: CfLight | None = None
    virtual: bool = False


class CfRoadnet(_Cf):
    intersections: list[CfIntersection] = Field(min_length=1)
    roads: list[CfRoad] = Field(min_length=1)


class CfVehicle(_Cf):
    length: float = Field(gt=0)
    width: float = Field(gt=0)
    maxPosAcc: float = Field(gt=0)
    maxNegAcc: float = Field(gt=0)
    usualNegAcc: float = Field(gt=0)
    minGap: float = Field(ge=0)
    maxSpeed: float = Field(gt=0)
    headwayTime: float = Field(gt=0)


class CfFlow(_Cf):
    vehicle: CfVehicle
    route: list[str] = Field(min_length=1)
    interval: float = Field(gt=0)
    startTime: float = Field(default=0.0, ge=0)
    endTime: float = -1.0


class CfConfig(_Cf):
    interval: float = Field(default=1.0, gt=0)
    seed: int = Field(default=0, ge=0)
    dir: str = ""
    roadnetFile: str = ""
    flowFile: str = ""
    rlTrafficLight: bool = False
    laneChange: bool = False


class CityFlowOptions(BaseModel):
    """Conversion options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "cityflow_import"
    """Scenario name."""
    duration: float | None = Field(default=C.DURATION, gt=0)
    """Simulated duration, s (CityFlow configs have none)."""
    derive_connections: bool = False
    """Replace CityFlow's lane fan-out by UrbanFlow's lane mapping."""
    keep_all_phases: bool = False
    """Keep short all-right-turn phases instead of treating them as intergreens."""
    yellow_threshold: float = Field(default=5.0, ge=0)
    """Longest phase, s, that can be an intergreen."""


@dataclass(frozen=True, slots=True)
class ImportResult:
    """The converted scenario and every conversion warning (lossy points)."""

    scenario: Scenario
    warnings: tuple[ValidationIssue, ...]


class _Ids:
    """Valid, unique UrbanFlow ids for arbitrary CityFlow ids."""

    def __init__(self) -> None:
        self.map: dict[str, str] = {}
        self._used: set[str] = set()
        self.renamed = 0

    def __call__(self, raw: str) -> str:
        if raw not in self.map:
            base = _BAD_ID.sub("_", raw)[: C.MAX_ID_LENGTH - 8] or "id"
            out, n = base, 1
            while out in self._used:
                out, n = f"{base}_{n}", n + 1
            self.renamed += out != raw
            self.map[raw] = out
            self._used.add(out)
        return self.map[raw]


def _fail(path: str, message: str) -> ScenarioValidationError:
    return ScenarioValidationError([ValidationIssue(path, message, "E700")], "CityFlow import")


def _clamp(value: float, lo: float, hi: float, what: str, warn: Any) -> float:
    if lo <= value <= hi:
        return value
    out = min(max(value, lo), hi)
    warn(what, f"{value:g} is outside [{lo:g}, {hi:g}]; using {out:g}")
    return out


def _parse[M: BaseModel](model: type[M], data: object, prefix: tuple[str | int, ...]) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        issues = issues_from_pydantic(exc, prefix=prefix, root=model)
        raise ScenarioValidationError(issues, "CityFlow import") from None


def convert(
    roadnet: Mapping[str, Any],
    flows: Sequence[Any],
    config: Mapping[str, Any] | None = None,
    options: CityFlowOptions | None = None,
) -> ImportResult:
    """An UrbanFlow scenario from parsed CityFlow ``roadnet``, ``flows`` and ``config``."""
    opts = options or CityFlowOptions()
    net = _parse(CfRoadnet, roadnet, ("roadnet",))
    cfg = _parse(CfConfig, config or {}, ("config",))
    cf_flows = [_parse(CfFlow, f, ("flow", i)) for i, f in enumerate(flows)]
    warnings: list[ValidationIssue] = []

    def warn(path: str, message: str) -> None:
        warnings.append(ValidationIssue(path, message, "W700", Severity.warning))

    for key in ("saveReplay", "roadnetLogFile", "replayLogFile"):
        if config and key in config:
            warn(f"config.{key}", "ignored (UrbanFlow records replays with --record)")

    road_ids, ix_ids = _Ids(), _Ids()
    roads_by_cf = {r.id: r for r in net.roads}
    ix_by_cf = {ix.id: ix for ix in net.intersections}
    roads: list[dict[str, Any]] = []
    for i, r in enumerate(net.roads):
        for end, ref in (
            ("startIntersection", r.startIntersection),
            ("endIntersection", r.endIntersection),
        ):
            if ref not in ix_by_cf:
                raise _fail(f"roadnet.roads[{i}].{end}", f'unknown intersection "{ref}"')
        points = [[p.x, p.y] for p in r.points]
        points = [p for k, p in enumerate(points) if k == 0 or p != points[k - 1]]
        if len(points) < len(r.points):
            warn(f"roadnet.roads[{i}].points", "repeated points removed")
        lanes = [
            {
                "width": _clamp(
                    ln.width,
                    C.LANE_WIDTH_MIN,
                    C.LANE_WIDTH_MAX,
                    f"roadnet.roads[{i}].lanes[{k}].width",
                    warn,
                ),
                "speed_limit": _clamp(
                    ln.maxSpeed,
                    C.SPEED_LIMIT_MIN,
                    C.SPEED_LIMIT_MAX,
                    f"roadnet.roads[{i}].lanes[{k}].maxSpeed",
                    warn,
                ),
            }
            for k, ln in enumerate(r.lanes)
        ]
        roads.append(
            {
                "id": road_ids(r.id),
                "from": ix_ids(r.startIntersection),
                "to": ix_ids(r.endIntersection),
                "points": points,
                "lanes": lanes,
            }
        )

    connected: set[tuple[str, str]] = set()
    intersections: list[dict[str, Any]] = []
    controller = "external" if cfg.rlTrafficLight else "fixed_time"
    for i, ix in enumerate(net.intersections):
        where = f"roadnet.intersections[{i}]"
        out: dict[str, Any] = {
            "id": ix_ids(ix.id),
            "point": [ix.point.x, ix.point.y],
            "radius": 0.0
            if ix.virtual
            else _clamp(ix.width, 0.0, C.INTERSECTION_RADIUS_MAX, f"{where}.width", warn),
        }
        intersections.append(out)
        if ix.virtual:
            out["kind"] = "boundary"
            if ix.roadLinks:
                warn(f"{where}.roadLinks", "a virtual intersection's road links are ignored")
            continue
        movements: dict[tuple[str, str], dict[str, Any]] = {}
        link_movement: list[tuple[str, str]] = []
        for k, link in enumerate(ix.roadLinks):
            lp = f"{where}.roadLinks[{k}]"
            if link.type not in _TURNS:
                raise _fail(
                    f"{lp}.type",
                    f'unknown road link type "{link.type}" (expected one of: {", ".join(_TURNS)})',
                )
            for key, ref in (("startRoad", link.startRoad), ("endRoad", link.endRoad)):
                if ref not in roads_by_cf:
                    raise _fail(f"{lp}.{key}", f'unknown road "{ref}"')
            pair = (link.startRoad, link.endRoad)
            link_movement.append(pair)
            connected.add(pair)
            mov = movements.get(pair)
            if mov is None:
                mov = movements[pair] = {
                    "id": f"{road_ids(pair[0])}->{road_ids(pair[1])}",
                    "from_road": road_ids(pair[0]),
                    "to_road": road_ids(pair[1]),
                    "turn": _TURNS[link.type],
                    "connections": [],
                }
            else:
                warn(lp, f"duplicate road link {pair[0]} -> {pair[1]} merged")
            n_in = len(roads_by_cf[link.startRoad].lanes)
            n_out = len(roads_by_cf[link.endRoad].lanes)
            seen = {(c["from_lane"], c["to_lane"]) for c in mov["connections"]}
            for c, ll in enumerate(link.laneLinks):
                if ll.startLaneIndex >= n_in or ll.endLaneIndex >= n_out:
                    raise _fail(
                        f"{lp}.laneLinks[{c}]",
                        f"lane index out of range ({n_in} lanes in, {n_out} out)",
                    )
                key2 = (ll.startLaneIndex, ll.endLaneIndex)
                if key2 in seen:
                    continue
                seen.add(key2)
                conn: dict[str, Any] = {"from_lane": key2[0], "to_lane": key2[1]}
                shape = [[p.x, p.y] for p in ll.points]
                shape = [p for j, p in enumerate(shape) if j == 0 or p != shape[j - 1]]
                if len(shape) >= 2:
                    conn["shape"] = shape
                mov["connections"].append(conn)
        fan_out = any(
            len({c["from_lane"] for c in m["connections"]}) < len(m["connections"])
            for m in movements.values()
        )
        for mov in movements.values():
            if opts.derive_connections or not mov["connections"]:
                del mov["connections"]
        if fan_out and not opts.derive_connections:
            warn(
                f"{where}.roadLinks",
                "lane fan-out kept (one lane links to several "
                "lanes); --derive-connections uses UrbanFlow's lane mapping",
            )
        if movements:
            out["movements"] = list(movements.values())
        else:
            warn(where, "no road links; movements are derived")
        phases_in = ix.trafficLight.lightphases if ix.trafficLight else []
        signal = _signal(phases_in, ix.roadLinks, link_movement, movements, opts, where, warn)
        if signal is None:
            out["kind"] = "uncontrolled"
        else:
            out["kind"] = "signalized"
            signal["controller"] = {"type": controller}
            out["signal"] = signal

    types: dict[tuple[float, ...], str] = {}
    vehicle_types: list[dict[str, Any]] = []
    out_flows: list[dict[str, Any]] = []
    dt = _clamp(cfg.interval, C.DT_MIN, C.DT_MAX, "config.interval", warn)
    for i, f in enumerate(cf_flows):
        where = f"flow[{i}]"
        v = f.vehicle
        vt_key = (
            v.length,
            v.width,
            v.maxPosAcc,
            v.usualNegAcc,
            v.maxNegAcc,
            v.minGap,
            v.maxSpeed,
            v.headwayTime,
        )
        if vt_key not in types:
            types[vt_key] = f"cf_type_{len(types)}"
            decel = _clamp(
                v.usualNegAcc, 0.1, C.VEHICLE_ACCEL_MAX, f"{where}.vehicle.usualNegAcc", warn
            )
            vehicle_types.append(
                {
                    "id": types[vt_key],
                    "length": _clamp(
                        v.length, 0.5, C.VEHICLE_LENGTH_MAX, f"{where}.vehicle.length", warn
                    ),
                    "width": _clamp(
                        v.width, 0.5, C.VEHICLE_WIDTH_MAX, f"{where}.vehicle.width", warn
                    ),
                    "max_speed": _clamp(
                        v.maxSpeed, 0.5, C.SPEED_LIMIT_MAX, f"{where}.vehicle.maxSpeed", warn
                    ),
                    "accel": _clamp(
                        v.maxPosAcc, 0.1, C.VEHICLE_ACCEL_MAX, f"{where}.vehicle.maxPosAcc", warn
                    ),
                    "decel": decel,
                    "emergency_decel": _clamp(
                        max(v.maxNegAcc, decel),
                        decel,
                        C.EMERGENCY_DECEL_MAX,
                        f"{where}.vehicle.maxNegAcc",
                        warn,
                    ),
                    "min_gap": _clamp(
                        v.minGap, 0.0, C.MIN_GAP_MAX, f"{where}.vehicle.minGap", warn
                    ),
                    "headway": _clamp(
                        v.headwayTime, 0.1, C.HEADWAY_MAX, f"{where}.vehicle.headwayTime", warn
                    ),
                    "speed_factor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                }
            )
        for k, ref in enumerate(f.route):
            if ref not in roads_by_cf:
                raise _fail(f"{where}.route[{k}]", f'unknown road "{ref}"')
        route = [road_ids(r) for r in f.route]
        flow: dict[str, Any] = {
            "id": f"flow_{i}",
            "vehicle_type": types[vt_key],
            "period": f.interval,
            "arrival": "uniform",
            "begin": f.startTime,
        }
        if f.endTime >= 0:
            if f.endTime < f.startTime:
                warn(where, "endTime is before startTime; the flow is dropped")
                continue
            flow["end"] = f.endTime + dt / 2
        if all((a, b) in connected for a, b in zip(f.route, f.route[1:], strict=False)):
            flow["route"] = route
        else:
            flow |= {"origin": route[0], "destination": route[-1], "via": route[1:-1]}
        out_flows.append(flow)

    if road_ids.renamed or ix_ids.renamed:
        warn(
            "roadnet",
            f"{road_ids.renamed + ix_ids.renamed} ids had characters UrbanFlow "
            "does not allow and were renamed",
        )
    data = {
        "format": "urbanflow.scenario",
        "version": "1.0",
        "meta": {
            "name": opts.name,
            "description": f"Imported from CityFlow ({len(net.intersections)} intersections, "
            f"{len(net.roads)} roads, {len(cf_flows)} flows).",
            "generator": {
                "name": "cityflow",
                "params": opts.model_dump(mode="json"),
                "urbanflow_version": __version__,
            },
        },
        "simulation": {
            "dt": dt,
            "duration": opts.duration,
            "seed": cfg.seed,
            "lane_changing": cfg.laneChange,
        },
        "network": {"intersections": intersections, "roads": roads},
        "vehicle_types": vehicle_types,
        "demand": {"flows": out_flows},
    }
    scenario = Scenario.from_dict(data, source="CityFlow import")
    return ImportResult(scenario, (*warnings, *scenario.issues))


def _signal(
    phases: Sequence[CfPhase],
    links: Sequence[CfRoadLink],
    link_movement: Sequence[tuple[str, str]],
    movements: Mapping[tuple[str, str], Mapping[str, Any]],
    opts: CityFlowOptions,
    where: str,
    warn: Any,
) -> dict[str, Any] | None:
    """The signal of an intersection, or None when it has at most one phase."""
    if len(phases) <= 1:
        return None
    kept: list[dict[str, Any]] = []
    intergreens: list[float] = []
    for p, ph in enumerate(phases):
        pp = f"{where}.trafficLight.lightphases[{p}]"
        for k, idx in enumerate(ph.availableRoadLinks):
            if not 0 <= idx < len(links):
                raise _fail(
                    f"{pp}.availableRoadLinks[{k}]",
                    f"phase references road link "
                    f"index {idx} but the intersection has {len(links)} road links",
                )
        released = [links[idx] for idx in ph.availableRoadLinks]
        rights_only = all(link.type == "turn_right" for link in released)
        if not opts.keep_all_phases and ph.time <= opts.yellow_threshold and rights_only:
            intergreens.append(ph.time)
            continue
        if not released or ph.time <= 0:
            warn(pp, "empty or zero-length phase dropped")
            continue
        straight = {link.startRoad for link in released if link.type == "go_straight"}
        left = {link.startRoad for link in released if link.type == "turn_left"}
        green: dict[str, str] = {}
        for idx in ph.availableRoadLinks:
            link = links[idx]
            road = link.startRoad
            if link.type == "turn_left":
                state = "g" if straight - {road} else "G"
            elif link.type == "turn_right":
                state = "g" if (straight | left) - {road} else "G"
            else:
                state = "G"
            mov_id = movements[link_movement[idx]]["id"]
            if green.get(mov_id) != "G":
                green[mov_id] = state
        kept.append({"id": f"p{p}", "green": green, "duration": min(ph.time, C.PHASE_DURATION_MAX)})
    if len(kept) <= 1:
        warn(where, "fewer than two phases after removing intergreens; treated as uncontrolled")
        return None
    shortest = min(ph["duration"] for ph in kept)
    signal: dict[str, Any] = {
        "phases": kept,
        "min_green": min(C.DEFAULT_MIN_GREEN_S, shortest),
    }
    if intergreens:
        signal |= {"yellow": min(min(intergreens), C.SIGNAL_INTERGREEN_MAX), "all_red": 0.0}
    return signal


def _read(path: Path, what: str) -> Any:
    text = read_text(path)
    try:
        return json.loads(text, parse_constant=lambda _c: math.nan)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise _fail(what, f"{path}: invalid JSON: {exc}") from None


def _locate(base: Path, directory: str, name: str) -> Path:
    """CityFlow resolves ``dir + file`` from its working directory; try that, then the
    config's own directory."""
    for root in (Path.cwd(), base):
        candidate = root / directory / name
        if candidate.is_file():
            return candidate
    raise NotFoundError(f'CityFlow file "{directory}{name}" not found from {Path.cwd()} or {base}')


def load(
    *,
    config: str | Path | None = None,
    roadnet: str | Path | None = None,
    flow: str | Path | None = None,
    options: CityFlowOptions | None = None,
) -> ImportResult:
    """Read CityFlow files (``config`` locates the others unless given) and convert them."""
    cfg: dict[str, Any] = {}
    if config is not None:
        cfg_path = Path(config)
        data = _read(cfg_path, "config")
        if not isinstance(data, dict):
            raise _fail("config", "expected a JSON object")
        cfg = data
        parsed = _parse(CfConfig, cfg, ("config",))
        if roadnet is None and parsed.roadnetFile:
            roadnet = _locate(cfg_path.parent, parsed.dir, parsed.roadnetFile)
        if flow is None and parsed.flowFile:
            flow = _locate(cfg_path.parent, parsed.dir, parsed.flowFile)
    if roadnet is None or flow is None:
        raise _fail("$", "a roadnet and a flow file are required (directly or via the config)")
    net_data, flow_data = _read(Path(roadnet), "roadnet"), _read(Path(flow), "flow")
    if not isinstance(flow_data, list):
        raise _fail("flow", "expected a JSON array of flows")
    return convert(net_data, flow_data, cfg, options)
