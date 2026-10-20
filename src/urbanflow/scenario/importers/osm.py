"""Convert an OpenStreetMap extract (``.osm`` XML) into an UrbanFlow scenario.

The file is parsed as data only: documents with a DTD or entity declarations are refused
(no entity expansion), and only nodes and ways are read. Steps:

1. **Roads.** Ways tagged ``highway=`` with a drivable class are kept (``service`` roads
   only with ``include_service``; ``access=no|private`` roads are dropped). Positions use an
   equirectangular projection around the extract's centre (metres; fine at city scale).
2. **Junctions.** Way ends and nodes shared by ways become junction nodes, except a node
   that only joins two ways with the same direction, lanes and speed (a tag split, not a
   junction). Junction nodes closer than ``cluster_radius`` merge into one junction, and so
   do the two ends of any road too short to hold a lane between its junction boxes; this
   collapses dual carriageways and complex crossings into one simulated junction each.
3. **Lanes, direction, speed.** ``lanes``, ``lanes:forward`` / ``lanes:backward``,
   ``oneway`` (``yes``/``-1``, motorways and roundabouts) and ``maxspeed`` (km/h, mph,
   ``walk``, country-zone codes) with per-class defaults. One road per direction and
   junction pair is kept (the one with most lanes).
4. **Control.** Junctions with a ``highway=traffic_signals`` node within ``signal_radius``
   and at least three neighbours are signalised (UrbanFlow derives the phases); a junction
   with one neighbour is a boundary (traffic enters and leaves there); others are priority
   junctions (major roads derived from lanes and speed) or simple joints.
5. **Traffic.** Each road entering from a boundary gets a Poisson flow sized by its class
   and lanes (x ``demand``), split over up to ``exits_per_entry`` boundary exits it can
   actually reach, picked with a seeded random choice weighted by exit size.

Roads that still cannot be simulated (e.g. degenerate geometry) are dropped and reported.
Everything lossy is a warning in the result.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from urbanflow._version import __version__
from urbanflow.core import constants as C
from urbanflow.core.errors import ScenarioValidationError, Severity, ValidationIssue
from urbanflow.scenario.derive import RoadGraph
from urbanflow.scenario.importers.cityflow import ImportResult
from urbanflow.scenario.io import read_text
from urbanflow.scenario.scenario import Scenario

__all__ = ["OsmOptions", "convert", "load"]

EARTH_RADIUS_M = 6_371_008.8
# class -> (default speed km/h, default lanes per direction, entry demand veh/h per lane)
CLASSES: dict[str, tuple[float, int, float]] = {
    "motorway": (110, 2, 600), "trunk": (90, 2, 450), "primary": (60, 1, 300),
    "secondary": (50, 1, 220), "tertiary": (50, 1, 150), "unclassified": (40, 1, 80),
    "residential": (30, 1, 40), "living_street": (10, 1, 10), "road": (40, 1, 40),
    "service": (20, 1, 10),
}  # fmt: skip
MIN_FLOW_VPH = 1.0  # veh/h; smaller generated flows are left out
JUNCTION_CLEARANCE = 12.0  # m; road shape points this close to a junction centre are dropped
ZONE_SPEEDS = {"urban": 50, "rural": 90, "motorway": 130, "living_street": 10, "walk": 5}


class OsmOptions(BaseModel):
    """Conversion options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "osm_import"
    """Scenario name."""
    duration: float = Field(default=C.DURATION, gt=0)
    """Simulated duration, s."""
    demand: float = Field(default=1.0, gt=0, le=5)
    """Multiplier of the generated traffic."""
    exits_per_entry: int = Field(default=4, ge=1, le=20)
    """Destinations per entry road."""
    cluster_radius: float = Field(default=25.0, ge=0, le=100)
    """Junction nodes closer than this merge into one junction, m."""
    signal_radius: float = Field(default=40.0, ge=0, le=150)
    """A traffic-signal node this close to a junction signalises it, m."""
    include_service: bool = False
    """Keep ``highway=service`` roads (parking aisles, driveways)."""
    seed: int = Field(default=0, ge=0)
    """Seed of the destination choice."""


@dataclass
class _Way:
    id: str
    nodes: list[str]
    kind: str
    oneway: int  # 0 two-way, 1 forward only, -1 backward only
    lanes_fwd: int
    lanes_bwd: int
    speed: float  # m/s


@dataclass
class _Edge:
    way: str
    a: str  # junction node ids (OSM)
    b: str
    points: list[tuple[float, float]]
    kind: str
    oneway: int
    lanes_fwd: int
    lanes_bwd: int
    speed: float


def _speed(tags: dict[str, str], kind: str) -> float:
    raw = tags.get("maxspeed", "").strip().lower()
    kmh = CLASSES.get(kind, CLASSES["road"])[0]
    if m := re.match(r"^(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?$", raw):
        kmh = float(m.group(1)) * (1.609344 if m.group(2) == "mph" else 1.0)
    elif raw in ZONE_SPEEDS:
        kmh = ZONE_SPEEDS[raw]
    elif ":" in raw and raw.split(":", 1)[1] in ZONE_SPEEDS:
        kmh = ZONE_SPEEDS[raw.split(":", 1)[1]]
    elif raw == "none":
        kmh = 130
    return min(max(kmh / 3.6, C.SPEED_LIMIT_MIN), C.SPEED_LIMIT_MAX)


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    m = re.match(r"^\s*(\d+)", value)
    return int(m.group(1)) if m else None


def _way(elem: ET.Element, include_service: bool) -> _Way | None:
    tags = {t.get("k", ""): t.get("v", "") for t in elem.findall("tag")}
    highway = tags.get("highway", "")
    kind = highway.removesuffix("_link")
    if kind not in CLASSES or (kind == "service" and not include_service):
        return None
    if tags.get("area") == "yes" or tags.get("access") in {"no", "private"}:
        return None
    nodes = [nd.get("ref", "") for nd in elem.findall("nd")]
    if len(nodes) < 2:
        return None
    ow = tags.get("oneway", "").lower()
    oneway = 1 if ow in {"yes", "true", "1"} else -1 if ow == "-1" else 0
    if not ow and (kind == "motorway" or tags.get("junction") in {"roundabout", "circular"}):
        oneway = 1
    if ow in {"no", "false", "0", "reversible", "alternating"}:
        oneway = 0
    default = CLASSES[kind][1]
    total = _int(tags.get("lanes"))
    fwd, bwd = _int(tags.get("lanes:forward")), _int(tags.get("lanes:backward"))
    if oneway:
        n = total or default
        fwd = bwd = n
    else:
        if fwd is None and bwd is None and total:
            fwd = max(1, total // 2)
            bwd = max(1, total - fwd)
        fwd = fwd or (default if total is None else max(1, (total or 2) - (bwd or 0)))
        bwd = bwd or (default if total is None else max(1, (total or 2) - fwd))
    return _Way(
        elem.get("id", ""), nodes, kind, oneway, _lanes(fwd), _lanes(bwd), _speed(tags, kind)
    )


def _lanes(n: int) -> int:
    return min(max(int(n), 1), 4)


class _Clusters:
    """Union-find over junction node ids."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        lo, hi = sorted((ra, rb), key=lambda s: (len(s), s))
        self.parent[hi] = lo
        return True


def _length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(p, q) for p, q in pairwise(points))


def _parse(text: str) -> ET.Element:
    head = text[:4096].lower()
    if "<!doctype" in head or "<!entity" in text.lower():
        raise ScenarioValidationError(
            [ValidationIssue("osm", "DTDs and entity declarations are not allowed", "E700")],
            "OSM import",
        )
    try:
        return ET.fromstring(text)  # noqa: S314 (no DTD/entities; checked above)
    except ET.ParseError as exc:
        raise ScenarioValidationError(
            [ValidationIssue("osm", f"not valid OSM XML: {exc}", "E700")], "OSM import"
        ) from None


def convert(text: str, options: OsmOptions | None = None) -> ImportResult:
    """An UrbanFlow scenario from the text of an ``.osm`` file."""
    opts = options or OsmOptions()
    root = _parse(text)
    warnings: list[ValidationIssue] = []

    def warn(path: str, message: str) -> None:
        warnings.append(ValidationIssue(path, message, "W700", Severity.warning))

    coords: dict[str, tuple[float, float]] = {}
    signals: list[str] = []
    for nd in root.iter("node"):
        nid = nd.get("id", "")
        try:
            coords[nid] = (float(nd.get("lat", "nan")), float(nd.get("lon", "nan")))
        except ValueError:
            continue
        if any(
            t.get("k") == "highway" and t.get("v") == "traffic_signals" for t in nd.findall("tag")
        ):
            signals.append(nid)
    ways = [w for e in root.iter("way") if (w := _way(e, opts.include_service)) is not None]
    ways = [
        w for w in ways if all(n in coords and all(map(math.isfinite, coords[n])) for n in w.nodes)
    ]
    if not ways:
        raise ScenarioValidationError(
            [ValidationIssue("osm", "no drivable roads (highway=...) found in this file", "E700")],
            "OSM import",
        )

    # projection around the centre of the used nodes
    used = {n for w in ways for n in w.nodes} | {s for s in signals if s in coords}
    lat0 = float(np.mean([coords[n][0] for n in used]))
    lon0 = float(np.mean([coords[n][1] for n in used]))
    kx = EARTH_RADIUS_M * math.cos(math.radians(lat0)) * math.pi / 180
    ky = EARTH_RADIUS_M * math.pi / 180
    xy = {n: ((coords[n][1] - lon0) * kx, (coords[n][0] - lat0) * ky) for n in used}

    # junction nodes: way ends and shared nodes, except plain joints between similar ways
    uses: dict[str, list[tuple[_Way, int]]] = defaultdict(list)
    for w in ways:
        for i, n in enumerate(w.nodes):
            uses[n].append((w, i))

    def similar(a: _Way, b: _Way) -> bool:
        return (a.oneway, a.lanes_fwd, a.lanes_bwd, a.speed) == (
            b.oneway,
            b.lanes_fwd,
            b.lanes_bwd,
            b.speed,
        )

    junction: set[str] = set()
    for n, refs in uses.items():
        if len(refs) == 1:
            w, i = refs[0]
            if i in (0, len(w.nodes) - 1):
                junction.add(n)  # a dead end or the edge of the extract
        elif len(refs) == 2:
            (w1, i1), (w2, i2) = refs
            end_to_start = (i1 == len(w1.nodes) - 1 and i2 == 0) or (
                i2 == len(w2.nodes) - 1 and i1 == 0
            )
            if w1 is w2 or not end_to_start or not similar(w1, w2):
                junction.add(n)
        else:
            junction.add(n)

    # chain ways into edges between junction nodes (joints at similar-way ends are passed)
    edges: list[_Edge] = []
    for w in ways:
        start = 0
        for i in range(1, len(w.nodes)):
            if w.nodes[i] in junction or i == len(w.nodes) - 1:
                pts = [xy[n] for n in w.nodes[start : i + 1]]
                edges.append(
                    _Edge(
                        w.id,
                        w.nodes[start],
                        w.nodes[i],
                        pts,
                        w.kind,
                        w.oneway,
                        w.lanes_fwd,
                        w.lanes_bwd,
                        w.speed,
                    )
                )
                start = i
    edges = [e for e in _join_joints(edges, junction) if e.a in junction and e.b in junction]

    # merge close junction nodes, then the ends of roads too short to simulate
    uf = _Clusters()
    jlist = sorted(junction)
    jxy = np.array([xy[n] for n in jlist]) if jlist else np.zeros((0, 2))
    if opts.cluster_radius > 0 and len(jlist) > 1:
        order = np.argsort(jxy[:, 0])
        for k, i in enumerate(order):
            for j in order[k + 1 :]:
                if jxy[j, 0] - jxy[i, 0] > opts.cluster_radius:
                    break
                if math.dist(jxy[i], jxy[j]) <= opts.cluster_radius:
                    uf.union(jlist[i], jlist[j])
    for _ in range(50):
        neighbours: dict[str, set[str]] = defaultdict(set)
        width: dict[str, float] = defaultdict(float)
        for e in edges:
            ca, cb = uf.find(e.a), uf.find(e.b)
            if ca == cb:
                continue
            neighbours[ca].add(cb)
            neighbours[cb].add(ca)
            w_lanes = max(e.lanes_fwd, e.lanes_bwd) * C.LANE_WIDTH + C.SETBACK
            width[ca] = max(width[ca], w_lanes)
            width[cb] = max(width[cb], w_lanes)
        radius = {c: 0.0 if len(nb) == 1 else width[c] for c, nb in neighbours.items()}
        changed = False
        for e in edges:
            ca, cb = uf.find(e.a), uf.find(e.b)
            if (
                ca != cb
                and _length(e.points)
                < radius.get(ca, 0) + radius.get(cb, 0) + C.MIN_LANE_LENGTH + 2
            ):
                changed |= uf.union(ca, cb)
        if not changed:
            break

    members: dict[str, list[str]] = defaultdict(list)
    for n in junction:
        members[uf.find(n)].append(n)
    centre = {c: tuple(np.mean([xy[n] for n in ns], axis=0)) for c, ns in members.items()}

    # one road per direction and cluster pair (most lanes, then shortest)
    best: dict[tuple[str, str], tuple[int, float, _Edge, bool]] = {}
    for e in edges:
        ca, cb = uf.find(e.a), uf.find(e.b)
        if ca == cb:
            continue
        length = _length(e.points)
        for forward in (True, False):
            if (e.oneway == 1 and not forward) or (e.oneway == -1 and forward):
                continue
            key = (ca, cb) if forward else (cb, ca)
            lanes = e.lanes_fwd if forward else e.lanes_bwd
            cand = (lanes, -length, e, forward)
            if key not in best or cand[:2] > best[key][:2]:
                best[key] = cand
    if not best:
        raise ScenarioValidationError(
            [
                ValidationIssue(
                    "osm", "the roads collapse into a single junction; export a larger area", "E700"
                )
            ],
            "OSM import",
        )

    roads: list[dict[str, Any]] = []
    for (ca, cb), (lanes, _neg, e, forward) in sorted(
        best.items(), key=lambda kv: (kv[1][2].way, kv[0])
    ):
        line = list(e.points) if forward else list(reversed(e.points))
        a, b = centre[ca], centre[cb]
        inner = [
            q
            for q in line[1:-1]
            if math.dist(q, a) > JUNCTION_CLEARANCE and math.dist(q, b) > JUNCTION_CLEARANCE
        ]
        line = [a, *inner, b]
        clean = [(round(x, 2), round(y, 2)) for x, y in line]
        clean = [p for i, p in enumerate(clean) if i == 0 or p != clean[i - 1]]
        rid = f"w{e.way}_{ca}_{cb}"
        roads.append(
            {
                "id": _safe(rid),
                "from": _safe(f"n{ca}"),
                "to": _safe(f"n{cb}"),
                "points": [list(p) for p in clean],
                "lanes": [{} for _ in range(lanes)],
                "speed_limit": round(e.speed, 2),
                "name": "",
                "_kind": e.kind,
            }
        )
    return _finish(roads, centre, signals, xy, opts, warnings, len(ways))


def _join_joints(edges: list[_Edge], junction: set[str]) -> list[_Edge]:
    """Edges are already split only at junction nodes; joints between similar ways were
    left out of ``junction``, so a way end there continues in the next way: chain them."""
    by_start: dict[str, list[_Edge]] = defaultdict(list)
    for e in edges:
        by_start[e.a].append(e)
    out: list[_Edge] = []
    for e in edges:
        if e.a not in junction:
            continue  # a continuation, consumed below
        cur = e
        seen = 0
        while cur.b not in junction and seen < 10_000:
            nxt = [f for f in by_start.get(cur.b, []) if f is not cur]
            if not nxt:
                break
            f = nxt[0]
            cur = _Edge(
                cur.way,
                cur.a,
                f.b,
                [*cur.points, *f.points[1:]],
                cur.kind,
                cur.oneway,
                cur.lanes_fwd,
                cur.lanes_bwd,
                cur.speed,
            )
            seen += 1
        out.append(cur)
    return out


def _safe(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:\-]", "_", raw)[: C.MAX_ID_LENGTH]


def _finish(
    roads: list[dict[str, Any]],
    centre: dict[str, tuple[float, ...]],
    signals: list[str],
    xy: dict[str, tuple[float, float]],
    opts: OsmOptions,
    warnings: list[ValidationIssue],
    n_ways: int,
) -> ImportResult:
    def warn(path: str, message: str) -> None:
        warnings.append(ValidationIssue(path, message, "W700", Severity.warning))

    sig_xy = [xy[s] for s in signals if s in xy]
    dropped: list[str] = []
    for _attempt in range(25):
        nodes_used = {r["from"] for r in roads} | {r["to"] for r in roads}
        nb: dict[str, set[str]] = defaultdict(set)
        for r in roads:
            nb[r["from"]].add(r["to"])
            nb[r["to"]].add(r["from"])
        intersections = []
        for c, p in centre.items():
            iid = _safe(f"n{c}")
            if iid not in nodes_used:
                continue
            ix: dict[str, Any] = {"id": iid, "point": [round(p[0], 2), round(p[1], 2)]}
            degree = len(nb[iid])
            if degree == 1:
                ix["kind"] = "boundary"
            elif degree >= 3:
                near = any(math.dist(p, s) <= opts.signal_radius for s in sig_xy)
                ix["kind"] = "signalized" if near else "priority"
            intersections.append(ix)
        network = {
            "format": "urbanflow.scenario",
            "version": "1.0",
            "meta": {"name": opts.name, "description": "Imported from OpenStreetMap."},
            "simulation": {"duration": opts.duration},
            "network": {"intersections": intersections, "roads": [_public(r) for r in roads]},
        }
        try:
            base = Scenario.from_dict(network, source="OSM import")
            break
        except ScenarioValidationError as exc:
            bad = {int(m.group(1)) for i in exc.issues if i.severity is Severity.error
                   and (m := re.match(r"network\.roads\[(\d+)\]", i.path))}  # fmt: skip
            if not bad:
                raise
            for i in sorted(bad, reverse=True):
                dropped.append(roads[i]["id"])
                del roads[i]
            if not roads:
                raise
    else:  # pragma: no cover - 25 rounds of drops
        raise ScenarioValidationError(
            [ValidationIssue("osm", "could not build a valid network", "E700")], "OSM import"
        )
    if dropped:
        warn(
            "osm",
            f"{len(dropped)} road(s) could not be simulated (too short or odd shape); left out",
        )

    # traffic: boundary entries to reachable boundary exits
    resolved = base.resolved
    kinds = {ix.id: ix.kind for ix in resolved.network.intersections}
    by_id = {r["id"]: r for r in roads}
    entries = [r for r in roads if kinds.get(r["from"]) == "boundary"]
    exits = {r["id"] for r in roads if kinds.get(r["to"]) == "boundary"}
    graph = RoadGraph(resolved)
    rng = np.random.default_rng(opts.seed)
    flows: list[dict[str, Any]] = []
    for r in entries:
        reach, _prev = graph.tree(r["id"])
        targets = sorted(x for x in exits if x in reach and by_id[x]["to"] != r["from"])
        if not targets:
            continue
        weight = np.array(
            [len(by_id[x]["lanes"]) * CLASSES[by_id[x]["_kind"]][2] for x in targets], float
        )
        k = min(opts.exits_per_entry, len(targets))
        pick = rng.choice(len(targets), size=k, replace=False, p=weight / weight.sum())
        rate = CLASSES[r["_kind"]][2] * len(r["lanes"]) * opts.demand
        chosen = sorted(int(i) for i in pick)
        share = weight[chosen] / weight[chosen].sum()  # big roads attract more of the traffic
        for t, w in zip(chosen, share.tolist(), strict=True):
            if rate * w < MIN_FLOW_VPH:
                continue
            flows.append(
                {
                    "id": f"f{len(flows)}",
                    "origin": r["id"],
                    "destination": targets[t],
                    "rate": round(rate * w, 2),
                    "arrival": "poisson",
                }
            )
    if not flows:
        warn(
            "osm",
            "no boundary road leads to another one; the map has no traffic (add flows yourself)",
        )
    data = {**network, "demand": {"flows": flows}}
    data["meta"] = {
        "name": opts.name,
        "description": (
            f"Imported from OpenStreetMap: {len(intersections)} junctions, {len(roads)} roads "
            f"from {n_ways} ways."
        ),
        "generator": {
            "name": "osm",
            "params": opts.model_dump(mode="json"),
            "urbanflow_version": __version__,
        },
    }
    scenario = Scenario.from_dict(data, source="OSM import")
    return ImportResult(scenario, (*warnings, *scenario.issues))


def _public(road: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in road.items() if not k.startswith("_")}


def load(path: str | Path, options: OsmOptions | None = None) -> ImportResult:
    """Read an ``.osm`` file (size-capped) and convert it."""
    return convert(read_text(path), options)
