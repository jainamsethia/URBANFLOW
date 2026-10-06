# Scenario format

A scenario is one JSON file: the road network, vehicle types, demand and default simulation
settings. The format is versioned (`"version": "1.0"`), described by a JSON Schema
([reference/scenario.schema.json](reference/scenario.schema.json)) and validated with
friendly, path-addressed messages.

Most structure can be left out and is **derived**: intersection kinds and radii, road
geometry, movements (turns), lane connections, priority roads, signal phases and stop
lanes. `urbanflow validate` checks a file; `Scenario.save(path, resolved=True)` writes the
fully explicit form that the simulator actually uses.

## Conventions

| Topic | Rule |
|---|---|
| Units | Metres, seconds, m/s, m/s², veh/h. No angles in JSON. |
| Coordinates | Local planar x (east), y (north) in metres, `abs(coord) <= 1e7`. |
| Ids | `^[A-Za-z0-9_.:>\-]{1,128}$` |
| Derived ids | lane `"{road}_{i}"`, connection `"{lane_in}->{lane_out}"`, movement `"{from_road}->{to_road}"`, phase `"p{k}"`, stop `"{line}.{k}"` |
| Vehicle ids | flows `"{flow}.{k}"`, transit `"{line}.{k}"`, trips their own id |
| Lanes | Lane 0 is the median (innermost) lane; lane `n-1` is the curb lane. A road's `points` are its median edge. |
| Driving side | `network.drive_side` is `right` (default) or `left`; lane stacking and turn sides mirror. |
| Trimming | Each road end is trimmed by `max(0, radius - dist(endpoint, intersection.point))`. |
| Time | Step `n` covers `[n dt, (n+1) dt)`; demand windows are half-open `[begin, end)`. |
| Point | `[x, y]`, a JSON array of two finite numbers. |

## A complete example

<!-- BEGIN GENERATED: example -->
```json
{
  "format": "urbanflow.scenario",
  "version": "1.0",
  "meta": {"name": "single-intersection-demo",
           "description": "4-arm signalized junction: cars, a probe trip and a bus line.",
           "tags": ["demo", "signalized"]},
  "simulation": {"dt": 1.0, "duration": 3600, "seed": 42},
  "network": {
    "drive_side": "right", "lane_width": 3.2, "speed_limit": 13.89,
    "intersections": [
      {"id": "J", "kind": "signalized", "point": [0, 0],
       "signal": {
         "controller": {"type": "fixed_time", "params": {"offset": 0}},
         "yellow": 3, "all_red": 1, "min_green": 5, "max_green": 60,
         "phases": [
           {"id": "NS", "duration": 30, "green": {
             "N_in->S_out": "G", "N_in->W_out": "G", "N_in->E_out": "g",
             "S_in->N_out": "G", "S_in->E_out": "G", "S_in->W_out": "g"}},
           {"id": "EW", "duration": 25, "green": {
             "E_in->W_out": "G", "E_in->N_out": "G", "E_in->S_out": "g",
             "W_in->E_out": "G", "W_in->S_out": "G", "W_in->N_out": "g"}}]}},
      {"id": "N", "kind": "boundary", "point": [0, 200]},
      {"id": "E", "kind": "boundary", "point": [200, 0]},
      {"id": "S", "kind": "boundary", "point": [0, -200]},
      {"id": "W", "kind": "boundary", "point": [-200, 0]}
    ],
    "roads": [
      {"id": "N_in",  "from": "N", "to": "J", "lanes": [{}, {}]},
      {"id": "N_out", "from": "J", "to": "N", "lanes": [{}, {}]},
      {"id": "S_in",  "from": "S", "to": "J", "lanes": [{}, {}]},
      {"id": "S_out", "from": "J", "to": "S", "lanes": [{}, {}]},
      {"id": "E_in",  "from": "E", "to": "J", "lanes": [{}, {}]},
      {"id": "E_out", "from": "J", "to": "E", "lanes": [{}, {}]},
      {"id": "W_in",  "from": "W", "to": "J", "points": [[-200, 0], [-100, 0], [0, 0]],
       "lanes": [{}, {"speed_limit": 11.11}]},
      {"id": "W_out", "from": "J", "to": "W", "lanes": [{}, {}]}
    ]
  },
  "vehicle_types": [
    {"id": "car", "length": 4.5},
    {"id": "city_bus", "vclass": "bus", "length": 12.0, "width": 2.55, "max_speed": 22.0,
     "accel": 1.0, "decel": 1.5, "min_gap": 2.5, "speed_factor": {"mean": 1.0, "std": 0.0}}
  ],
  "demand": {
    "flows": [
      {"id": "ns", "route": ["N_in", "S_out"], "rate": 450, "arrival": "poisson"},
      {"id": "sn", "origin": "S_in", "destination": "N_out", "period": 8.0},
      {"id": "we", "origin": "W_in", "destination": "E_out", "rate": 400, "arrival": "binomial",
       "type_mix": {"car": 0.9, "city_bus": 0.1}, "begin": 0, "end": 3000},
      {"id": "ew_left", "route": ["E_in", "S_out"], "rate": 120, "arrival": "poisson"}
    ],
    "trips": [{"id": "probe", "depart": 120, "route": ["W_in", "N_out"], "depart_speed": 0}],
    "transit": [
      {"id": "bus1", "vehicle_type": "city_bus", "route": ["W_in", "E_out"], "headway": 600, "begin": 60,
       "stops": [{"road": "W_in", "position": 120, "dwell": 20},
                 {"road": "E_out", "position": 60, "dwell": 15}]}
    ]
  }
}
```
<!-- END GENERATED: example -->

Movements are derived here: `N_in->E_out` is a left turn because the heading changes by
+90°. The radius of `J` derives to 2 × 3.2 + 2 = 8.4 m, so its approach lanes are 191.6 m
long.

## Derivation rules

`derive.resolve(spec)` is pure, deterministic and idempotent. In order:

1. **Vehicle types.** The built-in `car`, `bus`, `truck` and `emergency` types are updated
   with the fields you set on a type of the same id; new ids are added.
2. **Kind.** One neighbouring intersection (or none) gives `boundary`; at most two incoming
   and two outgoing roads between exactly two neighbours gives `uncontrolled`; everything
   else is `signalized`.
3. **Radius.** 0 for boundaries, otherwise the widest incident road plus 2 m.
4. **Road points.** A straight line between the two intersection points.
5. **Movements.** Every (incoming, outgoing) road pair. With `Δ` the heading change: a
   U-turn returns to the origin with `|Δ| >= 135°` (only with `allow_uturns`); the pair
   with the smallest `|Δ| <= 45°` is straight; otherwise `Δ > 0` is left, else right.
6. **Lane connections.** Movements are ordered from the median side outward. With at
   least as many lanes as movements each movement gets a contiguous block and the extra
   lanes go to the straight movement; otherwise lanes are shared and the straight movement
   uses every lane. 2 lanes with L/S/R give L{0} S{0,1} R{1}; 3 lanes give L{0} S{1} R{2};
   4 lanes give L{0} S{1,2} R{3}.
7. **Major roads** (priority kind): the opposing pair of incoming roads with the most
   lanes, then the highest speed limit.
8. **Signal phases** (signalized, when `phases` is omitted): opposing approaches pair into
   axes; the template (`two_phase`, `protected_left`, `split`, or `auto`) builds the
   phases, 30 s each, controller `fixed_time`.
9. **Stops.** Default lane is the curb lane, default id `"{line}.{k}"`.

## Demand

A flow uses exactly one of `route` (fixed road list), `routes` (a weighted route
distribution drawn per vehicle) or `origin` + `destination` (+ `via`), and exactly one of
`rate` (veh/h) or `period` (s). Arrivals are `uniform` (`t_k = begin + k h`), `poisson`
(exponential gaps from the flow's own random stream) or `binomial` (at most one vehicle per
step with probability `rate dt / 3600`). Trips are single vehicles; transit lines run on a
headway or explicit departures and dwell at their stops.

## Validation

Validation runs in stages: **L**oad (size cap 64 MiB, UTF-8, JSON without NaN, format and
version), **S**tructure (the JSON Schema rules), **P**re-derive (ids, references,
geometry), derive, and **Q** post-derive (movements, signals, demand, transit, simulation).
Deep checks (**X**: registered model and router names, model parameters, network
compilation, signal controller names and parameters) run in `urbanflow.check`,
which `urbanflow validate` (by default) and `Simulation` call. Every issue has a code, a JSON path and a message; one broken
reference is reported once, not as a cascade:

```text
Scenario validation failed:
  - network.intersections[0].signal.phases[1]: phase contains unknown movement "E_in->W_ot" (did you mean "E_in->W_out"?)
  - demand.flows[3].route[1]: route is not connected: no movement from "E_in" to "E_out" at intersection "J"
  - demand.trips[0].vehicle_type: unknown vehicle type "cr" (did you mean "car"?) (available: bus, car, city_bus, emergency, truck)
  - demand.transit[0].stops[1].position: stop position 250.0 m must be in [12.0, 190.6] (bus length .. lane length − 1 m)
```

Errors (E) prevent loading; warnings (W) are reported and fail only with `--strict`.
Stage B codes come from the builder.

<!-- BEGIN GENERATED: issue-codes -->
| code | severity | stage | message |
|---|---|---|---|
| E000 | error | L (load) | `invalid JSON: {detail} (line {line}, column {col})` / `file is not valid UTF-8: {detail}` / `NaN/Infinity are not allowed (line {line})` / `invalid JSON: {detail}` |
| E001 | error | S (structure) | `missing required field "{field}"` |
| E002 | error | S (structure) | `unknown field "{field}"{hint}` |
| E003 | error | S (structure) | `expected {expected}, got {got}` |
| E004 | error | S (structure) | `must be {op} {bound} (got {input})` / `must have at least {n} items` / `must have at most {n} items` / `must be at least {n} characters long` / `must be at most {n} characters long` |
| E005 | error | S (structure) | `invalid value {input}; expected one of: {choices}` |
| E006 | error | S (structure) | `invalid id {input}: use 1-128 characters from A-Z a-z 0-9 _ . : - >` |
| E007 | error | S (structure) | `specify exactly one of "rate" or "period"` / `specify exactly one of "route", "routes" or "origin" and "destination"` / `"vehicle_type" and "type_mix" are mutually exclusive` / `specify exactly one of "headway" or "departures"` / `"end" ({end}) must be greater than "begin" ({begin})` / `max_green ({a}) must exceed min_green ({b})` / `emergency_decel ({e}) must be >= decel ({d})` / `speed_factor requires min <= mean <= max` / `specify exactly one of "route" or "origin" and "destination"` / `"via" requires "origin" and "destination"` / `"departures" must be sorted in ascending order` |
| E009 | error | S (structure) | `{message}` |
| E010 | error | L (load) | `not an UrbanFlow scenario: "format" must be "urbanflow.scenario" (got {got})` |
| E011 | error | L (load) | `scenario version {v} is newer than supported {cur}; upgrade urbanflow` |
| E012 | error | L (load) | `unsupported scenario version {v}; no migration path to {cur}` |
| E013 | error | L (load) | `file is {size:.1f} MB; the limit is {limit} MB` |
| E021 | error | B (builder / ops) | `unknown {kind} "{id}"{hint}` |
| E022 | error | B (builder / ops) | `{kind} "{id}" is still referenced by {refs}; remove those first or use cascade=True` |
| E024 | error | B (builder / ops) | `the id of an existing {kind} cannot be changed; remove it and add a new one` |
| E101 | error | P (pre-derive) | `duplicate intersection id "{id}" (first defined at network.intersections[{j}])` |
| E102 | error | P (pre-derive) | `duplicate road id "{id}" (first defined at network.roads[{j}])` |
| E103 | error | P (pre-derive) | `unknown intersection "{id}"{hint}` |
| E104 | error | P (pre-derive) | `road starts and ends at intersection "{id}"` |
| E105 | error | P (pre-derive) | `road id "{id}" collides with lane id "{lane}" derived from road "{other}"` |
| E106 | error | P (pre-derive) | `only signalized intersections can have a signal (kind is "{kind}")` |
| E107 | error | P (pre-derive) | `boundary intersections cannot have movements` |
| E108 | error | P (pre-derive) | `major road "{road}" does not end at intersection "{id}"` |
| E201 | error | Q (post-derive) | `road "{road}" does not end at intersection "{id}"` |
| E202 | error | Q (post-derive) | `road "{road}" does not start at intersection "{id}"` |
| E203 | error | Q (post-derive) | `unknown road "{road}"{hint}` |
| E204 | error | Q (post-derive) | `duplicate movement id "{id}" (first defined at {where})` |
| E205 | error | Q (post-derive) | `duplicate movement "{a}" -> "{b}" (also movements[{j}])` |
| E206 | error | Q (post-derive) | `lane {lane} does not exist: road "{road}" has {n} lanes (0-{max})` |
| E207 | error | Q (post-derive) | `lane {lane} does not exist: road "{road}" has {n} lanes (0-{max})` |
| E208 | error | Q (post-derive) | `duplicate connection {a}->{b}` |
| E301 | error | Q (post-derive) | `signalized intersection has no movements to control` |
| E302 | error | Q (post-derive) | `phase contains unknown movement "{m}"{hint}` |
| E303 | error | Q (post-derive) | `phase contains movement "{m}" of another intersection ("{other}")` |
| E304 | error | Q (post-derive) | `duplicate phase id "{id}"` |
| E305 | error | Q (post-derive) | `initial_phase {k} out of range ({n} phases)` |
| E306 | error | Q (post-derive) | `phase min_green ({a} s) exceeds max_green ({b} s)` / `duration {d} s is below min_green {g} s` |
| E401 | error | P (pre-derive) | `duplicate vehicle type id "{id}"` |
| E402 | error | Q (post-derive) | `unknown vehicle type "{t}"{hint} (available: {list})` |
| E501 | error | Q (post-derive) | `duplicate demand id "{id}" (also demand.{kind2}[{j}])` / `id "{id}" collides with vehicle ids generated by {kind} "{other}"` |
| E502 | error | Q (post-derive) | `unknown road "{road}"{hint}` |
| E503 | error | Q (post-derive) | `route is not connected: no movement from "{a}" to "{b}" at intersection "{j}"` / `route is not connected: "{a}" ends at "{ja}" but "{b}" starts at "{jb}"` |
| E504 | error | Q (post-derive) | `"{d}" is not reachable from "{o}"{via_text}` |
| E505 | error | Q (post-derive) | `lane {k} does not exist on road "{road}" ({n} lanes)` / `lane {k} of road "{road}" has no connection to the next road "{next}" of the route` |
| E506 | error | Q (post-derive) | `binomial arrivals allow at most one vehicle per step: {rate} veh/h > {max:.0f} veh/h at dt={dt} s (use "poisson" or split the flow)` |
| E507 | error | Q (post-derive) | `flow has neither "end" nor "count" but simulation.duration is null; the run would never end` |
| E601 | error | Q (post-derive) | `stop road "{road}" is not on the line's route` |
| E602 | error | Q (post-derive) | `stops must follow route order: stop {k} is before stop {k1} along the route` |
| E604 | error | Q (post-derive) | `lane {k} does not exist on road "{road}" ({n} lanes)` |
| E606 | error | Q (post-derive) | `duplicate stop id "{id}"` |
| E801 | error | P (pre-derive) | `point repeats the previous point; consecutive points must differ` |
| E802 | error | P (pre-derive) | `road is {L:.1f} m long but intersections "{a}" and "{b}" reserve {ra:.1f} m + {rb:.1f} m; lanes would be {rem:.1f} m (minimum {min} m)` |
| E806 | error | X (deep check) | `lane "{id}" is {L:.1f} m long but vehicle type "{t}" routed through it needs {need:.1f} m (length + min_gap); don't-block-the-box admission (F.3) could never admit it and insertion would place it past the lane end` |
| E807 | error | Q (post-derive) | `stop position {p} m must be in [{len}, {Lm1:.1f}] (bus length .. lane length − 1 m)` |
| E901 | error | X (deep check) | `unknown signal controller "{t}"{hint} (available: {list})` |
| E902 | error | X (deep check) | `{message}` |
| E903 | error | X (deep check) | `unknown parameter "{p}" for car-following model "{m}" (known: {list})` |
| E904 | error | X (deep check) | `unknown {kind} "{name}"{hint} (available: {list})` |
| E905 | error | X (deep check) | `lane geometry is degenerate: {detail}` |
| W101 | warning | P (pre-derive) | `intersection "{id}" is not connected to any road` |
| W102 | warning | P (pre-derive) | `major_roads is ignored for kind "{kind}"` |
| W103 | warning | P (pre-derive) | `intersections "{a}" and "{b}" overlap ({d:.1f} m apart, radii {ra:.1f} + {rb:.1f} m)` |
| W201 | warning | Q (post-derive) | `road "{road}" enters "{id}" but has no movements out of it; trips can only end there (use kind "boundary" for exits)` |
| W202 | warning | Q (post-derive) | `lane "{lane}" has no outgoing connection; vehicles must change lanes before the stop line` |
| W301 | warning | Q (post-derive) | `movement "{m}" is never green in any phase; its traffic can never enter` |
| W302 | warning | X (deep check) | `movements "{a}" and "{b}" cross but are both protected (G); make one permissive (g)` |
| W303 | warning | Q (post-derive) | `yellow is 0 s; vehicles approaching at speed cannot stop safely at phase changes` |
| W304 | warning | X (deep check) | `connectors "{a}" and "{b}" cross twice; their conflict zones were merged` |
| W305 | warning | Q (post-derive) | `phase durations, yellow or all_red are not multiples of dt={dt} s; the realised cycle is {cq} s instead of {c} s` |
| W306 | warning | Q (post-derive) | `stop lane cannot reach the next route road and the stop is {d:.0f} m (< 50 m) from the stop line` |
| W501 | warning | Q (post-derive) | `{rate} veh/h exceeds the entry capacity of road "{road}" (~{cap:.0f} veh/h); vehicles will queue outside the network` |
| W502 | warning | Q (post-derive) | `starts at {t} s, after the simulation ends ({d} s)` |
| W701 | warning | Q (post-derive) | `dt={dt} s is outside the recommended 0.2-1.0 s` |
| W702 | warning | Q (post-derive) | `duration {d} s is not a multiple of dt {dt} s; it is rounded down to {r} s` |
| W801 | warning | P (pre-derive) | `road ends {d:.1f} m from intersection "{j}", outside its radius {r:.1f} m; connectors bridge the gap` |
| W802 | warning | P (pre-derive) | `sharp bend ({deg:.0f} deg); offset lanes may self-intersect` |
<!-- END GENERATED: issue-codes -->

The entry capacity of W501 is the road's lane count times the peak IDM equilibrium flow
of one lane at the road speed limit v<sub>0</sub>, 3600 · max<sub>v</sub> v /
(s<sub>e</sub>(v) + ℓ) with s<sub>e</sub>(v) = (s<sub>0</sub> + vT) /
√(1 − (v/v<sub>0</sub>)<sup>δ</sup>), for the flow's vehicle types (averaged by headway
over a `type_mix`); see [Simulation model](simulation-model.md#road-capacity).

## Building scenarios in Python

```python
from urbanflow import ScenarioBuilder

S = 300.0
b = ScenarioBuilder("grid-2x2", duration=3600)
for r in range(2):
    for c in range(2):
        b.intersection(f"J{r}{c}", (c * S, r * S))
    b.boundary(f"W{r}", (-S / 2, r * S)).boundary(f"E{r}", (1.5 * S, r * S))
for c in range(2):
    b.boundary(f"S{c}", (c * S, -S / 2)).boundary(f"N{c}", (c * S, 1.5 * S))
for r in range(2):
    b.two_way(f"W{r}", f"J{r}0", lanes=2).two_way(f"J{r}0", f"J{r}1", lanes=2)
    b.two_way(f"J{r}1", f"E{r}", lanes=2)
for c in range(2):
    b.two_way(f"S{c}", f"J0{c}", lanes=2).two_way(f"J0{c}", f"J1{c}", lanes=2)
    b.two_way(f"J1{c}", f"N{c}", lanes=2)
for r in range(2):
    b.flow(f"we{r}", origin=f"W{r}_J{r}0", destination=f"J{r}1_E{r}", rate=500, arrival="poisson")
    b.flow(f"ew{r}", origin=f"E{r}_J{r}1", destination=f"J{r}0_W{r}", rate=500, arrival="poisson")
for c in range(2):
    b.flow(f"sn{c}", origin=f"S{c}_J0{c}", destination=f"J1{c}_N{c}", rate=400, arrival="poisson")
b.signal_all(template="two_phase", green=30)
scenario = b.build()
print(scenario.summary())
```

Builder methods check ids locally and raise `ScenarioValidationError` with one issue;
`build()` runs the full validation. `Scenario.edit()` returns a builder for an existing
scenario; `update`, `move`, `remove` and `materialize` edit it in place. Fields you set on a
built-in vehicle type (`car`, `bus`, `truck`, `emergency`) override only those fields; the
merged type must still satisfy `emergency_decel >= decel` and `min <= mean <= max`.

## Generators and bundled scenarios

```bash
urbanflow generate --list
urbanflow generate single_intersection -p kind=priority -p arms=3 -o t.json
urbanflow generate single_intersection --describe
```

In Python, `urbanflow.generate("single_intersection", kind="uncontrolled")` returns a
`Scenario`. Bundled scenarios ship with the package: `urbanflow.bundled("single_intersection")`
is their path and `urbanflow init --template NAME` copies one into a new workspace.
Custom generators register with `@register_generator("name", params=Params)`.

## Versioning

Files carry `"version": "MAJOR.MINOR"`. Minor versions only add optional fields, so older
minor versions load unchanged; a newer minor version asks you to upgrade UrbanFlow (E011);
other versions need a migration (E012). `save()` always writes the current version.

## Schema reference

Generated from the pydantic models by `scripts/gen_scenario_docs.py`.

<!-- BEGIN GENERATED: schema -->
### ScenarioSpec

An UrbanFlow scenario (format urbanflow.scenario, version 1.0).

| key | type | default | constraints | description |
|---|---|---|---|---|
| `$schema` | string or null | null |  | Ignored; an editor hint only. |
| `format` | "urbanflow.scenario" | required |  | Always "urbanflow.scenario". |
| `version` | string | required | pattern `^\d+\.\d+$` | Format version "MAJOR.MINOR". |
| `meta` | [MetaSpec](#metaspec) | required |  | Name, description, tags and provenance. |
| `simulation` | [SimulationSpec](#simulationspec) | derived / optional |  | Scenario-level simulation defaults. |
| `network` | [NetworkSpec](#networkspec) | required |  | Road network. |
| `vehicle_types` | [VehicleTypeSpec](#vehicletypespec)[] | [] | max 256 items | Vehicle types, merged over the built-ins by id. |
| `demand` | [DemandSpec](#demandspec) | derived / optional |  | Flows, trips and transit lines. |

### MetaSpec

Descriptive metadata; excluded from the content hash.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `name` | string | required | min length 1, max length 100 | Scenario name. |
| `description` | string | "" | max length 10000 | Free-text description. |
| `tags` | string[] | [] | max 32 items | Free-form tags. |
| `authors` | string[] | [] |  | Authors. |
| `generator` | [GeneratorInfo](#generatorinfo) or null | null |  | Provenance (set by generators and importers). |

### GeneratorInfo

Provenance written by generators and importers.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `name` | string | required |  | Generator or importer name. |
| `params` | object | derived / optional |  | Parameters it was called with. |
| `urbanflow_version` | string | required |  | UrbanFlow version that produced the file. |

### SimulationSpec

Scenario-level defaults of the run configuration (same fields as SimulationConfig).

| key | type | default | constraints | description |
|---|---|---|---|---|
| `dt` | number | 1.0 | >= 0.1, <= 2.0 | Step length, s (W701 outside 0.2-1.0 s). |
| `duration` | number or null | 3600.0 | > 0 | Simulated time, s; null runs until demand is exhausted and the network is empty. |
| `seed` | integer | 0 | >= 0, <= 9223372036854775807 | Root seed of every random stream. |
| `car_following` | string | "idm" |  | Registered car-following model. |
| `lane_change_model` | string | "mobil" |  | Registered lane-change model. |
| `lane_changing` | boolean | true |  | Enable lane changes. |
| `router` | string | "shortest" |  | Registered router. |
| `routing_weight` | "length" or "freeflow_time" | "freeflow_time" |  | Edge weight of the shortest-path router. |
| `halting_speed` | number | 0.1 | >= 0 | Vehicles slower than this are halting, m/s. |
| `deadlock_timeout` | number | 300.0 | >= 0 | Teleport vehicles stuck this long, s; 0 turns the watchdog off. |
| `max_vehicles` | integer or null | null | >= 1 | Cap on running vehicles; extra demand queues. |

### NetworkSpec

The road network.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `drive_side` | [DriveSide](#driveside) | "right" |  | Side of the road vehicles drive on (right or left). |
| `lane_width` | number | 3.2 | >= 2.0, <= 5.0 | Default lane width, m. |
| `speed_limit` | number | 13.89 | >= 1.0, <= 70.0 | Default speed limit, m/s. |
| `allow_uturns` | boolean | false |  | Also derive U-turn movements. |
| `intersections` | [IntersectionSpec](#intersectionspec)[] | required | min 1 items, max 200000 items | Intersections (nodes). |
| `roads` | [RoadSpec](#roadspec)[] | required | min 1 items, max 500000 items | Roads (directed edges). |

### IntersectionSpec

A node of the road network.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Intersection id. |
| `point` | [number, number] | required | min 2 items, max 2 items | Centre [x, y], m. |
| `kind` | [IntersectionKind](#intersectionkind) or null | null |  | signalized, priority, uncontrolled or boundary; derived from topology. |
| `radius` | number or null | null | >= 0, <= 200.0 | Stop-line distance from point, m; derived: widest incident road + 2 m. |
| `major_roads` | string[] or null | null |  | Incoming roads with right of way (priority kind); derived. |
| `movements` | [MovementSpec](#movementspec)[] or null | null |  | Movements; when present the list is authoritative (all or nothing). |
| `signal` | [SignalSpec](#signalspec) or null | null |  | Signal (signalized only); derived for signalized intersections. |

### RoadSpec

A one-way road from one intersection to another.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Road id. |
| `from` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Start intersection. |
| `to` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | End intersection. |
| `points` | [number, number][] or null | null | min 2 items, max 10000 items | Median-edge polyline [[x, y], ...], m; derived: straight between the intersections. |
| `lanes` | [LaneSpec](#lanespec)[] | required | min 1 items, max 16 items | Lanes from the median outward; [{}, {}] is two default lanes. |
| `speed_limit` | number or null | null | >= 1.0, <= 70.0 | Default lane speed limit, m/s; derived: network.speed_limit. |
| `lane_width` | number or null | null | >= 2.0, <= 5.0 | Default lane width, m; derived: network.lane_width. |
| `name` | string | "" | max length 200 | Street name for display. |

### LaneSpec

One lane; lane 0 is the median (innermost) lane.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `width` | number or null | null | >= 2.0, <= 5.0 | Lane width, m (2-5); derived: the road's lane_width. |
| `speed_limit` | number or null | null | >= 1.0, <= 70.0 | Speed limit, m/s (1-70); derived: the road's speed_limit. |

### MovementSpec

A road-to-road turn at an intersection.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string or null | null | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Movement id; derived: "{from_road}->{to_road}". |
| `from_road` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Incoming road (ends at this intersection). |
| `to_road` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Outgoing road (starts at this intersection). |
| `turn` | [TurnKind](#turnkind) or null | null |  | straight, left, right or uturn; derived from geometry. |
| `priority` | "major" or "minor" or null | null |  | Right of way at priority intersections; derived from major_roads. |
| `connections` | [ConnectionSpec](#connectionspec)[] or null | null | min 1 items | Lane connections; derived from the lane mapping rules. |

### ConnectionSpec

A lane-to-lane connector through an intersection.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `from_lane` | integer | required | >= 0 | Lane index on the incoming road. |
| `to_lane` | integer | required | >= 0 | Lane index on the outgoing road. |
| `shape` | [number, number][] or null | null | min 2 items | Explicit connector polyline [[x, y], ...]; null = cubic Bezier from the lane ends. |

### SignalSpec

Traffic signal of a signalized intersection.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `controller` | [ControllerSpec](#controllerspec) | derived / optional |  | Controller type and parameters. |
| `phases` | [PhaseSpec](#phasespec)[] or null | null | min 1 items, max 32 items | Phases in cycle order; derived from template when omitted. |
| `template` | "auto" or "two_phase" or "protected_left" or "split" | "auto" |  | Phase template used when phases are omitted. |
| `yellow` | number | 3.0 | >= 0, <= 10.0 | Yellow interval, s (inserted automatically for movements losing green). |
| `all_red` | number | 1.0 | >= 0, <= 10.0 | All-red interval after yellow, s. |
| `min_green` | number | 5.0 | >= 0, <= 120.0 | Minimum green, s. |
| `max_green` | number | 60.0 | > 0, <= 600.0 | Maximum green, s (> min_green). |
| `initial_phase` | integer | 0 | >= 0 | Index of the phase active at t = 0. |

### ControllerSpec

Signal controller: a registry name plus its parameters.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `type` | string | "fixed_time" | min length 1 | Registered controller (fixed_time, actuated, max_pressure, webster, external, ...). |
| `params` | object | derived / optional |  | Controller parameters, checked against the controller's Params model (deep check). |

### PhaseSpec

A signal phase: the movements that have green.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string or null | null | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Phase id; derived: "p{k}". |
| `green` | object | required | min 1 entries | movement id -> "G" (protected) or "g" (permissive); unlisted movements are red. |
| `duration` | number | 30.0 | > 0, <= 600.0 | Fixed-time green, s. |
| `min_green` | number or null | null | >= 0, <= 120.0 | Minimum green, s; inherits the signal's min_green. |
| `max_green` | number or null | null | > 0, <= 600.0 | Maximum green, s; inherits the signal's max_green. |

### VehicleTypeSpec

A vehicle type; fields you set override the built-in type with the same id.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Type id (car, bus, truck and emergency are built in). |
| `vclass` | [VehicleClass](#vehicleclass) | "car" |  | car, bus, truck or emergency. |
| `length` | number | 5.0 | > 0, <= 30.0 | Length, m. |
| `width` | number | 1.8 | > 0, <= 4.0 | Width, m. |
| `max_speed` | number | 36.1 | > 0, <= 70.0 | Maximum speed, m/s; effective v0 = min(max_speed, limit x speed_factor). |
| `accel` | number | 1.5 | > 0, <= 10.0 | IDM maximum acceleration a, m/s^2. |
| `decel` | number | 2.0 | > 0, <= 10.0 | IDM comfortable deceleration b, m/s^2. |
| `emergency_decel` | number | 6.0 | > 0, <= 15.0 | Physical braking bound, m/s^2 (>= decel). |
| `min_gap` | number | 2.0 | >= 0, <= 20.0 | Jam distance s0, m. |
| `headway` | number | 1.1 | > 0, <= 10.0 | Desired time headway T, s. |
| `speed_factor` | [SpeedFactorSpec](#speedfactorspec) | derived / optional |  | Speed factor distribution, sampled per vehicle from the spawning flow's stream. |
| `politeness` | number | 0.2 | >= 0, <= 1 | MOBIL politeness p. |
| `lc_threshold` | number | 0.1 | >= 0, <= 5.0 | MOBIL threshold a_th, m/s^2. |
| `lc_safe_decel` | number | 4.0 | > 0, <= 15.0 | MOBIL safe deceleration b_safe, m/s^2. |
| `model_params` | {string: number} | derived / optional |  | Model-specific extras such as delta (checked by the deep validation). |
| `color` | string or null | null | pattern `^#[0-9A-Fa-f]{6}$` | #RRGGBB; null uses the per-class palette. |

### SpeedFactorSpec

Per-vehicle speed factor f ~ N(mean, std) clipped to [min, max] (min <= mean <= max).

| key | type | default | constraints | description |
|---|---|---|---|---|
| `mean` | number | 1.0 | > 0 | Mean speed factor. |
| `std` | number | 0.1 | >= 0 | Standard deviation. |
| `min` | number | 0.8 | > 0 | Lower clip. |
| `max` | number | 1.2 | > 0 | Upper clip. |

### DemandSpec

Traffic demand. Flow, trip and transit ids share one namespace.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `flows` | [FlowSpec](#flowspec)[] | [] |  | Vehicle flows. |
| `trips` | [TripSpec](#tripspec)[] | [] |  | Individual trips. |
| `transit` | [TransitLineSpec](#transitlinespec)[] | [] |  | Transit lines. |

### FlowSpec

A stream of vehicles.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Flow id; vehicles are "{flow}.{k}". |
| `vehicle_type` | string or null | null | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Vehicle type; "car" when neither this nor type_mix is given. |
| `type_mix` | object or null | null | min 1 entries | type id -> weight, sampled per vehicle (mutually exclusive with vehicle_type). |
| `route` | string[] or null | null | min 1 items | Fixed road sequence. |
| `routes` | [RouteChoiceSpec](#routechoicespec)[] or null | null | min 1 items, max 64 items | Route distribution; one entry is drawn per vehicle at spawn. |
| `origin` | string or null | null | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Origin road (the router resolves the path at departure). |
| `destination` | string or null | null | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Destination road. |
| `via` | string[] | [] |  | Ordered waypoint roads (origin/destination mode only). |
| `rate` | number or null | null | > 0 | Vehicles per hour. |
| `period` | number or null | null | > 0 | Seconds between vehicles (instead of rate). |
| `arrival` | "uniform" or "poisson" or "binomial" | "uniform" |  | Arrival process: uniform, poisson or binomial. |
| `begin` | number | 0.0 | >= 0 | First possible departure, s. |
| `end` | number or null | null |  | End of the half-open window [begin, end), s; null = until the simulation ends. |
| `count` | integer or null | null | >= 1 | Maximum number of vehicles. |
| `depart_lane` | "best" or "random" or "first" or integer | "best" | >= 0 | best, random, first or a lane index. |
| `depart_speed` | "max" or number | "max" | >= 0 | max (insertion-safe) or a speed in m/s. |

### RouteChoiceSpec

One entry of a flow's route distribution.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `roads` | string[] | required | min 1 items | Fully connected road sequence. |
| `weight` | number | required | > 0 | Relative weight (normalised over the distribution). |

### TripSpec

A single vehicle with a fixed departure time.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Trip id (also the vehicle id). |
| `depart` | number | required | >= 0 | Departure time, s. |
| `vehicle_type` | string | "car" | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Vehicle type. |
| `route` | string[] or null | null | min 1 items | Fixed road sequence. |
| `origin` | string or null | null | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Origin road. |
| `destination` | string or null | null | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Destination road. |
| `via` | string[] | [] |  | Ordered waypoint roads (origin/destination mode only). |
| `depart_lane` | "best" or "random" or "first" or integer | "best" | >= 0 | best, random, first or a lane index. |
| `depart_speed` | "max" or number | "max" | >= 0 | max (insertion-safe) or a speed in m/s. |

### TransitLineSpec

A bus line with scheduled departures and stops.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Line id; vehicles are "{line}.{k}". |
| `vehicle_type` | string | "bus" | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Vehicle type. |
| `route` | string[] | required | min 1 items | Road sequence (always explicit). |
| `stops` | [StopSpec](#stopspec)[] | required | min 1 items | Stops in route order. |
| `headway` | number or null | null | > 0 | Seconds between departures. |
| `departures` | number[] or null | null | min 1 items | Explicit departure times, s (sorted). |
| `begin` | number | 0.0 | >= 0 | First departure, s. |
| `end` | number or null | null |  | No departures at or after this time, s. |
| `depart_lane` | "best" or "random" or "first" or integer | "best" | >= 0 | best, random, first or a lane index. |

### StopSpec

A transit stop on one of the line's roads.

| key | type | default | constraints | description |
|---|---|---|---|---|
| `id` | string or null | null | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Stop id; derived: "{line}.{k}". |
| `road` | string | required | pattern `^[A-Za-z0-9_.:>\-]{1,128}$` | Road on the line's route. |
| `position` | number | required | >= 0 | Where the bus front stops, m along the trimmed lane. |
| `lane` | integer or null | null | >= 0 | Lane index; derived: the curb lane (n-1). |
| `dwell` | number | 20.0 | >= 0 | Dwell time, s. |

### DriveSide

Side of the road vehicles drive on; lanes stack from the median toward this side.

| key | type | default | constraints | description |
|---|---|---|---|---|

### IntersectionKind

Control kind of an intersection (``int_kind``).

| key | type | default | constraints | description |
|---|---|---|---|---|

### TurnKind

Turn of a movement (``mov_turn``).

| key | type | default | constraints | description |
|---|---|---|---|---|

### VehicleClass

Vehicle class (``VehicleTypeSpec.vclass``); selects energy and palette defaults.

| key | type | default | constraints | description |
|---|---|---|---|---|
<!-- END GENERATED: schema -->
