# Python API

Everything below is importable from the top-level `urbanflow` package. Imports are lazy, so
`import urbanflow` stays fast. This page covers the parts available so far: scenarios,
deep validation, the `Simulation` facade, its views and vector state, traffic signals,
and results.

## Scenarios

```python
from urbanflow import Scenario, generate

scenario = generate("single_intersection", kind="uncontrolled", duration=600)
print(scenario.name, scenario.short_hash, scenario.summary()["roads"])
path = scenario.save("cross.json")  # atomic write
same = Scenario.load(path)  # validates; raises ScenarioValidationError
assert same.content_hash == scenario.content_hash
```

| member | meaning |
|---|---|
| `Scenario.load(path)`, `from_dict(data)`, `from_json(text)`, `from_spec(spec)` | validated constructors |
| `Scenario.generate(name, **params)` / `urbanflow.generate(...)` | run a registered generator |
| `urbanflow.bundled(name)` | path of a bundled scenario file |
| `spec`, `resolved` | the declared spec and the fully explicit (derived) one |
| `name`, `content_hash`, `short_hash`, `issues` | identity and load warnings |
| `save(path, resolved=False)`, `to_dict()`, `to_json()`, `edit()`, `summary()` | output and editing |

## Deep validation: `urbanflow.check`

```python
from urbanflow import SimulationConfig, check, generate

report = check(
    generate("single_intersection", kind="uncontrolled"), SimulationConfig(router="shortest")
)
assert report.ok
for issue in report.warnings:
    print(issue.code, issue.path, issue.message)
```

`check(scenario, config=None, *, strict=False) -> ValidationReport` takes a `Scenario` or a
file path. On top of the file-level validation it checks what a run needs: registered
names (`E904` for `simulation.car_following` and `simulation.router`), car-following model
parameters (`E903`), signal controller names and parameters (`E901`, `E902`) and the
network compile diagnostics (`E802`, `E806`, `E905`, `W302`,
`W304`). `strict=True` turns warnings into errors. `urbanflow validate` runs the same
checks, and `Simulation` runs them before it starts.

## Simulation

```python
from urbanflow import Simulation, SimulationConfig, generate

scenario = generate("single_intersection", kind="uncontrolled")
sim = Simulation(scenario, SimulationConfig(dt=0.5), seed=7, duration=600)
while not sim.done:
    sim.step(20)
    print(f"t={sim.time:5.0f} s  running={len(sim.vehicles):3d}")
print(sim.metrics.summary()["travel_time.mean"])
```

`Simulation(scenario, config=None, *, controllers=None, router=None, **overrides)`:

- `scenario`: a `Scenario`, a `ScenarioSpec`, a scenario dict or a file path.
- Configuration layers, lowest first: defaults < the scenario's `simulation` block <
  `config` (only the fields it sets) < `router` < `**overrides` (any `SimulationConfig`
  field, e.g. `seed=7, duration=600, debug_checks=True`).
- `controllers`: signal controllers replacing the scenario's, keyed by intersection id or
  `"*"` (every signalised intersection; per-id entries win). A value is a registry name,
  a `{"type": ..., "params": {...}}` mapping or a zero-argument factory returning a
  controller (for example a controller class); factories are called again at every
  `reset()`. Names and parameters are deep-checked (E901, E902), unknown ids raise
  `NotFoundError`.
- `router`: a registered router name or a router instance.
- `Simulation.from_scenario(source, **kwargs)` is the same constructor.

| member | meaning |
|---|---|
| `scenario`, `config`, `seed`, `dt` | what is being run |
| `time`, `step_count`, `end_time` | `time = step_count · dt`; `end_time = ⌊duration/dt⌋ · dt` (None without a duration) |
| `done`, `is_done()` | the time limit is reached, or (no duration) the network drained |
| `is_drained()` | demand exhausted and nobody running or waiting (ignores the limit) |
| `step(n=1)` | advance up to `n` steps (stops at `done`); `SimulationError` if done, closed or corrupted |
| `run(until=None, *, duration=None, progress=False)` | step to the absolute time `until`, for `duration` seconds, or until done; returns `get_results()`. `progress=True` draws a bar on stderr; a callable receives `ProgressInfo(time, end_time, step_count, running, arrived, steps_per_s)` |
| `reset(seed=None)` | fresh run from the cached network; `None` keeps the current seed |
| `close()`, `with Simulation(...) as sim:` | release; later steps raise |
| `get_results()` | a `SimulationResult`, valid at any time (partial if not done) |
| `metrics.summary()` | the flattened summary (see [Simulation model](simulation-model.md#summary)) |
| `snapshot()`, `restore(snapshot)`, `state_digest()` | [snapshots](#snapshots-and-digests) |
| `events` | [event callbacks](#events) and counts |
| `network`, `vehicles`, `lanes`, `roads`, `state`, `intersections`, `signals` | the views below |

## Vehicles

```python
from urbanflow import Simulation, generate

sim = Simulation(generate("single_intersection", kind="uncontrolled"), seed=1)
sim.run(until=120)
car = next(iter(sim.vehicles))  # running vehicles, uid order
print(car.id, car.road, car.lane_index, round(car.speed, 2), car.route)
sim.vehicles.remove(car.id)
probe = sim.vehicles.add(route=["E_in", "W_out"], id="probe")
sim.step()  # inserted during this step
sim.vehicles.set_speed("probe", 5.0, duration=30)
sim.vehicles.set_route("probe", ["E_in", "S_out"])  # turn left instead
print(sim.vehicles.count("running"), sim.vehicles.ids("removed"))
```

`sim.vehicles` is a `VehicleCollection`:

- `[id]` returns a `VehicleView`; unknown ids raise `NotFoundError` with a did-you-mean hint,
  and ids of vehicles that already left say so. `get(id)`, `in` and `len()` (running
  vehicles) work as usual.
- `ids(status="running")` and `count(status)` accept `pending`, `waiting_insert`,
  `running`, `arrived` and `removed`. `uid(id)` and `id_of(uid)` map between the two
  identities; `to_columns()` returns the running vehicles as columns (pandas-ready).
- Control: `add(route=... | origin=..., destination=..., via=(), vehicle_type="car",
  id=None, depart_lane="best", depart_speed="max")`, `remove(id)`,
  `set_speed(id, speed | None, *, duration=None)` and `set_route(id, roads)`. They take
  effect in the next step.
- `set_route(id, roads)` replaces the rest of the route: `roads` must start at the current
  road (the first road while the vehicle waits; the next road while it is on a connector)
  and be connected, else `CommandError` (unknown roads: `NotFoundError`). A vehicle
  admitted across its next stop line must continue through the road its connector leads
  to ([Simulation model](simulation-model.md#commands)).

A `VehicleView` is a frozen snapshot with the identity (`id`, `uid`, `type`, `vclass`,
`status`), the type parameters (`length`, `width`, `max_speed`, `max_accel`, `decel`,
`emergency_decel`, `min_gap`, `headway`, `politeness`, `model_params`), `speed_factor`,
`reaction_time` (= dt), the location (`road`, `lane`, `connector`, `intersection`,
`link_index`, `lane_index`, `position`, `x`, `y`, `heading`, `lateral_offset`; None while
waiting to be inserted), the motion (`speed`, `acceleration`, `desired_speed`), the route
(`route`, `route_index`, `next_road`, `destination`) and the trip so far (`depart_time`,
`insert_time`, `travel_time`, `waiting_time`, `stops`, `distance`, `speed_override`).
Coordinates are world coordinates; headings are radians counter-clockwise from +x.

## Lanes, roads and the network

```python
from urbanflow import Simulation, generate

sim = Simulation(generate("single_intersection", kind="uncontrolled"), seed=1)
sim.run(until=120)
queues = sim.lanes.queue_lengths()  # one entry per lane, read-only
busiest = sim.lanes.ids[int(queues.argmax())]
print(busiest, sim.lanes[busiest].vehicle_ids)
print(sim.roads["E_in"].travel_time, sim.network.index("road", "E_in"))
```

`sim.lanes` and `sim.roads` list ids in compiled order and index by id or position. Their
vector forms, `vehicle_counts()`, `halting_counts()`, `mean_speeds()` (NaN when empty),
`occupancy()` and `queue_lengths()`, are read-only arrays aligned with `ids`, computed once
per step. The queue is the run of halting vehicles from the stop line; the front vehicle
must be within 10 m of it. A `LaneView` has `id, road, index, length, width, speed_limit,
vehicle_count, halting_count, mean_speed, occupancy, queue_length, vehicle_ids` (front to
back); a `RoadView` has `id, from_intersection, to_intersection, lanes, n_lanes, length,
speed_limit, capacity_vph, utilization, vehicle_count, halting_count, mean_speed,
travel_time`. Every running vehicle is on exactly one lane or connector:
Σ lane counts + vehicles on connectors = `len(sim.vehicles)`.

`sim.network` exposes `lane_ids`, `connection_ids`, `road_ids`, `intersection_ids`,
`movement_ids`, `index(kind, id)`, the arrays `lane_length`, `lane_speed_limit`,
`lane_road` and `link_kind`, `geometry()` (render geometry) and `graph()` (a copy of the
road graph), and the read-only `compiled` network.

## Vector state

```python
import numpy as np
from urbanflow import Simulation, generate

sim = Simulation(generate("single_intersection", kind="uncontrolled"), seed=1)
sim.run(until=60)
state = sim.state
fast = state.uids[state.speeds > 10.0]  # key rows by uid
print(state.xy.shape, fast.size, float(np.mean(state.speeds)))
```

`sim.state` has one row per running vehicle: `uids`, `type_idx`, `links`, `positions`,
`speeds`, `accels`, `xy` (N × 2), `headings`, `waiting_times`, `distances`, and `ids`.
The arrays are read-only copies cached for the current step. `sim.state.raw` gives the
underlying vehicle table as zero-copy read-only views (one row per engine slot, `active`
marks the running ones); they may be stale after the next step.

## Signals and intersections

```python
from urbanflow import Simulation, bundled

sim = Simulation.from_scenario(bundled("single_intersection"), controllers={"J": "external"})
sim.signals.request_phase("J", "p1")  # min green, yellow and all-red are honoured
sim.run(until=30)
light = sim.signals["J"]
print(light.phase_id, light.stage, light.state_string, sim.signals.can_switch("J"))
junction = sim.intersections["J"]
print(junction.kind, [(m.id, m.state, m.rank) for m in junction.movements][:3])
```

`sim.signals` is a `SignalsAPI`:

| member | meaning |
|---|---|
| `ids`, `len()`, `in`, iteration | the signalised intersections (iteration yields `SignalView`s) |
| `[j]` | a `SignalView`: `phase_index, phase_id, stage, target, stage_elapsed, green_elapsed, remaining, min_green, max_green, cycle, held, movement_states, state_string, controller` (see [Traffic signals](traffic-signals.md#reading-the-signals)) |
| `phases(j)` | `PhaseInfo(index, id, green, duration, min_green, max_green)` per phase; `green` maps movement ids to `"G"`/`"g"` |
| `request_phase(j, phase)` | ask for a phase; needs a controller that accepts requests (`external`), else `CommandError` |
| `hold_phase(j, phase)`, `release(j)` | manual override with any controller; `release` resumes the parked controller |
| `set_phase(j, phase)` | jump at once, without intergreen (setup and tests) |
| `set_controller(j, ref)`, `controller(j)` | install a controller (continues from the current phase); the active instance |
| `can_switch(j)` | the signal is in green and past its minimum green, so a request would start switching in the next step |
| `phase_indices()`, `movement_states()` | read-only arrays for the current step: the phase per signalised intersection (aligned with `ids`) and the state code per movement (r 0, y 1, g 2, G 3, 255 unsignalised) |

Intersections are given by id (or index), phases by index or id. Unknown or unsignalised
intersections raise `NotFoundError` with a hint when read; invalid commands raise
`CommandError`. Commands take effect in the next step and are logged.

`sim.intersections` lists every node (boundaries too) in compiled order. An
`IntersectionView` has `id`, `kind`, `point` (world coordinates), `movements` and `signal`
(a `SignalView`, or None). Each `MovementInfo` has `id, from_road, to_road, turn, rank,
connections, state`: `rank` is the current right of way (the signal state code at a
signalised intersection: G 3, g 2, y 1, r 0; otherwise priority 3/2/1 or uncontrolled 1)
and `state` is `"G"`, `"g"`, `"y"`, `"r"` or None.

## Events

```python
from urbanflow import EventType, Simulation, bundled

sim = Simulation.from_scenario(bundled("single_intersection"), duration=300)
arrivals = []
sim.events.subscribe(EventType.vehicle_arrived, lambda e: arrivals.append((e.time, e.vehicle)))
changes = sim.events.subscribe("phase_changed", lambda e: print(e.step, e.intersection, e.data))
sim.run(until=100)
changes.unsubscribe()
sim.run()
assert len(arrivals) == sim.events.counts()[EventType.vehicle_arrived]
```

`sim.events.subscribe(types, callback)` registers `callback(event)` for one event type, a
list of types (names work too) or `None` for every type, and returns a `Subscription`
with `unsubscribe()`. An `Event` is frozen: `type`, `time` (s; arrivals are interpolated
inside the step), `step`, `vehicle` (id), `link` (lane or connector id), `intersection`
(id), each None when the event has none, and `data` (`phase_changed`: `{"phase": index,
"forced": bool}`). The types are `vehicle_departed`, `vehicle_inserted`,
`vehicle_entered_link`, `vehicle_exited_link`, `vehicle_stopped`, `vehicle_resumed`,
`vehicle_arrived`, `vehicle_removed`, `vehicle_teleported` and `phase_changed`.

Callbacks run after the step has completed, in event order. Control calls they make take
effect in the next step, and an exception they raise propagates out of `step()`/`run()`
with the state already consistent. Events nobody subscribed to are never built, so
subscribing costs nothing for other types. Subscriptions survive `reset()`;
`sim.events.counts()` (events per `EventType`, every type) restarts at each reset.

## Snapshots and digests

```python
from urbanflow import Simulation, Snapshot, bundled

sim = Simulation.from_scenario(bundled("single_intersection"), seed=7)
sim.run(until=300)
snap = sim.snapshot()
sim.run(until=400)
after = sim.state_digest()
sim.restore(snap)  # back at t = 300
sim.run(until=400)
assert sim.state_digest() == after  # the same 100 steps again

path = snap.save("t300.ufs")  # engine state only: state.npz + state.json in a zip
other = Simulation.from_scenario(bundled("single_intersection"), seed=7)
other.restore(Snapshot.load(path))  # the summary accumulators restart (logged warning)
other.run(until=400)
assert other.state_digest() == after
```

- `snapshot()` returns a `Snapshot(state, metrics, scenario_hash, config, time)`: deep
  copies of the engine state and of the summary accumulators; `step_count` is also
  available. A snapshot can be restored many times. After an exception interrupted a
  step (e.g. Ctrl-C), `snapshot()` and `state_digest()` raise `SimulationError`: the
  half-applied step cannot be saved; `reset()` or `restore()` an earlier snapshot.
- `restore(snapshot)` continues from it; `SimulationError` if it comes from another
  scenario or config (the message names the differing config fields), from a simulation
  built with other signal controllers (`controllers=`, named per intersection), or if the
  state is malformed (every index is range-checked). It clears the corrupted flag and
  every per-step cache.
- `Snapshot.save(path)` / `Snapshot.load(path)` store the engine state without pickle;
  the summary accumulators stay in memory, so a loaded snapshot restarts them: travel
  times, event counts and `vehicles.ids("arrived")` cover only what happens after the
  restore (and `ids("removed")` then also lists the earlier arrivals). The vehicle counts
  of the summary are engine counters and continue.
- `state_digest()` is the sha256 hex of the canonical engine state: equal digests on one
  platform mean equal states and equal futures
  ([Simulation model](simulation-model.md#determinism) lists what it covers).

## Results

```python
from urbanflow import Simulation, SimulationResult, generate

result = Simulation(generate("single_intersection", kind="uncontrolled"), duration=300).run()
print(result)  # aligned summary table
result.save("out/run1")  # out/run1/result.json
again = SimulationResult.load("out/run1")
assert again.summary["vehicles.arrived"] == result.summary["vehicles.arrived"]
```

`SimulationResult` is frozen: `scenario_name`, `scenario_hash`, `config`, `seed`,
`sim_time`, `steps`, `wall_time`, `interrupted`, `summary` (flattened keys such as
`travel_time.mean`; NaN where undefined), `event_counts`, `provenance` (urbanflow, numpy,
python, platform, machine), `run_id` and `controllers` (the configured signal controller
of each signalised intersection, `{"type", "params"}`: the scenario's or the
`controllers=` override, so runs that differ only in their controllers stay
distinguishable) and `state_digest` (the engine state digest at the end of the run).
`to_dict()` is JSON-ready (NaN becomes null);
`save(directory)` / `load(directory)` round-trip losslessly.

## Errors

| error | when |
|---|---|
| `ScenarioValidationError` | the scenario or the deep checks found errors (`.errors`, `.warnings`, `to_dict()`) |
| `ConfigError` | an invalid configuration value or option |
| `NotFoundError` | an unknown file, id or name (a `LookupError`, with a hint) |
| `SimulationError` | stepping when done, closed or corrupted; restoring a snapshot of another scenario or config; features not available yet |
| `CommandError` | an invalid control command (a `ValueError`) |
| `InvariantViolation` | a `debug_checks` invariant failed |
