# Simulation model

The engine (`urbanflow.engine.Engine`) advances the whole network from t<sub>n</sub> = n·dt
to t<sub>n+1</sub> in one synchronous step. Decisions read the state at t<sub>n</sub>, so
the order in which vehicles are stored never changes the result. Time is always
`step_count · dt`; it is never accumulated.

## The step pipeline

| # | Sub-step | What happens | Events |
|---|---|---|---|
| 0 | begin | clear the event buffer; handles freed in the last step become reusable; events of commands issued since the last step | `vehicle_departed`, `vehicle_removed` (commands) |
| 2 | spawn | flows and trips due in this step get a table row and wait in their first road's insertion queue | `vehicle_departed` |
| 3 | insert | queue heads enter their lanes where the safe-speed rule allows (at most one per lane, FIFO per road, `max_vehicles` respected) | `vehicle_inserted`, `vehicle_entered_link` |
| 4 | leaders | leader and gap of every vehicle, across links, merges and diverges | – |
| 6 | intersections | admission at the stop lines (below) | – |
| 7 | longitudinal | IDM against the leader and against the virtual obstacle, speed-limit anticipation, the safe-speed cap and the ballistic update ([vehicle model](vehicle-model.md)) | – |
| 8 | advance | positions move; vehicles hop lane → connector → lane, possibly several links in one step; arrivals | `vehicle_exited_link`, `vehicle_entered_link`, `vehicle_arrived` |
| 9 | bookkeeping | halting, waiting times, stops, distance, free-flow time; arrived vehicles are freed | `vehicle_stopped`, `vehicle_resumed` |
| 10 | finish | `step_count += 1`; invariant checks | – |

Sub-steps 1 (traffic signals) and 5 (lane changes) arrive with signalised intersections and
lane changing. Until then the engine refuses scenarios with signalised intersections or
transit lines with an error naming them.

Every event of a step carries `step = step_count` after the step. Spawn, insertion and
command events are stamped with the step's start time; link crossings and arrivals with
their exact time inside the step (interpolated from the step's constant acceleration); halting
transitions with the step's end time.

## Leaders

Vehicles are sorted by (link, position, uid); the next vehicle on the same link is the
leader. Beyond the link end the search follows the planned route while the distance is below
max(200 m, v²/(2b) + vT + s<sub>0</sub>) and stops at the first stop line the vehicle is not
admitted to cross. Vehicles leaving the same lane on different connectors follow each other
until their paths separate (diverge), and vehicles on connectors that end on the same lane
are ordered by their distance to that lane (merge). The nearest candidate wins.

## Unsignalised intersections

A vehicle on a lane that is not the last road of its route must be **committed** before it
may cross the stop line. Each step, the first uncommitted vehicle of every lane is a
candidate. Candidates are considered in the order (rank, then earliest expected arrival,
then uid): rank 1 at uncontrolled intersections (first come, first served), 3/2/1 for
major straight or near-side turn / major far-side turn / minor at priority intersections.

1. Farther than its decision distance v²/(2b) + v·dt + 5 m, the vehicle approaches freely.
2. **Don't block the box**: the exit lane must have room for it, net of the space already
   reserved by committed vehicles heading there.
3. **Gaps**: for every conflict zone (crossing or merging) on its path, the time window in
   which it would occupy the zone, from an optimistic and a pessimistic arrival estimate
   widened by 1 s, must not overlap the window of any vehicle that has not cleared the
   conflicting connector, any committed vehicle heading for it, or (only for a higher-rank
   conflicting movement) the nearest uncommitted vehicle of that approach.
4. If everything passes the vehicle commits and reserves its space on the exit lane.
   Otherwise it stops at the line (a virtual obstacle 0.5 m before it) if it still can with
   its emergency deceleration, and is force-committed (counted in `forced_commits`) if it
   cannot.

A vehicle whose lane has no connector toward its next road waits at the lane end until it
can change lanes.

## Invariants

After every step the engine checks that positions and speeds are finite and speeds are
non-negative. With `SimulationConfig.debug_checks = True` it also checks, every step:

| Id | Check |
|---|---|
| I3 | every vehicle lies on its link |
| I4 | no vehicle overlaps its leader (the leader search above, on the moved state) |
| I6 | generated = waiting + running + arrived + removed |
| I7 | every vehicle's link matches its route |
| I8 | nobody crossed a stop line without a commit |
| I11 | vehicle ids, uids and slots are consistent |

A failure raises `InvariantViolation(rule, step, uids, details)`. The counters
`safety_cap_violations` and `forced_commits` record the rare cases in which a vehicle had
to exceed its planned deceleration.

## Commands

`engine.commands` (in Python: `sim.vehicles.add/remove/set_speed`) applies `add_vehicle`, `remove_vehicle` and `set_speed` between steps;
their effects appear in the next step, and every call is appended to `command_log` as
`(step, name, args)`. API vehicles are named `api.0`, `api.1`, … unless an id is given,
and draw their speed factor from the `vehicle_params` random stream.

## Running a simulation

`urbanflow.Simulation` wraps the engine (see [Python API](python-api.md)). Its constructor
validates the scenario with `urbanflow.check` (registered model and router names, model
parameters, the network compile diagnostics), compiles the network once, builds the engine
and resets it to `config.seed`.

- **Time.** `time = step_count · dt`. With a `duration` the run ends after
  ⌊duration / dt⌋ steps (`end_time`); `done` is true from then on and `step()` raises
  `SimulationError` until `reset()`. With `duration = None` the run is `done` once the
  network has *drained*: every flow and trip has spawned and no vehicle is running or
  waiting to be inserted (`is_drained()`, which ignores the time limit).
- **`run(until=…)`** steps until the first step boundary at or after `until` (or for
  `duration` more seconds), never past `end_time`.
- **Control timing.** Commands issued between steps (`sim.vehicles.add`, `remove`,
  `set_speed`) take effect in the next step, in call order.
- **Views.** `sim.vehicles[...]`, `sim.lanes[...]` and `sim.roads[...]` return snapshot
  copies. `sim.state.*` and the per-lane vectors are read-only arrays computed once per
  step; keep a `.copy()` if you need them after the next `step()` or `reset()`.
- **Interruptions.** If an exception escapes a step (Ctrl-C mid-kernel, an invariant
  violation), the state may be half-updated: the simulation is marked corrupted and refuses
  to step until `reset()`. `get_results()` still works and reports `interrupted=True` after
  Ctrl-C.
- **Determinism.** The same scenario, resolved config and seed give identical states at
  every step and identical results; `reset(seed)` re-seeds every random stream, so
  repeating it reproduces the run from scratch.

### Summary

`sim.metrics.summary()` (and `SimulationResult.summary`) uses flattened keys:

| key | meaning |
|---|---|
| `vehicles.generated` | vehicles spawned so far (flows, trips, API) |
| `vehicles.inserted` | vehicles that entered the network |
| `vehicles.arrived`, `vehicles.removed`, `vehicles.teleported` | cumulative |
| `vehicles.en_route`, `vehicles.backlog` | running / waiting to be inserted now |
| `travel_time.mean`, `.median`, `.p95`, `.std` | arrival − insertion time over arrived trips, s (the arrival time is interpolated inside its step) |
| `throughput_vph` | 3600 · arrivals / elapsed time, veh/h |

Conservation holds at every step: generated = backlog + en route + arrived + removed.
Trips that depart before `metrics.warmup`, and arrivals before it, are left out of the
travel-time and throughput figures (the counts are totals). Travel-time statistics are NaN
until the first arrival.
