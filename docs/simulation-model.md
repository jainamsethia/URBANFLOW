# Simulation model

The engine (`urbanflow.engine.Engine`) advances the whole network from t<sub>n</sub> = n·dt
to t<sub>n+1</sub> in one synchronous step. Decisions read the state at t<sub>n</sub>, so
the order in which vehicles are stored never changes the result. Time is always
`step_count · dt`; it is never accumulated.

## The step pipeline

| # | Sub-step | What happens | Events |
|---|---|---|---|
| 0 | begin | clear the event buffer; handles freed in the last step become reusable (handles freed by commands one step later, after their `vehicle_removed` event); events of commands issued since the last step | `vehicle_departed`, `vehicle_removed` (commands) |
| 1 | signals | every signalised intersection in green asks its controller for a phase; the signal state machine then advances min-green, yellow and all-red timers and sets the movement states used by this step | `phase_changed` |
| 2 | spawn | flows and trips due in this step get a table row and wait in their first road's insertion queue | `vehicle_departed` |
| 3 | insert | queue heads enter their lanes where the safe-speed rule allows (at most one per lane, FIFO per road, `max_vehicles` respected) | `vehicle_inserted`, `vehicle_entered_link` |
| 4 | leaders | leader and gap of every vehicle, across links, merges and diverges | – |
| 6 | intersections | admission at the stop lines, then the zone locks inside intersections (below) | – |
| 7 | longitudinal | IDM against the leader and against the virtual obstacle, speed-limit anticipation, the safe-speed cap and the ballistic update ([vehicle model](vehicle-model.md)) | – |
| 8 | advance | positions move; vehicles hop lane → connector → lane, possibly several links in one step; arrivals; zone locks whose owner's rear has cleared are released | `vehicle_exited_link`, `vehicle_entered_link`, `vehicle_arrived` |
| 9 | bookkeeping | halting, waiting times, stops, distance, free-flow time; the watchdog (below); arrived vehicles are freed | `vehicle_stopped`, `vehicle_resumed`, `vehicle_teleported` |
| 10 | finish | `step_count += 1`; invariant checks | – |

Sub-step 5 (lane changes) arrives with lane changing. Until then the engine refuses
scenarios with transit lines with an error naming them.

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
are ordered by their distance to that lane (merge) once the follower is inside their merge
zone; before it the two paths are more than a vehicle width apart. A vehicle whose
connector is shorter than itself still occupies the end of the lane it came from, so a lane
is only free up to the rear of its last body. A long vehicle that has just left a merging
connector still hangs back over it; a vehicle on the other connector of the merge sees that
rear at most at the start of the merge zone until it is inside the zone itself. The nearest
candidate wins.

## Unsignalised intersections

A vehicle on a lane that is not the last road of its route must be **committed** before it
may cross the stop line. Each step, the first uncommitted vehicle of every lane is a
candidate. Candidates are considered in the order (rank, then earliest expected arrival,
then uid): rank 1 at uncontrolled intersections (first come, first served), 3/2/1 for
major straight or near-side turn / major far-side turn / minor at priority intersections.

1. Farther than its decision distance v²/(2b) + v·dt + 5 m, the vehicle approaches freely.
2. **Don't block the box**: the exit lane must have room for it, net of the space already
   reserved by committed vehicles heading there (a vehicle leaving the exit lane whose rear
   still hangs back over its end takes room too). The room ends at the rear of the exit
   lane's last vehicle plus v²/(2b<sub>emerg</sub>) of that vehicle, the distance it still
   travels even under emergency braking, so a moving platoon does not look parked.
3. **Gaps**: for every conflict zone (crossing or merging) on its path, the time window in
   which it would occupy the zone, from an optimistic and a pessimistic arrival estimate
   widened by τ = 0.5 s on each side, must not overlap the window of any vehicle that has
   not cleared the conflicting connector (including one whose front already reached the
   next lane while its rear, still holding its zone lock, is in the zone), any committed
   vehicle heading for it, or (only for a higher-rank conflicting movement) the nearest
   uncommitted vehicle of that approach.
4. If everything passes the vehicle commits and reserves its space on the exit lane.
   Otherwise it stops at the line (a virtual obstacle 0.5 m before it) if it still can with
   its emergency deceleration, and is force-committed (counted in `forced_commits`) if it
   cannot.

The arrival estimates bound the true time from both sides: the optimistic one accelerates
at the full a to max(v, v<sub>c</sub>), the pessimistic one at 0.75·a to the connector
cruise speed v<sub>c</sub>. The windows only order vehicles; the zone locks (below) keep
each conflict zone exclusive. A left turner stopped at the line of the bundled single
intersection therefore needs a lag of 6.8-7.4 s to the opposing through vehicle
(13.9 m/s) at the conflict zone, against the 5.5-6.1 s it physically needs to clear the
zone from rest with IDM's comfortable acceleration a = 1.5 m/s². The Highway Capacity
Manual's critical headway for permitted left turns is 4.1-5.5 s, for drivers who
accelerate harder.

A vehicle whose lane has no connector toward its next road waits at the lane end until it
can change lanes.

## Signalised intersections

A signal program lists phases of protected (G) and permissive (g) movements; every other
movement is red. Switching from phase p to q shows yellow on the movements that lose green,
then all-red, then green q (at once if nothing loses green). Stage lengths are rounded up
to whole steps. The controller (`fixed_time` by default, with an optional `offset`;
`external` switches on request) only chooses phases: min-green, yellow and all-red are
enforced for every controller.

At a signalised intersection a movement's rank is its signal state (G 3, g 2), so a
permissive movement yields to the opposing protected one, and admission starts with the
light:

- **Red:** the vehicle stops at the line if it can with its emergency deceleration;
  otherwise it is force-committed and counted in `red_runs` (only possible after a forced
  `set_phase` or with yellow = 0).
- **Yellow:** it stops if that needs at most 3 m/s². Otherwise (the dilemma zone) it
  crosses if the exit lane has room and the gaps are free, stops if it still can, and is
  force-committed (`forced_commits`) if it cannot.
- **Green:** as at unsignalised intersections.

Committed vehicles never check the light again, which is what the intergreen clears.

Vehicles looking past a stop line (on a connector, or committed) also brake for the next
signal on their route with the same tests: for yellow only if stopping there needs at most
3 m/s², for red only if they can still stop with their emergency deceleration. Otherwise
they drive on and admission decides at that stop line.

**End-of-green clearing.** A permissive left turner that shares its lane with straight
traffic can wait through a whole green when the opposing flow never leaves it a large
enough gap, and then block the straights behind it for the next cycle as well. Real
drivers clear such a queue head at the end of the green ("sneakers"; the Highway Capacity
Manual assumes 1-2 per cycle and approach). While a movement that was permissive (g)
shows the yellow ending that green, the first uncommitted vehicle of each of its approach
lanes commits if it has waited at the stop line since before the yellow (it was held
there, within 5 m of the line, in the last green step), is still within 5 m of the line
and the exit lane has room for it. A vehicle that arrives or brakes for the line during
the yellow is not a sneaker: it stops as usual. The gap check is skipped, and at most
`SNEAKERS_PER_PHASE` = 1 vehicle per lane and yellow does so. Sneakers are admitted after
every other candidate of the step, so an opposing vehicle in its dilemma zone commits
first. This is safe because the zone locks (next section), not the gap windows, keep the
conflict zones exclusive:

- the sneaker requests its zones like any committed vehicle, and is held at its first
  zone while an opposing lock, or an opposing vehicle that committed earlier and arrives
  first (rule b), is in the way;
- an opposing vehicle that reaches its line later in the yellow sees the sneaker in its
  gap check and stops. Only one that can no longer stop even with its emergency
  deceleration is force-committed; its force grant revokes the sneaker's grant if the
  sneaker has not entered the zone yet (it is slow and can stop), and any overlap left is
  counted in `zone_conflicts`;
- the next phase's vehicles commit after it, so they wait for its locks by rules (a) and
  (b), and the deadlock argument below applies unchanged: every wait follows decreasing
  commit order or a lock whose owner progresses.

## Zone locks

Admission decides who may cross a stop line; zone locks decide who may enter a conflict
zone. All their state lives in vehicle columns (`committed`, `commit_seq`, `granted`,
`forced`, `lock_conn`), so it is part of every snapshot. A vehicle's front and rear
(pos − length) are mapped onto its connector across link ends: minus the approach lane's
length while it is still on the approach lane, plus the connector's length once it is on
the exit lane. Its *remaining* zones are the crossing and merging zones of its connector
whose end its rear has not passed.

- **Request.** Committed vehicles are processed in commit order. Once the distance to its
  first remaining zone is at most its decision distance (plus a·dt²/2, the extra distance
  one step of acceleration covers), a vehicle requests **all** its remaining zones at
  once. A connector without crossing or merging zones is granted at commit.
- **Grant** iff (a) no zone of the set is locked by another vehicle, i.e. the owner's rear
  has not passed that zone's end on the owner's connector; (b) no committed, not yet
  granted conflicting vehicle that committed earlier is predicted to arrive before this one
  clears (the windows of admission); (c) its leader from the same approach lane (on the
  lane, its connector or a diverging sibling) is not committed but ungranted; and (d) it
  no longer holds the lock of the connector it came from. A grant sets `granted` and
  `lock_conn` to the connector.
- **Not granted:** a virtual obstacle at the first remaining zone's entry (the front stops
  0.5 m before it), and `held`.
- **Release.** Each zone is free again once the owner's rear has passed its end;
  `lock_conn` returns to −1 after the last one (checked after the advance sub-step), and at
  once on arrival, removal and teleport.
- **Force grants.** A force-committed vehicle, and a requesting vehicle that can no longer
  stop before its first zone (v²/(2·dist) > b<sub>emerg</sub>), is granted at once and
  marked `forced`. The grant **cascades** to the committed but ungranted vehicles ahead of
  it from the same approach lane (they are granted and forced too; it would run into them
  otherwise) and **revokes** the grant of every other owner of an overlapping zone that
  has not entered any of its zones and can still stop before the first: its lock is
  released, it is held this step and requests again from the next. Overlaps that no
  revocation can remove (the owner is inside the zone, cannot stop or is forced itself)
  are counted in `zone_conflicts`; they are the only exemption from the zone-exclusivity
  invariant I5.

**Why this is deadlock-free inside intersections.**

1. A granted vehicle is never held again on its connector: a grant is only revoked before
   the vehicle enters its zones. Only car-following can stop it, and its leader chain ends
   in space reserved for it on the exit lane (don't block the box; insertion and teleports
   honour reservations). So every lock is released in finite time.
2. An ungranted committed vehicle waits for one of: a lock owner (which progresses by 1);
   a vehicle that committed earlier (rules b and c); or the vehicle ahead of it on its
   approach lane, connector or a diverging sibling (car-following), which committed
   earlier too, because vehicles leave a lane in order and nothing may cut in ahead of a
   committed vehicle.
3. The commit order strictly decreases along the waits of the second and third kind, so
   no cycle of waits can form. A force grant waits on nothing, so revocations cannot close
   a cycle either.

This does not cover a vehicle force-committed into a full exit lane, or gridlock of the
network itself (queues spilling back around a block), which is real traffic behaviour. The
watchdog handles those.

**Watchdog.** `stuck_time` counts continuous halting while held (at a stop line, a lane
end or a conflict zone), or while halting on a connector. With `deadlock_timeout > 0`
(default 300 s; 0 turns it off) every vehicle stuck that long is teleported, connector
vehicles first: it releases its locks and joins the back of the insertion queue of its
next road (or arrives if it is on its last road). Each teleport is counted
(`vehicles.teleported`) and emitted as `vehicle_teleported`.

## Road capacity

`CompiledNetwork.road_capacity_vph` (`sim.roads[...].capacity_vph`) and the entry capacity
of the validation warning W501 are the road's lane count times the peak of the IDM
equilibrium flow of one lane at the road speed limit v<sub>0</sub>:

q<sub>max</sub> = 3600 · max<sub>0 &lt; v &lt; v<sub>0</sub></sub> v / (s<sub>e</sub>(v) + ℓ),
with s<sub>e</sub>(v) = (s<sub>0</sub> + vT) / √(1 − (v/v<sub>0</sub>)<sup>δ</sup>)

for the built-in car (T = 1.1 s, s<sub>0</sub> = 2 m, ℓ = 5 m, δ = 4); W501 uses the
flow's vehicle types, averaged by headway. The maximum is taken on a grid of 1000 speeds
(relative error below 10<sup>−5</sup>): about 1790 veh/h per lane at 13.89 m/s.

## Invariants

After every step the engine checks that positions and speeds are finite and speeds are
non-negative. With `SimulationConfig.debug_checks = True` it also checks, every step:

| Id | Check |
|---|---|
| I3 | every vehicle lies on its link |
| I4 | no vehicle overlaps its leader (the leader search above, on the moved state) |
| I5 | no two vehicles on different connectors are inside the same crossing or merging zone (bodies mapped back onto the locked connector); pairs with a forced vehicle are exempt and counted in `zone_conflicts` |
| I6 | generated = waiting + running + arrived + removed |
| I7 | every vehicle's link matches its route |
| I8 | nobody crossed a stop line without a commit |
| I9 | every vehicle held at a stop line, lane end or conflict zone is still behind that point, unless the obstacle appeared in this step when it could no longer stop |
| I10 | signal states are r, y, g or G, and no movement goes from green to red without yellow (unless a forced `set_phase` did it or the program has no yellow) |
| I11 | vehicle ids, uids and slots are consistent |

A failure raises `InvariantViolation(rule, step, uids, details)`. The engine counters
`safety_cap_violations`, `forced_commits`, `red_runs` and `zone_conflicts` record the
rare cases in which a vehicle had to exceed its planned deceleration or share a conflict
zone; [Troubleshooting](troubleshooting.md) explains each rule and counter.

## Commands

`engine.commands` (in Python: `sim.vehicles.add/remove/set_speed/set_route`) applies
`add_vehicle`, `remove_vehicle`, `set_speed` and `set_route` between steps; their effects
appear in the next step, and every call is appended to `command_log` as
`(step, name, args)`. A rejected call raises `CommandError` or `NotFoundError` (with a
did-you-mean hint) and changes nothing. API vehicles are named `api.0`, `api.1`, … unless
an id is given, and draw their speed factor from the `vehicle_params` random stream.

`set_route(id, roads)` replaces the rest of a route. The new roads must start at the
vehicle's current road (its first road while it waits for insertion; the road its
connector leads to while it crosses an intersection) and be connected. The lane plan is
recomputed; if the current lane cannot reach the new next road, the vehicle needs a
mandatory lane change (until lane changing arrives it waits at the lane end, and the
watchdog eventually teleports it). A vehicle already admitted across its next stop line
keeps its connector, because its reserved exit space and zone locks refer to it, so its new
route must continue through that connector's road; otherwise the command fails with
`vehicle <id> is committed to connector <c>; the new route must continue through it`.

Signal commands take an intersection and a phase (index or id): `request_phase` (needs a
controller that accepts requests, such as `external`; min-green, yellow and all-red are
honoured), `set_phase` (an immediate jump without intergreen, for tests and setup),
`hold_phase` / `release` (a manual override with any controller: the configured controller
is parked and resumes from the current phase on release) and `set_controller`. In Python
they are `sim.signals.request_phase(...)` and so on; see
[Traffic signals](traffic-signals.md).

## Running a simulation

`urbanflow.Simulation` wraps the engine (see [Python API](python-api.md)). Its constructor
validates the scenario with `urbanflow.check` (registered model, router and signal
controller names, model and controller parameters, including the `controllers=`
overrides, the network compile diagnostics, the flow rules E506/E507 for the final `dt`
and `duration`, integer depart lanes that cannot reach the route's next road), compiles the network once, builds the engine
and resets it to `config.seed`.

- **Time.** `time = step_count · dt`. With a `duration` the run ends after
  ⌊duration / dt⌋ steps (`end_time`); `done` is true from then on and `step()` raises
  `SimulationError` until `reset()`. With `duration = None` the run is `done` once the
  network has *drained*: every flow and trip has spawned and no vehicle is running or
  waiting to be inserted (`is_drained()`, which ignores the time limit).
- **`run(until=…)`** steps until the first step boundary at or after `until` (or for
  `duration` more seconds), never past `end_time`.
- **Control timing.** Commands issued between steps (`sim.vehicles.add`, `remove`,
  `set_speed`, and the `sim.signals` commands) take effect in the next step, in call
  order. A controller asked for a phase between steps sees it in the next step's signal
  sub-step, so a switch that starts yellow shows it during that step.
- **Views.** `sim.vehicles[...]`, `sim.lanes[...]` and `sim.roads[...]` return snapshot
  copies. `sim.state.*` and the per-lane vectors are read-only arrays computed once per
  step; keep a `.copy()` if you need them after the next `step()` or `reset()`.
- **Interruptions.** If an exception escapes a step (Ctrl-C mid-kernel, an invariant
  violation), the state may be half-updated: the simulation is marked corrupted and refuses
  to step until `reset()`. `get_results()` still works and reports `interrupted=True` after
  Ctrl-C.
- **Determinism.** The same scenario, resolved config and seed give identical states at
  every step and identical results; `reset(seed)` re-seeds every random stream, so
  repeating it reproduces the run from scratch (see [Determinism](#determinism)).
- **After each step** the facade first drops every per-step cache, then adds the step's
  events to the summary, and calls the event subscribers last (`sim.events`). Callbacks
  therefore see the finished step; control calls they make apply to the next step, and an
  exception they raise propagates out of `step()` with the state consistent (the
  simulation is not marked corrupted).

## Determinism

Runs are reproducible bit for bit on one platform. Every source of nondeterminism is
removed by construction:

| Source | How it is neutralised |
|---|---|
| random draws | only named streams, `PCG64(SeedSequence(seed, spawn_key=utf8(name)))`: `flow:{id}`, `trip:{id}`, `vehicle_params`, `controller:{intersection}`, `model:{name}`. A flow draws type, route, speed factor and random lane, in that order, only from its own stream, so adding a flow or changing a controller never changes another flow's vehicles |
| iteration order | sorts are `lexsort` with a uid tie-break or stable; loops run over sorted index arrays or dicts built in sorted order; no set is iterated for output; `hash()` is never used |
| sequential decisions | admission runs in a total order (rank, time to the line, uid), zone grants in commit order |
| slot reuse | outputs are keyed by uid; the free list is LIFO and freed slots wait a step |
| time | `t = step_count · dt`, never summed; signal timers compare with a 10<sup>−9</sup> tolerance and stage ends fall on step boundaries |
| routing ties | the road graph is built in sorted order, so Dijkstra's ties are fixed |
| threads | the engine is single-threaded; parallel runs are separate processes built from JSON |
| floating point | the same numpy build on the same platform is bit-identical; other CPUs or operating systems may differ in the last bit (SIMD paths of transcendental functions) and then diverge, so cross-platform comparisons use metrics with tolerances |

`sim.state_digest()` is a sha256 over the whole runtime state in a fixed byte order:
the step and every counter (`next_uid`, the commit counter, generated, arrived, removed,
teleported, `forced_commits`, `red_runs`, `zone_conflicts`, `safety_cap_violations`);
every live or waiting vehicle in uid order with every vehicle column packed little-endian
(NaNs canonicalised, so zone locks, commits and holds are included); the routes; the
insertion queues as uid lists; the signal state arrays and the per-lane end-of-green
eligibility and detector state; and the canonical JSON of the controllers (active and parked by a manual
hold, with their parameters and state), the hold flags, the spawners, the router, every
random stream and the pending command events. Two runs with equal digests are in the same
state and have the same future. Engine slots and freed rows are not part of it, and
neither is the command log. `SimulationResult.state_digest` is the digest at the end of the
run, and `urbanflow run` prints it.

## Snapshots

`sim.snapshot()` deep-copies the runtime state (everything the digest covers, plus the
slot lists and the command log) and the run-summary accumulators; `sim.restore(snapshot)`
puts it back, after which the run continues exactly as it did after the snapshot was
taken: snapshot at t, step k, restore, step k again gives the same digests. A snapshot can
be restored any number of times, also into another `Simulation` of the same scenario
(equal content hash) and config; anything else raises `SimulationError` naming the
difference. Restoring clears the corrupted flag and every per-step cache. Signal
controllers are restored from `{type, params, state_dict}`: the simulation's own instance
is reused when type and parameters match (so unregistered controller classes work), other
ones are created from the registry.

`snapshot.save(path)` writes the engine state as a zip of `state.npz` (standard `.npy`
members, read back through a strict reader that never unpickles) and `state.json`;
`Snapshot.load(path)` reads it. The run-summary accumulators are not saved: restoring a
loaded snapshot restarts them (travel times and event counts count from the restore) and
logs a warning. The engine counters (generated, arrived, …) are part of the state.

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
