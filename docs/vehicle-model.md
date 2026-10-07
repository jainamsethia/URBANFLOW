# Vehicle model

Every vehicle is a row of a structure-of-arrays table (`urbanflow.vehicles.VehicleTable`)
and every model below is a vectorised NumPy kernel over all running vehicles. This page
gives the formulas the engine applies each step: car-following (IDM), the safe-speed cap
that makes collisions impossible, the ballistic position update, and how vehicles are
created and inserted.

## Identity

The engine stores vehicles in reusable slots (*handles*). A handle freed during a step is
reused only after the next step starts, so it never names two vehicles within one step.
Everything outside the engine (frames, replays, events, views) identifies a vehicle by its
**uid**, a spawn serial that is never reused within a run and restarts at 0 on `reset()`.
Vehicle ids are `"{flow}.{k}"` for flows and the trip id for trips.

## Car-following: IDM

For a follower with speed v, desired speed v⁰, bumper-to-bumper gap s and approach rate
Δv = v − v<sub>lead</sub> (Treiber, Hennecke & Helbing 2000):

- s\* = s<sub>0</sub> + max(0, vT + vΔv / (2√(ab)))
- a<sub>IDM</sub> = a · [max(1 − (v/v⁰)<sup>δ</sup>, −b/a) − (s\*/s)²]

The free-road term is bounded below by −b (as in the IIDM), so a vehicle above its desired
speed slows down comfortably instead of braking hard. Without a leader (s = ∞) only the
free-road term remains; a gap of 0.01 m or less gives −b<sub>emerg</sub>; the result is
clamped to [−b<sub>emerg</sub>, a]. A positive acceleration never takes the speed past
max(v, v⁰) within one step.

The desired speed is v⁰ = min(max_speed, f · speed limit of the current link), where f is
the vehicle's speed factor drawn at spawn.

Custom models implement the `CarFollowingModel` protocol and register with
`@register_car_following("name")`; select them with `SimulationConfig.car_following`.
Their `Params` model validates `vehicle_types[].model_params` (unknown names are **E903**)
and each field becomes a per-vehicle array of `ParamArrays`.

## Safe-speed cap

The cap rests on four assumptions:

- **A1** every vehicle can always brake at its emergency deceleration b<sub>emerg</sub>;
- **A2** decisions have a reaction delay of one step (they read the state at the start of
  the step);
- **A3** a leader's speed drops by at most b<sub>emerg,L</sub>·dt per step, and it stops
  within v<sub>L</sub>² / (2 b<sub>emerg,L</sub>);
- **A4** new neighbours appear only through insertion, lane changes and intersection
  admission, and each of those applies the same inequality; vehicles leaving a lane on
  different connectors stay each other's leaders while their paths overlap.

Whatever the model asks for, the next speed v′ must satisfy

(v + v′)/2 · dt + v′² / (2 b̂) ≤ C, with C = g + v<sub>L</sub>² / (2 b<sub>L</sub>) − s<sub>m</sub>

so that after this step the follower can still stop behind the leader's worst-case stopping
point, keeping s<sub>m</sub> = 0.5 m. C is the minimum over the leader and every obstacle
(a stop line or lane end counts as a standing leader with s<sub>m</sub> = 0). The planning
deceleration b̂ is the smallest emergency deceleration of all vehicle types: it never
exceeds any leader's b<sub>L</sub>, which keeps the gap concave while both brake, so the
check at the end points holds at every intermediate time. With a harder planning
deceleration than the leader's, the gap would be convex in time and could dip below zero
before both stop (v = 20, v<sub>L</sub> = 10, b<sub>F</sub> = 8, b<sub>L</sub> = 2 dips
8.3 m below the final gap). The largest admissible speed is
the positive root r = b̂ · (−dt/2 + √(dt²/4 + (2/b̂)(C − v·dt/2))).

**Why it is collision-free.** If v² / (2b̂) ≤ C holds for the leader and every obstacle at
the start of a step, choosing v′ ≤ r keeps it true at the start of the next one (A1–A3), and
A4 makes it hold for every new neighbour. By induction no vehicle can hit its leader or run
over a stop line; in debug mode the engine checks this every step (invariants I4 and I8).

## Ballistic update

With v\* = min(v + a·dt, max(0, r)) (Treiber & Kanagaraj 2015):

- **v\* > 0:** v′ = max(v\*, v − b<sub>emerg</sub>·dt) and Δx = (v + v′)/2 · dt. If the
  floor binds, the step counts a *safety cap violation*.
- **otherwise** the vehicle stops inside the step: Δx = v² / (2b\*) with
  b\* = min(b<sub>emerg</sub>, max(−a, v/dt, v² / (2C), 10⁻⁹)) and v′ = 0; it is a
  violation when v² / (2C) > b<sub>emerg</sub>. A vehicle faster than
  b<sub>emerg</sub>·dt cannot stop in one step and takes the first branch instead.

Speeds never go negative, and no deceleration exceeds b<sub>emerg</sub>. Violations only
happen after something broke the invariant v² / (2b̂) ≤ C (a forced speed, a teleport, a
forced commit); they are counted, never hidden.

## Lane changing: MOBIL

Each step (after the leader search, before intersection admission) every running vehicle
on a lane that is not committed to its stop line, off its 3 s cooldown and not dwelling may
change to an adjacent lane, keeping its relative position on the road.

- **Mandatory** changes: the current lane has no connector to the next road of the route.
  The vehicle moves toward the nearest lane that has one, up to the stop line; until a
  gap appears it waits at the lane end. **Discretionary** changes (overtaking) happen only
  farther than 20 m from the stop line and never leave a lane that serves the route.
- **Safety**: positive gaps to the new leader and follower, the space reserved by
  vehicles committed into the target lane left free, no cut-in ahead of a committed
  follower, the safe-speed inequality for the vehicle behind its new leader and for the
  new follower behind it, and the new follower's deceleration within `lc_safe_decel`.
- **Incentive** (Kesting, Treiber & Helbing 2007):
  ã<sub>i</sub> − a<sub>i</sub> + p[(ã<sub>n</sub> − a<sub>n</sub>) + (ã<sub>o</sub> − a<sub>o</sub>)]
  &gt; Δa<sub>th</sub> − β, with β = min(10, 1.0 · 200 / max(d, 1)) m/s² for mandatory
  changes (urgency grows toward the stop line) and p = 0 for mandatory changes closer
  than 50 m.
- **Yielding to emergency vehicles**: a vehicle with an emergency vehicle behind it on
  its lane within 100 m (front to front) moves over: it may change up to the stop line,
  with β = 10 m/s² and p = 0, still only into a lane that serves its route and with all
  the safety checks. No discretionary change ends within 100 m ahead of an emergency
  vehicle on the target lane. In a standing queue there is usually no gap to move into,
  so the emergency vehicle waits like everyone else (there is no shoulder or rescue lane).
- Changes execute one by one (mandatory first, then by incentive, then uid); the
  neighbours of an executed change decide again next step. The change is instantaneous
  for the physics; a lateral offset fades out over 2 s for drawing only. There are no
  shadow vehicles, so lane counts are exact. `lane_changing=False` turns it off.

## Parameters

Defaults are the built-in `car`; `bus`, `truck` and `emergency` override some of them (see
[Scenario format](scenario-format.md)).

| Symbol | Field | Default | Unit |
|---|---|---|---|
| a | `accel` | 1.5 | m/s² |
| b | `decel` | 2.0 | m/s² |
| b<sub>emerg</sub> | `emergency_decel` | 6.0 | m/s² |
| T | `headway` | 1.1 | s |
| s<sub>0</sub> | `min_gap` | 2.0 | m |
| δ | `model_params.delta` | 4 | – |
| v<sup>max</sup> | `max_speed` | 36.1 | m/s |
| f | `speed_factor` | N(1.0, 0.1) clipped to [0.8, 1.2] | – |
| p | `politeness` | 0.2 | – |
| Δa<sub>th</sub> | `lc_threshold` | 0.1 | m/s² |
| b<sub>safe</sub> | `lc_safe_decel` | 4.0 | m/s² |
| – | `YELLOW_MAX_DECEL` | 3.0 | m/s² |
| – | `TURN_LATERAL_ACCEL` | 2.0 | m/s² |
| τ | `GAP_ACCEPT_MARGIN` | 0.5 | s |
| – | `ETA_END_ACCEL_FACTOR` | 0.75 | – |
| – | `LC_MANDATORY_BIAS` | 1.0 | m/s² |
| s<sub>m</sub> | `SAFETY_MARGIN` | 0.5 | m |

Lower-case names are vehicle-type fields; upper-case names are constants in
`urbanflow.core.constants`. p, Δa<sub>th</sub> and b<sub>safe</sub> are the MOBIL
lane-change parameters; the yellow threshold is the dilemma-zone rule of signalised
junctions; the turn lateral acceleration sets connector speed limits
√(a<sub>lat</sub>·R<sub>min</sub>); τ and the late-arrival factor shape the conflict windows
of intersection admission.

The car's time gap T = 1.1 s is calibrated on queue discharge at a signal: a queue released
at green crosses the stop line at about 1580-1640 veh/h/lane (dt 1.0-0.2), inside the
1500-1900 veh/h/lane band of saturation flows. The IDM equilibrium flow
v / (s<sub>e</sub>(v) + ℓ), with s<sub>e</sub>(v) = (s<sub>0</sub> + vT) / √(1 − (v/v<sub>0</sub>)<sup>δ</sup>),
then peaks near 1800 veh/h/lane at 13.9 m/s. The simpler estimate
3600 / (T + (ℓ + s<sub>0</sub>)/v) ignores the (v/v<sub>0</sub>)<sup>δ</sup> term and
overstates it (T = 1.5 s measures only about 1420-1440 veh/h).

## Demand

A flow with headway h = 3600 / rate (or `period`) schedules its vehicles at

- **uniform:** t<sub>k</sub> = begin + k·h, computed directly (no drift);
- **poisson:** t<sub>k</sub> = t<sub>k−1</sub> + Exp(rate/3600), starting from begin;
- **binomial:** one vehicle with probability rate·dt/3600 in every step inside
  [begin, end);

while t<sub>k</sub> < end and k < count. A vehicle scheduled at t is created in step
⌈t/dt − 10⁻⁹⌉. Each flow draws from its own random stream `flow:{id}` (trips: `trip:{id}`)
in a fixed order per vehicle: type (`type_mix`), route (`routes`), speed factor, then the
`random` depart lane. Adding a flow, or changing a signal controller, therefore never
changes the vehicles of another flow (common random numbers). OD flows are routed by the
router (`shortest`: Dijkstra over free-flow time or length, cached per origin/destination).

## Lane choice and insertion

On each road a vehicle's valid lanes are those with a connector to its next road. Among its
lane's connectors it prefers one whose target lane is valid for the road after next, then
the smallest lane shift, then the lowest index; a vehicle on an invalid lane has no planned
connector and must change lanes.

New vehicles wait in one FIFO queue per first road. Each step, per road in index order, the
queue head tries one lane (`best`: most free space; `first`: lowest valid lane; `random`:
the lane drawn at spawn; a number: that lane) not yet used this step. It is inserted with
its rear at the lane start when

- the last vehicle's rear leaves room for its length and s<sub>0</sub> beyond the space
  reserved by vehicles committed to enter the lane;
- its depart speed satisfies v·dt + v² / (2b̂) ≤ C behind that vehicle (`"max"` takes the
  largest such speed, capped at v⁰; a numeric speed above it waits);
- every vehicle about to leave a connector into the lane can still follow it within its
  emergency deceleration.

Otherwise the head waits and blocks its road (FIFO). At most one vehicle enters each lane
per step; the number waiting is the *backlog*.
