# Troubleshooting

## Validation errors

`urbanflow validate scenario.json` (or `urbanflow.check(scenario)`) lists every problem
with a code, a JSON path and a message; errors (`E…`) stop a run, warnings (`W…`) do not.
The [scenario format](scenario-format.md#validation) page lists every code. The most
common ones in practice:

- **W501** a flow's rate exceeds the entry capacity of its first road (the IDM peak flow,
  about 1790 veh/h per lane at 50 km/h, see
  [Simulation model](simulation-model.md#road-capacity)): the vehicles that do not fit
  wait outside the network and show up as `vehicles.backlog`.
- **E806** a lane is shorter than a vehicle routed through it plus its minimum gap: it
  could never be admitted into that lane without blocking the junction.

## Invariant violations

With `debug_checks=True` (`Simulation(..., debug_checks=True)`, or `--debug-checks` on
`urbanflow run`) the engine checks the rules below after every step. I1 and I2 are always checked. A failure raises
`urbanflow.InvariantViolation` with:

- `rule`: the rule id (`"I5"`);
- `step`: the step count at which it was detected;
- `uids`: the uids of the vehicles involved (the `uid` field of each vehicle view);
- `details`: a short description, e.g. `1 pair(s) share a conflict zone`.

The message joins them: `invariant I5 violated at step 812 by vehicles uid 301, 344: 1
pair(s) share a conflict zone`. The simulation is then marked corrupted: `step()` refuses
to continue until `reset()`, while `get_results()` still works.

```python
import urbanflow
from urbanflow import InvariantViolation, Simulation

sim = Simulation(urbanflow.bundled("single_intersection"), debug_checks=True, duration=120)
try:
    sim.run()
except InvariantViolation as err:  # a model bug: report it with the scenario and seed
    print(err.rule, err.step, err.uids, err.details)
print(sim.metrics.summary()["vehicles.teleported"])
```

| Id | Rule | What a violation means |
|---|---|---|
| I1 | positions and speeds are finite | a numerical blow-up, usually from a custom car-following model returning NaN or inf |
| I2 | speeds are ≥ 0 | a custom model or a speed command produced a negative speed |
| I3 | every vehicle lies on its link (±1e-6 m) | the advance sub-step lost a vehicle between links |
| I4 | no vehicle overlaps its leader (gap ≥ −1e-6 m) | two bodies intersect on a lane, a connector or across a merge or diverge |
| I5 | no two vehicles of different connectors are inside one crossing or merging zone (by more than 1e-6 m) | a zone lock failed; overlaps with a force-granted vehicle are exempt and counted in `zone_conflicts` instead |
| I6 | generated = backlog + running + arrived + removed | a vehicle was created or lost without being counted |
| I7 | a vehicle's link matches its route | a route change left a vehicle on a road that is not on its route |
| I8 | nobody crosses a stop line or drives on a connector uncommitted | admission was bypassed |
| I9 | a held vehicle stays behind its obstacle (stop line, lane end or conflict zone) | a vehicle ignored the stop it was given; exempt when the obstacle appeared in this step and it could no longer stop |
| I10 | signal states are r, y, g, G, and green turns red only through yellow | a controller or the signal state machine skipped the yellow (forced `set_phase` and programs with `yellow = 0` are exempt) |
| I11 | ids, uids and table slots are consistent | internal bookkeeping of the vehicle table broke |

I1-I3 and I6-I11 should never fire with the built-in models: a violation is a bug, and
the scenario, the seed and the step are what is needed to reproduce it (runs are
deterministic). I4, I5 and I9 can also be provoked from outside the model, by
`sim.vehicles.set_speed` forcing a vehicle faster than it can stop, by a custom
car-following model that brakes harder than its type's `emergency_decel`, or by
`set_phase` jumping a signal to red in front of fast vehicles.

## Safety counters

These counters are always on (debug checks or not) and are attributes of
`urbanflow.engine.Engine`; teleports are also `vehicles.teleported` in
`sim.metrics.summary()` and the run summary of the command line. In a healthy run all of
them are 0 or rare.

| counter | counts | typical cause |
|---|---|---|
| `forced_commits` | vehicles that could neither commit normally nor stop before the line: yellow dilemma vehicles whose gap or exit check failed, and approaching vehicles surprised by a full exit lane | very short yellows, speed commands, dense traffic on a very short link after the junction |
| `red_runs` | vehicles force-committed at red because they could not stop | a forced `set_phase` or `yellow = 0` |
| `zone_conflicts` | overlaps a force grant could not remove by revocation (the other vehicle was already inside the zone or could not stop) | follows `forced_commits`; I5 exempts exactly these |
| `safety_cap_violations` | steps in which a vehicle had to brake harder than `emergency_decel` or was stopped at a line it had no right to cross | an obstacle appearing too close (a revoked grant is never one), speed commands, teleports into occupied space |
| teleports (`vehicles.teleported`) | vehicles moved by the watchdog after halting `deadlock_timeout` s | gridlock (below) |

## Gridlock and the watchdog

Intersections cannot deadlock internally (the argument is in
[Simulation model](simulation-model.md#zone-locks)), but a network can gridlock the way
real traffic does: queues spill back around a block until every vehicle waits for another.
Then halting vehicles stop being served and `stuck_time` grows. The watchdog
(`deadlock_timeout`, default 300 s; 0 turns it off) teleports every vehicle that has been
halting that long while held at a stop line, lane end or conflict zone, or while halting
on a connector: it moves to the insertion queue of its next road (or arrives on its last
road), and the run goes on.

A few teleports in a saturated scenario are expected. Many teleports, or a growing
backlog, usually mean:

- **demand above capacity:** check W501 and lower the flow rates, or add lanes;
- **exit spillback:** a short link after a junction fills up and don't-block-the-box
  admission holds the junction; lengthen the link or give the downstream junction more
  green;
- **a movement without green:** W301 warns about movements no phase serves; their
  vehicles wait at red until the watchdog moves them;
- **heavy opposing flow against a permissive turn:** permissive (g) left turns take gaps
  of about 7 s in the opposing stream and one vehicle per lane at the end of each green;
  give the turn a protected phase if the opposing flow leaves no gaps.

To see who is stuck, step the simulation and look at the vehicle views of `sim.vehicles`
(`waiting_time`, `speed`, `road`, `connector`).

## Snapshots and digests

- **`SimulationError: the snapshot is of another scenario`**: snapshots restore only into
  a simulation of a scenario with the same content hash (`Scenario.short_hash`); editing
  the scenario, even a flow rate, makes a new one.
- **`... another simulation config (it differs in: seed)`**: the resolved config must match
  too; the message names the fields. Build the target simulation with the same
  `SimulationConfig` and overrides (`snapshot.config` holds them).
- **`cannot restore the signal controller "X"`**: a snapshot records each controller as
  `{type, params, state_dict}`. The simulation's own instance is reused when type and
  parameters match; otherwise the controller is created from the registry, so a custom
  controller class installed by instance or factory must be registered
  (`@register_controller`) to be restored into a simulation that does not already run it.
  Controllers and routers with internal state implement `state_dict()` and
  `load_state_dict()`; state they keep elsewhere is not snapshotted.
- **Digests differ between two runs that should match**: first compare the resolved
  configs and scenario hashes (`SimulationResult.config`, `.scenario_hash`); then look for
  randomness outside the engine's streams (a custom controller or model calling
  `numpy.random` or `random` instead of `ctx.rng`) and for state kept in globals. Digests
  are only comparable on the same platform and numpy build: another CPU or operating
  system may round the last bit differently and diverge later, so compare metrics with a
  tolerance across machines.
- **Summary restarts after `Snapshot.load`**: files hold the engine state only. The
  engine counters (`vehicles.generated`, `.arrived`, …) continue; travel-time statistics
  and event counts restart at the restore, and a warning says so. Keep the in-memory
  `Snapshot` if you need them.
