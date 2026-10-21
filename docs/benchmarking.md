# Benchmarking

`urbanflow benchmark` measures how fast the engine simulates fixed workloads, in a way you
can repeat and compare: the same scenarios and seeds, one fresh process per repeat, the
machine recorded with every result, and a determinism check on every case.

```bash
urbanflow benchmark --list
urbanflow benchmark grid_scaling --quick
urbanflow benchmark grid_scaling -o grid.json
urbanflow benchmark overhead --compare benchmarks/baselines/windows-arm64-py313.json
urbanflow benchmark grid_scaling overhead --quick --repeat 3 -o mine.json
urbanflow benchmark my_city.json --quick --repeat 3
```

## What is measured

For each case and repeat, after a warm-up that is simulated but not timed:

| value | meaning |
|---|---|
| `steps_per_s` | timed steps per wall-clock second |
| `vehicle_steps_per_s` | running vehicles summed over the timed steps, per second: the main scaling measure, because the cost of a step grows with the vehicles in it |
| `rtf` | real-time factor: simulated seconds per wall-clock second (`steps_per_s x dt`) |
| `mean_active`, `max_active` | running vehicles per timed step (measured, not assumed) |
| `compile_s` | building the scenario, validating it, compiling the network and setting up the simulation |
| `wall_s` | wall time of the timed loop |
| `peak_rss_mb` | peak memory of the process (peak working set on Windows, `ru_maxrss` elsewhere) |
| `state_digest` | the engine state digest at the end; every repeat of a case must end in the same one, or the case is flagged `nondeterministic` |

Each value is reported as the median and quartiles over the repeats, with every repeat's
value kept in the JSON report.

## Workloads

| suite | cases | workload |
|---|---|---|
| `grid_scaling` | `g1`, `g2`, `g4`, `g8`, `g12` | square grids of 1, 4, 16, 64 and 144 signalised junctions (fixed-time), 300 m blocks, 3 lanes per direction; 360 veh/h of demand per junction, split evenly over the boundary entries |
| `vehicle_sweep` | `d0.25` ... `d4` | the 8 x 8 grid at 0.25, 0.5, 1, 2 and 4 times that demand |
| `overhead` | `baseline`, `no_lane_change`, `recording`, `debug_checks`, `empty` | the 4 x 4 grid as is, without lane changes, recording a replay, with every invariant checked each step, and with (almost) no traffic: the fixed cost of a step |
| a scenario file | its file name | your own scenario, no warm-up |

Full mode simulates 300 s of warm-up and 1500 s timed; `--quick` 60 s and 300 s.

## Procedure

- **Isolation**: each repeat runs in a new spawned process, one after the other (never
  two at once), so repeats share no caches and do not compete for cores. Default 5 repeats.
- **Timing**: `perf_counter_ns` around the timed steps only; garbage collection stays on
  (as in real use). Counting the running vehicles each step is included (about 50 µs).
- **Environment**: the report records Python, numpy and UrbanFlow versions, platform, CPU
  counts and memory, and a machine key such as `windows-arm64-py313`. Compare results only
  between runs with the same key.
- **Power**: on a laptop, run on mains power with the high-performance power plan and
  nothing else busy. On battery, processors run at a fraction of their clock speed and
  results can halve between runs.

## Comparing runs

`--compare BASELINE.json` prints every case whose median vehicle-steps/s (steps/s for the
empty network) fell by more than `--threshold` (default 10 %) **and** whose interquartile
range no longer overlaps the baseline's, then exits with code 10. Requiring both keeps
ordinary noise from being reported as a slowdown.

The repository keeps per-machine baselines in `benchmarks/baselines/<machine key>.json`;
`pytest -m performance` re-runs those cases quickly and fails on a slowdown of more than
30 %. It is skipped on machines without a baseline. A scheduled CI job runs the quick
`grid_scaling` and `overhead` suites on a GitHub runner and keeps the JSON reports as
artifacts; it never fails a build, because shared runners are too noisy for that.

## Profiling

`--profile DIR` writes a cProfile file of each case's timed loop (first repeat). Open it
with `python -m pstats DIR/grid_scaling-g12.prof` or `snakeviz`. On the 144-junction grid
the step time splits roughly as: finding each vehicle's leader about a quarter (it runs
again after lane changes), lane changes and junction admission with conflict-zone locks
about a sixth each, then signals, insertion and the car-following model. These are the
candidates for the optional numba backend. Any optimisation must leave every state digest
unchanged.

## Results

Numbers are published here only when measured with this harness on a machine running on
mains power, together with their JSON report. None are published yet.

## Limitations

- These are numbers for the numpy backend. `accel="numba"` is not available in this version.
- The workloads are synthetic grids; real networks with long roads, many lanes or heavy
  congestion behave differently. Benchmark your own scenario file for those.
- Other simulators use other car-following, lane-change and junction models, so their
  speed is not directly comparable; an imported CityFlow network has the same topology,
  not the same dynamics.
