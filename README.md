# UrbanFlow

**UrbanFlow is a Python-first microscopic city-traffic simulator and research workbench**
for traffic-signal control, reinforcement learning and teaching. It is an original,
Apache-2.0 project that revisits what [CityFlow](https://github.com/cityflow-project/CityFlow)
offered, with a deterministic NumPy engine, first-party Gymnasium/PettingZoo environments,
replays and a web workbench.

## What works today

- **Microscopic engine** (vectorised NumPy, deterministic): IDM car-following with a
  Gipps-style safe-speed cap, ballistic integration, MOBIL lane changing (discretionary
  and route-mandatory, no shadow vehicles), conflict-aware intersections
  (gap acceptance with time windows, "don't block the box" exit space, conflict-zone
  locks), priority / uncontrolled / signalised junctions, a deadlock watchdog, and
  debug-mode invariants (no overlaps, zone exclusivity, conservation, ...).
- **Signals**: programs with automatic yellow and all-red, `fixed_time`, `webster`
  (Webster's cycle and splits from the scenario demand), `actuated` (gap-out / max-out),
  `max_pressure` (Varaiya) and `external` (API / RL) controllers,
  manual hold/release, live controller switching.
- **Scenarios**: versioned JSON format with friendly, JSON-path validation errors; derived
  movements, lane mappings and signal programs; generators `single_intersection`, `grid`
  and `corridor` (an arterial with green-wave signal offsets); bundled: `single_intersection`,
  `grid_3x3`, `grid_4x4`, `corridor`.
- **Metrics**: travel time, delay, waiting, stops, throughput, queues, space-mean speed,
  VKT/VHT; global, per-intersection and per-trip tables; CSV / JSON / Parquet export.
- **Replay**: compact `.ufr` recordings with seek, step back and rewind.
- **Controller comparison**: controllers x seeds on identical demand (common random
  numbers), 95 % confidence intervals and paired differences.
- **RL**: `TrafficSignalEnv` (Gymnasium) and `TrafficSignalParallelEnv` (PettingZoo), with
  action masks, several rewards, and random / fixed-cycle / max-pressure / tabular-Q
  baselines.
- **Web workbench**: live map with moving vehicles and signal heads, lane heat maps, KPI
  charts, vehicle/intersection inspector, phase hold, controller switching, comparison
  view and replay viewer.
- **Snapshots** (save/restore any state) and **state digests** for bit-exact reproducibility.

Not built yet (planned): more generators (downtown, rush hour), emergency preemption, transit, the CityFlow importer, the visual
scenario editor, the experiment database/dashboards and the benchmark suite.

## Install

UrbanFlow uses [uv](https://docs.astral.sh/uv/) and Python 3.12-3.14 (Linux, macOS,
Windows, including Windows on ARM64).

```bash
git clone https://github.com/jainamsethia/URBANFLOW.git urbanflow
cd urbanflow
uv sync
uv run pytest
uv run urbanflow --help
```

The web UI is built once with Node 22+ (it is served by the Python server afterwards):

```bash
npm --prefix frontend install
npm --prefix frontend run build
```

## Demo in five commands

```bash
uv run urbanflow generate grid -p rows=3 -p cols=3 -o grid.json   # make a scenario
uv run urbanflow validate grid.json                                # friendly checks
uv run urbanflow run grid.json --duration 1800 --record            # simulate + replay
uv run urbanflow compare grid_3x3 -c fixed_time -c actuated -c max_pressure --seeds 3 --duration 1800
uv run urbanflow serve --open                                      # web workbench
```

In the workbench: start a session, press Play, click an intersection to see its phase and
hold a phase, switch the signal controller live, change the view to a lane heat map, run a
comparison in the *Compare controllers* tab, or open a recorded replay and scrub the
timeline.

## Python

```python
from urbanflow import Simulation, bundled

with Simulation.from_scenario(
    bundled("grid_3x3"), controllers={"*": "max_pressure"}, duration=1800, seed=1
) as sim:
    sim.run(until=60)
    print(sim.signals["r1c1"])  # phase, stage, movement states
    result = sim.run()
print(result.summary["delay.mean"], result.summary["throughput_vph"])
result.export("out", format="csv")  # timeseries, intersections, trips
```

Reinforcement learning:

```python
from urbanflow.rl import TrafficSignalEnv

env = TrafficSignalEnv("single_intersection", episode_length=900, reward="queue")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(info["action_mask"].argmax())
```

## Measured results (reproduce with the commands shown)

All measured with lane changing on (the default).

Controller comparison, `grid_3x3`, 1800 s, 3 seeds, paired against `fixed_time`
(`uv run urbanflow compare grid_3x3 -c fixed_time -c webster -c actuated -c max_pressure --seeds 3 --duration 1800`):

| metric | fixed_time | webster | actuated | max_pressure |
|---|---|---|---|---|
| mean travel time (s) | 99.4 | 85.4 (-14.1 %) | 86.5 (-12.9 %) | 81.0 (-18.5 %) |
| mean delay (s) | 38.4 | 24.4 (-36.5 %) | 25.5 (-33.6 %) | 20.0 (-48.1 %) |
| mean waiting time (s) | 21.3 | 7.1 (-66.8 %) | 8.4 (-60.6 %) | 6.9 (-67.5 %) |
| stops per vehicle | 1.4 | 1.2 (-13.1 %) | 1.0 (-24.9 %) | 0.8 (-39.7 %) |

Tabular Q-learning, single intersection with 3:1 asymmetric demand, 60 training episodes,
5 held-out evaluation seeds (`uv run python examples/reinforcement_learning/q_learning_single.py`):

| policy | mean waiting (s) | mean travel time (s) |
|---|---|---|
| random | 43.5 | 125.8 |
| fixed cycle (30 s) | 46.1 | 110.2 |
| max-pressure | 18.5 | 69.1 |
| Q-learning | 23.6 | 88.5 |

The tabular learner (60 short episodes, a 7-feature state) clearly beats random and
fixed-cycle control but not max-pressure on this junction.

Signal coordination: `corridor` (5 junctions, 90 s cycle, 1800 s, seed 2), eastbound
through traffic from `W` to `E`:

| offsets | mean travel time (s) | stops per vehicle |
|---|---|---|
| none (all 0) | 152.7 | 2.1 |
| green wave (`o_i = i * spacing / v`) | 109.7 | 0.84 |

Multi-agent: nine independent tabular Q-learners on `grid_3x3` (PettingZoo env, 40 training
episodes of 900 s, 3 held-out seeds; `uv run python examples/reinforcement_learning/independent_q_grid.py`):

| policy | mean waiting (s) | mean travel time (s) |
|---|---|---|
| fixed-time cycle (30 s) | 21.7 | 99.6 |
| max-pressure | 7.4 | 81.5 |
| independent Q-learning | 5.0 | 80.5 |

Numbers come from this machine (Windows 11 ARM64, Python 3.13); they are deterministic for
a given seed and platform.

## Documentation

`docs/` (build with `uv run --group docs mkdocs serve`): getting started, scenario format,
road network, simulation model, vehicle model, traffic signals, Python API, CLI and
troubleshooting. Examples are in `examples/`.

## Development

```bash
uv run pytest                       # ~1,470 tests
uv run ruff check . && uv run ruff format --check . && uv run mypy
npm --prefix frontend run typecheck
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). CityFlow was used as a
behavioural reference only; no CityFlow code is included.
