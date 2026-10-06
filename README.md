# UrbanFlow

**UrbanFlow is a Python-first microscopic city-traffic simulator and research workbench**
for traffic-signal control, reinforcement learning and teaching. It is an original,
Apache-2.0 project that revisits what [CityFlow](https://github.com/cityflow-project/CityFlow)
offered, with a deterministic NumPy engine, first-party Gymnasium/PettingZoo environments,
replays and a web workbench.

## What works today

- **Microscopic engine** (vectorised NumPy, deterministic): IDM car-following with a
  Gipps-style safe-speed cap, ballistic integration, conflict-aware intersections
  (gap acceptance with time windows, "don't block the box" exit space, conflict-zone
  locks), priority / uncontrolled / signalised junctions, a deadlock watchdog, and
  debug-mode invariants (no overlaps, zone exclusivity, conservation, ...).
- **Signals**: programs with automatic yellow and all-red, `fixed_time`, `actuated`
  (gap-out / max-out), `max_pressure` (Varaiya) and `external` (API / RL) controllers,
  manual hold/release, live controller switching.
- **Scenarios**: versioned JSON format with friendly, JSON-path validation errors; derived
  movements, lane mappings and signal programs; generators `single_intersection` and
  `grid` (bundled: `single_intersection`, `grid_3x3`, `grid_4x4`).
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

Not built yet (planned): lane changing (MOBIL), more generators (corridor, downtown, rush
hour), Webster and emergency preemption, transit, the CityFlow importer, the visual
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

with Simulation.from_scenario(bundled("grid_3x3"), controllers={"*": "max_pressure"},
                              duration=1800, seed=1) as sim:
    sim.run(until=60)
    print(sim.signals["r1c1"])          # phase, stage, movement states
    result = sim.run()
print(result.summary["delay.mean"], result.summary["throughput_vph"])
result.export("out", format="csv")     # timeseries, intersections, trips
```

Reinforcement learning:

```python
from urbanflow.rl import TrafficSignalEnv
env = TrafficSignalEnv("single_intersection", episode_length=900, reward="queue")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(info["action_mask"].argmax())
```

## Measured results (reproduce with the commands shown)

Controller comparison, `grid_3x3`, 1800 s, 3 seeds, paired against `fixed_time`
(`uv run urbanflow compare grid_3x3 -c fixed_time -c actuated -c max_pressure --seeds 3 --duration 1800`):

| metric | fixed_time | actuated | max_pressure |
|---|---|---|---|
| mean travel time (s) | 101.2 | 88.1 (-12.9 %) | 82.9 (-18.1 %) |
| mean delay (s) | 40.2 | 27.1 (-32.6 %) | 21.9 (-45.5 %) |
| mean waiting time (s) | 22.2 | 9.4 (-57.5 %) | 7.7 (-65.1 %) |

Tabular Q-learning, single intersection with 3:1 asymmetric demand, 60 training episodes,
5 held-out evaluation seeds (`uv run python examples/reinforcement_learning/q_learning_single.py`):

| policy | mean waiting (s) | mean travel time (s) |
|---|---|---|
| random | 42.7 | 124.9 |
| fixed cycle (30 s) | 39.8 | 101.8 |
| max-pressure | 27.1 | 82.6 |
| Q-learning | 20.6 | 80.8 |

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
