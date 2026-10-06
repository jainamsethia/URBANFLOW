# Getting started

## Requirements

- Python 3.12, 3.13 or 3.14. Linux, macOS and Windows are supported, including Windows on ARM64.
- [uv](https://docs.astral.sh/uv/) for environment management.
- Node.js 24, only if you want to build the web workbench from source.

## Install uv

=== "Windows"

    ```bash
    winget install --id=astral-sh.uv -e
    ```

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

On Windows on ARM64, make sure the interpreter is native:

```bash
uv python install cpython-3.13-windows-aarch64-none
```

## Set up the project

```bash
git clone <repository-url> urbanflow
cd urbanflow
uv sync
uv run pytest
uv run urbanflow --help
uv run urbanflow doctor
```

`urbanflow doctor` reports your Python and platform, the core dependencies, the optional extras (`rl`, `rl-train`, `data`, `accel`, `postgres`), the frontend bundle, the server port and whether the workspace is writable.

### Optional extras on Windows ARM64

Some optional packages publish no Windows-ARM64 wheels:

- **torch** (needed by the `rl-train` extra): run training in a separate x64 environment. See the reinforcement-learning guide.
- **numba** (the `accel` extra): Windows-ARM64 wheels exist only for Python 3.14.

The core simulator, the RL environments and the data export work natively.

## Your first scenario

```bash
uv run urbanflow init my-study             # workspace with a bundled example scenario
cd my-study
uv run urbanflow validate scenarios/single_intersection.json
uv run urbanflow generate single_intersection -p kind=priority -p arms=3 -o scenarios/tee.json
```

Scenario files are JSON; see [Scenario format](scenario-format.md) for the fields, the
derivation rules and every validation message, and [Command line](cli.md) for the options.

## Your first headless run

Generate a single junction without traffic lights and run it for ten simulated minutes:

```bash
uv run urbanflow generate single_intersection -p kind=uncontrolled -o scenarios/cross.json
uv run urbanflow run scenarios/cross.json --duration 600 --seed 7
```

```text
Loaded single_intersection: 1 intersection, 8 roads, 16 lanes, 16 connectors  (hash 3c0ffa9b370b)
     Results: single_intersection (seed 7, 600 s)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric                      ┃       Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Vehicles departed / arrived │   386 / 345 │
│ Vehicles en route / waiting │      41 / 0 │
│ Throughput                  │ 2,070 veh/h │
│ Mean travel time            │      51.9 s │
│ Teleports                   │           0 │
│ Wall time                   │      0.48 s │
└─────────────────────────────┴─────────────┘
State digest: c831f98be7f625dcc42aac30f92dc6ef6fae2c45c048c48b75eede3cc1ad7668
Run saved: runs/20260929T110843-single-intersection-s7-0b3c
```

The run directory holds the exact scenario, the resolved configuration, the environment
and the results (`result.json`, `summary.json`). The same run from Python:

```python
from urbanflow import Simulation, generate

scenario = generate("single_intersection", kind="uncontrolled")
with Simulation(scenario, seed=7, duration=600) as sim:
    result = sim.run()
print(result)
print(result.summary["travel_time.mean"], result.summary["vehicles.arrived"])
```

## Your first signalised run

The bundled `single_intersection` scenario is a four-arm junction with a fixed-time
traffic light (two 30 s phases, 3 s yellow, 1 s all-red). From the repository root:

```bash
uv run urbanflow run src/urbanflow/scenario/bundled/single_intersection.json --duration 600
```

```text
Loaded single_intersection: 1 intersection, 8 roads, 16 lanes, 16 connectors  (hash 3aefba1f3f7f)
     Results: single_intersection (seed 0, 600 s)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric                      ┃       Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Vehicles departed / arrived │   385 / 331 │
│ Vehicles en route / waiting │      54 / 0 │
│ Throughput                  │ 1,986 veh/h │
│ Mean travel time            │      59.7 s │
│ Teleports                   │           0 │
│ Wall time                   │      0.51 s │
└─────────────────────────────┴─────────────┘
State digest: 4d675f0d14b67fe7aecb2ca358fd0104a76d17903e24f91b7bc4243f003c7147
Run saved: runs/20260930T101512-single-intersection-s0-7c2e
```

`--controller external` (or `--controller J=external`) swaps the controller; see
[Command line](cli.md#urbanflow-run-scenario). In Python, `sim.signals` shows and
controls the light:

```python
from urbanflow import Simulation, bundled

sim = Simulation.from_scenario(bundled("single_intersection"), duration=600)
sim.run(until=45)
light = sim.signals["J"]
print(light.phase_id, light.stage, light.state_string, f"{light.remaining:g} s left")
sim.signals.hold_phase("J", "p0")  # manual override: back to the east-west phase
sim.run(until=120)
print(sim.signals["J"].phase_id, sim.signals["J"].held)
```

Signalised intersections run their configured controller (`fixed_time` by default;
`external` switches phases on request). See [Traffic signals](traffic-signals.md) for
programs, the signal state machine and custom controllers, [Python API](python-api.md)
for stepping, vehicle views and vector state, and [Simulation model](simulation-model.md)
for what happens in a step.
