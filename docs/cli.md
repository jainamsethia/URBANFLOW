# Command line

`urbanflow` (also `python -m urbanflow`) groups its commands into panels; `urbanflow --help`
lists them and every command has `--help`. This page covers the commands available so far;
it grows as the replay, experiment and workbench commands land.

## Global options

| option | meaning |
|---|---|
| `--version` | Print the version, Python, numpy and platform, then exit. |
| `--workspace`, `-w PATH` | Workspace directory (env `URBANFLOW_WORKSPACE`, default: current directory). |
| `--verbose`, `-v` | `-v` INFO, `-vv` DEBUG logging (default WARNING). |
| `--log-level LEVEL` | DEBUG, INFO, WARNING or ERROR (overrides `-v`). |
| `--log-format text\|json` | Log format on stderr. |
| `--debug` | Show full tracebacks. |

Results go to stdout; logs, progress and errors go to stderr. `NO_COLOR` is respected.

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | internal error, or `doctor` found a failure |
| 2 | usage error (unknown option, missing argument) |
| 3 | scenario invalid |
| 4 | configuration error (bad option value, file exists, non-empty directory) |
| 5 | not found (file, generator, bundled scenario) |
| 6 | simulation error (a feature not available yet, lifecycle misuse) |
| 7 | invariant violation (`--debug-checks`) |
| 130 | interrupted |

## Scenarios

### `urbanflow init [DIR]`

Scaffold a workspace: `urbanflow.toml`, `scenarios/<name>.json` (copied from a bundled
scenario), `experiments/example.json`, `runs/`, `.gitignore` and `README.md`.

| option | meaning |
|---|---|
| `--template`, `-t NAME` | Bundled scenario to copy (default `single_intersection`; unknown names list the available ones). |
| `--name TEXT` | File name of the copied scenario (default: the template name). |
| `--force` | Scaffold into a non-empty directory. Existing files are never overwritten. |

`DIR` defaults to `--workspace` or the current directory.

### `urbanflow validate SCENARIO...`

Validate scenario files: structure, ids and references, geometry, movements, signals,
demand and transit. Errors are printed as `path: message` lines with did-you-mean hints;
warnings follow under `Warnings:`; a valid file prints
`OK: <name> (hash <12 hex>; <counts>)`. Every file is checked even if an earlier one is
invalid or missing; the exit code is 5 if a file was not found, else 3 if any file failed.

| option | meaning |
|---|---|
| `--strict` | Warnings also fail (exit 3). |
| `--format text\|json` | `json` prints `{"ok": ..., "files": [{"path", "ok", "errors", "warnings", ...}]}` (a missing file has an `error` message). |
| `--deep/--no-deep` | Also run the deep checks of every valid file (default on), the same as `urbanflow.check`: unknown car-following models or routers (E904), car-following `model_params` the model rejects (E903), and the network compile: short or degenerate lanes (E802, E905), lanes too short for a routed vehicle type (E806), crossing protected movements (W302) and connectors crossing twice (W304). See [Road network](road-network.md#diagnostics). |

```text
$ urbanflow validate scenarios/broken.json
Scenario validation failed:
  - network.intersections[0].signal.phases[1]: phase contains unknown movement "E_in->W_ot" (did you mean "E_in->W_out"?)
  - demand.trips[0].vehicle_type: unknown vehicle type "cr" (did you mean "car"?) (available: bus, car, emergency, truck)
Warnings:
  - simulation.dt: dt=1.5 s is outside the recommended 0.2-1.0 s
$ echo $?
3
```

### `urbanflow generate [NAME]`

Generate a scenario with a registered generator and write it to
`<workspace>/scenarios/<name>.json` (or `-o`).

| option | meaning |
|---|---|
| `-p`, `--param KEY=VALUE` | A parameter (repeatable). Dotted keys nest (`turn_ratios.far=0.2`); values are JSON literals, falling back to strings (`kind=priority`). |
| `--params FILE.json` | Parameters from a JSON object; `-p` values override it. |
| `-o`, `--output PATH` | Output file. |
| `--list` | List the generators (name, source, description). |
| `--describe` | Show the generator's parameters with types and defaults. |
| `--no-validate` | Write the file even if it does not validate. |
| `--force` | Overwrite an existing output file. |

Parameter errors are reported with paths such as `params.lanes` (exit 3).

```text
$ urbanflow generate single_intersection -p kind=uncontrolled -o s.json
Wrote s.json (5 intersections, 8 roads, 12 flows; hash 3c0ffa9b370b)
```

### `urbanflow import cityflow`

Convert a CityFlow road network and flow file into an UrbanFlow scenario (see
[Scenario format: importing CityFlow](scenario-format.md#importing-cityflow)).

| option | meaning |
|---|---|
| `--config PATH` | CityFlow `config.json`; locates `dir` + `roadnetFile` / `flowFile` (from the working directory, as CityFlow does, else next to the config) and supplies `interval`, `seed`, `laneChange` and `rlTrafficLight`. |
| `--roadnet PATH`, `--flow PATH` | The files directly (override the config's). |
| `-o`, `--output PATH` | Scenario file to write (default `imported.json`). |
| `--name TEXT`, `--duration S` | Scenario name; simulated duration (default 3600 s). |
| `--derive-connections` | Use UrbanFlow's lane mapping instead of CityFlow's lane fan-out. |
| `--keep-all-phases` | Keep short all-right-turn phases instead of treating them as intergreens. |
| `--force` | Overwrite an existing output file. |

Every lossy conversion step is printed as a warning on stderr; input errors name the input
path (`roadnet.intersections[3].trafficLight.lightphases[2].availableRoadLinks[0]`, exit 3).

```text
$ urbanflow import cityflow --config data/config.json -o city.json
4 conversion warning(s):
  - config.saveReplay: ignored (UrbanFlow records replays with --record)
  - config.roadnetLogFile: ignored (UrbanFlow records replays with --record)
  - roadnet.intersections[0].roadLinks: lane fan-out kept (one lane links to several lanes); --derive-connections uses UrbanFlow's lane mapping
  - roadnet.intersections[1].roadLinks: lane fan-out kept (one lane links to several lanes); --derive-connections uses UrbanFlow's lane mapping
Wrote city.json (8 intersections, 14 roads, 3 flows; hash 05f6aa433010)
```

### `urbanflow schema [scenario|config]`

Print the JSON Schema of scenario files or of the simulation config, or write it with
`-o FILE`. Editors can use the scenario schema for completion and inline checks; it is
also committed as [reference/scenario.schema.json](reference/scenario.schema.json).

## Simulation

### `urbanflow run SCENARIO`

Load and deep-check a scenario, run it headless, print the results and save the run to
`<workspace>/runs/<run_id>/`.

| option | meaning |
|---|---|
| `--duration S` | Time limit, s (the config `duration`). |
| `--until S` | Stop at this simulated time, s (never past the time limit). |
| `--seed N`, `--dt S`, `--router NAME` | Seed, step length and router. |
| `--controller NAME`, `--controller J=NAME` | Signal controller for every signalised intersection, or for intersection `J` (repeatable; per-intersection options win). Unknown names exit 3 with a did-you-mean hint (E901), unknown intersections exit 5. |
| `--config FILE` | TOML file whose `[simulation]` table sets defaults (default: the workspace `urbanflow.toml`, if it exists). |
| `--set KEY=VALUE` | A config override (repeatable; dotted keys such as `metrics.warmup=300`, JSON values). |
| `--debug-checks` | Check every invariant each step (slow; a failure exits 7). |
| `--out DIR` | Save the run in `DIR` (new or empty) instead of `runs/<run_id>/`. |
| `--no-save` | Do not write a run directory. |
| `--progress/--no-progress` | Progress bar on stderr (default: when stderr is a terminal). |
| `--json` | Print only the result as JSON on stdout (`SimulationResult.to_dict()` plus `run_dir`). |
| `--record` | Record a replay: `<run dir>/replay.ufr`, or `<workspace>/replays/<scenario>-s<seed>.ufr` with `--no-save`. |
| `--export PATH` | Also write the global timeseries table to `PATH` (`.csv`, `.json` or `.parquet`). |

Configuration precedence, lowest first: defaults < the scenario's `simulation` block <
the config file `[simulation]` < `--set` < the dedicated options. The run directory holds
`spec.json` (scenario identity, seed, resolved config, `--controller` choices), `scenario.json` (an exact copy),
`env.json` (Python, platform, CPU, RAM, versions), `result.json` (the full result, loadable
with `SimulationResult.load`), the metrics tables `timeseries.csv`, `intersections.csv` and
`trips.csv`, and `summary.json`, the nested summary, written last.
The printed `State digest` (also `state_digest` in `result.json` and in `--json` output)
is the sha256 of the final engine state: two runs with equal digests on the same platform
ended in exactly the same state.
Run ids look like `20260923T141502-grid-3x3-s7-a1f0` (UTC time, scenario, seed, random
suffix). Ctrl-C stops the run, saves the partial results with `interrupted: true` and
exits 130.

For example, `urbanflow run scenarios/signals.json --controller external --controller
J2=fixed_time` puts every signal under on-request control except `J2`.

```text
$ urbanflow run scenarios/cross.json --duration 600 --seed 7
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

Exit codes: 3 invalid scenario (including the deep checks), 4 bad option or config value
(or a non-empty `--out`), 5 missing scenario or config file, 6 a feature not available
yet (transit lines), 7 invariant violation, 130 interrupted.

### `urbanflow replay FILE`

Print a replay's manifest (scenario, frames, vehicles, seed, controllers); `--json`
prints it as JSON. `--view` copies the file into `<workspace>/replays/` if needed, starts
the workbench (`--port`, default 8000) and opens the replay viewer. See [Replay](replay.md).

## Research

### `urbanflow compare SCENARIO -c NAME -c NAME ...`

Run every controller on the same seeds of one scenario (a bundled name or a file) in
parallel processes and print, per metric, the mean with a 95 % confidence interval and
the paired difference against the first controller (`*` marks differences whose 95 % CI
excludes 0). Options: `--seeds N` (default 3), `--duration S`, `--workers N`, `--json`,
`--out FILE` (JSON). Demand is drawn per flow at spawn, so a seed gives identical demand
to every controller.

```bash
uv run urbanflow compare grid_3x3 -c fixed_time -c actuated -c max_pressure --seeds 3 --duration 1800
```

## Workbench

### `urbanflow serve`

Start the web workbench and its API on `--host` (default `127.0.0.1`) and `--port`
(default 8000); `--open` opens a browser. Binding to a non-loopback address requires
`--token`/`--token-file` (clients then send `Authorization: Bearer <token>`). `--dev`
also accepts the Vite dev server on `localhost:5173`. See [Web workbench](workbench.md).

### `urbanflow doctor`

Check Python, platform, core dependencies, optional extras, the frontend bundle, the
server port, the workspace and the database. `--json` prints JSON, `--port N` checks
another port and `--deep` adds a multiprocessing round trip. Any failure exits 1.
