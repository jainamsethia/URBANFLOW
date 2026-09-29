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
Wrote s.json (5 intersections, 8 roads, 12 flows; hash 87f6c97627d0)
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
| `--config FILE` | TOML file whose `[simulation]` table sets defaults (default: the workspace `urbanflow.toml`, if it exists). |
| `--set KEY=VALUE` | A config override (repeatable; dotted keys such as `metrics.warmup=300`, JSON values). |
| `--debug-checks` | Check every invariant each step (slow; a failure exits 7). |
| `--out DIR` | Save the run in `DIR` (new or empty) instead of `runs/<run_id>/`. |
| `--no-save` | Do not write a run directory. |
| `--progress/--no-progress` | Progress bar on stderr (default: when stderr is a terminal). |
| `--json` | Print only the result as JSON on stdout (`SimulationResult.to_dict()` plus `run_dir`). |

Configuration precedence, lowest first: defaults < the scenario's `simulation` block <
the config file `[simulation]` < `--set` < the dedicated options. The run directory holds
`spec.json` (scenario identity, seed, resolved config), `scenario.json` (an exact copy),
`env.json` (Python, platform, CPU, RAM, versions), `result.json` (the full result, loadable
with `SimulationResult.load`) and `summary.json`, the nested summary, written last.
Run ids look like `20260923T141502-grid-3x3-s7-a1f0` (UTC time, scenario, seed, random
suffix). Ctrl-C stops the run, saves the partial results with `interrupted: true` and
exits 130.

```text
$ urbanflow run scenarios/cross.json --duration 600 --seed 7
Loaded single_intersection: 1 intersection, 8 roads, 16 lanes, 16 connectors  (hash 87f6c97627d0)
     Results: single_intersection (seed 7, 600 s)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric                      ┃       Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Vehicles departed / arrived │   386 / 346 │
│ Vehicles en route / waiting │      40 / 0 │
│ Throughput                  │ 2,076 veh/h │
│ Mean travel time            │      52.5 s │
│ Teleports                   │           0 │
│ Wall time                   │      0.80 s │
└─────────────────────────────┴─────────────┘
Run saved: runs/20260929T110843-single-intersection-s7-0b3c
```

Exit codes: 3 invalid scenario (including the deep checks), 4 bad option or config value
(or a non-empty `--out`), 5 missing scenario or config file, 6 a feature not available
yet (signalised intersections, transit lines, recording), 7 invariant violation, 130
interrupted.

## Workbench

### `urbanflow doctor`

Check Python, platform, core dependencies, optional extras, the frontend bundle, the
server port, the workspace and the database. `--json` prints JSON, `--port N` checks
another port and `--deep` adds a multiprocessing round trip. Any failure exits 1.
