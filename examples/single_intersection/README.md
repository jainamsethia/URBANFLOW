# Single intersection

A four-arm signalised junction: the bundled `single_intersection` scenario and a generated
one. Both scripts use only the public API.

| script | shows |
|---|---|
| `run.py` | Load the bundled scenario, list its phases, step it and watch the light through `sim.signals["J"]` (phase, stage, movement states, time left in the stage, the fixed-time cycle), then print the results. |
| `inspect_vehicles.py` | Generate a junction with `urbanflow.generate`, run it for five minutes, then read the intersection view (movements, their signal state and right-of-way rank), the longest stop-line queue and its vehicles, and the vector state (`sim.state`, `sim.vehicles.to_columns()`). |

Run them from the repository root:

```bash
uv run python examples/single_intersection/run.py
uv run python examples/single_intersection/inspect_vehicles.py
```

`URBANFLOW_EXAMPLE_QUICK=1` shortens every example; `tests/acceptance/test_examples.py`
runs them that way.

The same scenario from the command line, and with the signal switched to on-request
control:

```bash
uv run urbanflow run src/urbanflow/scenario/bundled/single_intersection.json --duration 600
uv run urbanflow run src/urbanflow/scenario/bundled/single_intersection.json --duration 600 --controller J=external
```

With `external` and nobody requesting phases, the light stays in its first phase, so the
north-south traffic never gets green.
