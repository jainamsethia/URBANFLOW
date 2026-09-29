# UrbanFlow

**UrbanFlow is a Python-first microscopic city-traffic simulator and research workbench.** It is designed for reinforcement-learning signal control, transportation research and teaching. It is an original, Apache-2.0 project that modernises what [CityFlow](https://github.com/cityflow-project/CityFlow) offered:

- Deterministic, vectorised NumPy engine: individual vehicles with IDM car-following and MOBIL lane changing, and conflict-aware intersections.
- Signal control framework: fixed-time, actuated, max-pressure, Webster, RL and emergency preemption.
- First-party Gymnasium and PettingZoo environments.
- Experiments with statistical comparison, and reproducible benchmarks.
- Compact, seekable replays.
- A modern web workbench for live simulation, replay, scenario editing and experiment dashboards.

> **Status:** early development (v0.1.0-dev). The project is built in vertical slices; this README grows with each one.

## Install (development)

UrbanFlow uses [uv](https://docs.astral.sh/uv/) and supports Python 3.12–3.14 on Linux, macOS and Windows, including Windows on ARM64.

```bash
git clone <this repository> urbanflow
cd urbanflow
uv sync
uv run pytest
uv run urbanflow --help
uv run urbanflow doctor
```

On Windows on ARM64, install a native uv first, for example `winget install --id=astral-sh.uv -e`. Then run `uv python install cpython-3.13-windows-aarch64-none`.

## First run

```bash
uv run urbanflow generate single_intersection -p kind=uncontrolled -o s.json
uv run urbanflow run s.json --duration 600
```

```python
from urbanflow import Simulation, generate

with Simulation(generate("single_intersection", kind="uncontrolled"), duration=600) as sim:
    print(sim.run())
```

See `docs/getting-started.md` and `docs/python-api.md`.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). CityFlow was used as a behavioural reference only, and no CityFlow code is included.
