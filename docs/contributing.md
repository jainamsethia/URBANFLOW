# Contributing

## Development loop

```bash
uv sync                          # core + dev dependencies (tests, lint, rl, data)
uv run pytest                    # unit, integration, regression and acceptance tests
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

The quality gates run in CI on Linux, macOS and Windows for Python 3.12–3.14: `ruff`, `ruff format`, `mypy --strict` and `pytest`.

## Architecture rules

UrbanFlow is a layered modular monolith. Imports may only point *down* the layers:

| Layer | Packages |
|---|---|
| L0 | `core` |
| L1 | `geometry`, `scenario` |
| L2 | `network`, `routing` |
| L3 | `vehicles`, `signals`, `demand` |
| L4 | `engine` |
| L5 | `metrics`, `replay`, `visualization`, and the facade (`simulation`, `views`, `results`, …) |
| L6 | `rl`, `experiments`, `benchmark`, `plugins` |
| L7 | `server`, `cli` |

`tests/unit/test_layering.py` enforces these rules by parsing every import. The engine never imports FastAPI, Gymnasium, Typer or SQLAlchemy.

## Style

- Type hints everywhere; public APIs carry docstrings.
- No magic numbers outside `urbanflow.core.constants`.
- Errors a user can trigger raise an `UrbanFlowError` subclass with a friendly message. Validation errors include JSON paths.
- Randomness comes only from `RngStreams` (no global RNG, no `hash()`).
- A deliberate simplification is marked with a `ponytail:` comment. The comment names the limit and the upgrade path.

## Generated files

Some committed files are generated from the code; tests fail when they drift.

| File | Regenerate with |
|---|---|
| `src/urbanflow/scenario/bundled/*.json` | `uv run python scripts/regen_bundled_scenarios.py` |
| `docs/reference/scenario.schema.json`, generated blocks of `docs/scenario-format.md` | `uv run python scripts/gen_scenario_docs.py` |

Both scripts accept `--check`, which only reports stale files.

## Adding a generator

Write a pydantic `Params` model and a function `fn(params) -> Scenario` built with
`ScenarioBuilder`, decorate it with `@register_generator("name", params=Params)` and, for a
built-in, add the module to the import list at the end of
`urbanflow/scenario/generators/__init__.py`. Generators must be pure functions of their
parameters (randomness only through a `seed` parameter and `RngStreams`).
