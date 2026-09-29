"""``urbanflow generate``: build a scenario with a registered generator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from pydantic import BaseModel


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``extra`` over ``base`` (nested dicts merge key by key)."""
    out = dict(base)
    for key, value in extra.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            out[key] = _merge(current, value)
        else:
            out[key] = value
    return out


def _params_file(path: Path) -> dict[str, Any]:
    from urbanflow.core.errors import ConfigError, NotFoundError

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise NotFoundError(f"params file not found: {path}") from None
    except (ValueError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a JSON object of parameters")
    return data


def _type_name(annotation: Any) -> str:
    text = str(annotation)
    if isinstance(annotation, type) and "[" not in text:
        text = annotation.__name__
    return text.replace("typing.", "").replace("NoneType", "None")


def _describe(name: str, params: type[BaseModel]) -> None:
    from urbanflow.cli.output import print_table

    rows = []
    for field_name, info in params.model_fields.items():
        default = info.get_default(call_default_factory=True)
        if hasattr(default, "model_dump"):
            default = json.dumps(default.model_dump(mode="json"))
        rows.append((field_name, _type_name(info.annotation), default, info.description or ""))
    print_table(f"Parameters of {name}", ("Name", "Type", "Default", "Description"), rows)


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def generate(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Generator name (see --list).")] = None,
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param", "-p", help="KEY=VALUE parameter (repeatable; dotted keys, JSON values)."
        ),
    ] = None,
    params_file: Annotated[
        Path | None, typer.Option("--params", help="JSON file with parameters.")
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output file [default: <workspace>/scenarios/<name>.json]."
        ),
    ] = None,
    list_all: Annotated[bool, typer.Option("--list", help="List generators and exit.")] = False,
    describe: Annotated[
        bool, typer.Option("--describe", help="Show the generator's parameters and exit.")
    ] = False,
    no_validate: Annotated[
        bool, typer.Option("--no-validate", help="Write the file even if it does not validate.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Generate a scenario from a registered generator."""
    from urbanflow.cli.main import state
    from urbanflow.cli.output import console, print_table
    from urbanflow.cli.params import parse_assignments
    from urbanflow.core.errors import ConfigError
    from urbanflow.scenario.generators import GENERATORS, list_generators
    from urbanflow.scenario.generators import generate as run_generator
    from urbanflow.scenario.validate import validate_spec

    if list_all:
        rows = [(n, e.source, e.description) for n, e in list_generators()]
        print_table("Generators", ("Name", "Source", "Description"), rows)
        return
    if name is None:
        raise typer.BadParameter("missing generator NAME (see --list)", param_hint="NAME")
    entry = GENERATORS.get(name)
    if describe:
        _describe(name, entry.params)
        return
    params = _params_file(params_file) if params_file is not None else {}
    params = _merge(params, parse_assignments(param or ()))
    target = output or state(ctx).settings().scenarios_dir / f"{name}.json"
    if target.exists() and not force:
        raise ConfigError(f"{_shown(target)} already exists (use --force to overwrite)")
    scenario = run_generator(name, params)
    if not no_validate:
        validate_spec(scenario.spec).raise_for_errors(source=f"generator {name}")
    scenario.save(target)
    counts = scenario.summary()
    console.print(
        f"Wrote {_shown(target)} ({counts['intersections']} intersections, "
        f"{counts['roads']} roads, {counts['flows']} flows; hash {scenario.short_hash})",
        markup=False,
        soft_wrap=True,
    )


def register(app: typer.Typer) -> None:
    app.command("generate", rich_help_panel="Scenarios")(generate)
