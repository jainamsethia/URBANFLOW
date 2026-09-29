"""``urbanflow schema``: print the JSON Schema of scenario files or the simulation config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

SCHEMAS = ("scenario", "config")


def _config_schema() -> dict[str, Any]:
    from urbanflow.core.config import SimulationConfig

    schema = SimulationConfig.model_json_schema(mode="validation")
    schema.pop("title", None)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:urbanflow:config:1.0",
        "title": "UrbanFlow simulation config",
        **schema,
    }


def schema(
    which: Annotated[str, typer.Argument(help="scenario|config.")] = "scenario",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write to this file instead of stdout.")
    ] = None,
) -> None:
    """Print JSON Schema (scenario|config)."""
    from urbanflow.cli.output import console

    if which not in SCHEMAS:
        raise typer.BadParameter(f"expected one of: {', '.join(SCHEMAS)}", param_hint="WHICH")
    if which == "scenario":
        from urbanflow.scenario.schema import scenario_json_schema

        data = scenario_json_schema()
    else:
        data = _config_schema()
    if output is None:
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return
    from urbanflow.scenario.io import write_json

    console.print(f"Wrote {write_json(output, data)}", markup=False, soft_wrap=True)


def register(app: typer.Typer) -> None:
    app.command("schema", rich_help_panel="Scenarios")(schema)
