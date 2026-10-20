"""``urbanflow import``: convert a foreign scenario format (CityFlow) to UrbanFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

FORMATS = ("cityflow", "osm")


def import_cmd(
    fmt: Annotated[str, typer.Argument(metavar="FORMAT", help="Input format: cityflow or osm.")],
    config: Annotated[
        Path | None, typer.Option("--config", help="CityFlow config.json (locates the files).")
    ] = None,
    roadnet: Annotated[
        Path | None, typer.Option("--roadnet", help="CityFlow roadnet file.")
    ] = None,
    flow: Annotated[Path | None, typer.Option("--flow", help="CityFlow flow file.")] = None,
    osm: Annotated[
        Path | None, typer.Option("--osm", help="OpenStreetMap extract (.osm XML).")
    ] = None,
    demand: Annotated[
        float, typer.Option("--demand", help="OSM: traffic multiplier.", min=0.01, max=5)
    ] = 1.0,
    output: Annotated[Path, typer.Option("--output", "-o", help="Scenario file to write.")] = Path(
        "imported.json"
    ),
    name: Annotated[str, typer.Option("--name", help="Scenario name.")] = "cityflow_import",
    duration: Annotated[
        float, typer.Option("--duration", help="Simulated duration, s.", min=1)
    ] = 3600.0,
    derive_connections: Annotated[
        bool,
        typer.Option("--derive-connections", help="Use UrbanFlow's lane mapping, not fan-out."),
    ] = False,
    keep_all_phases: Annotated[
        bool,
        typer.Option("--keep-all-phases", help="Keep short all-right-turn (intergreen) phases."),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Convert a foreign format (cityflow, osm) into an UrbanFlow scenario."""
    from urbanflow.cli.output import console, err_console
    from urbanflow.core.errors import ConfigError
    from urbanflow.scenario.importers.cityflow import CityFlowOptions, load

    if fmt not in FORMATS:
        raise typer.BadParameter(f"unknown format {fmt!r} (available: cityflow, osm)")
    if output.exists() and not force:
        raise ConfigError(f"{output} already exists (use --force to overwrite)")
    if fmt == "osm":
        from urbanflow.scenario.importers.osm import OsmOptions
        from urbanflow.scenario.importers.osm import load as load_osm

        if osm is None:
            raise typer.BadParameter("give the OpenStreetMap file with --osm", param_hint="--osm")
        result = load_osm(osm, OsmOptions(name=name, duration=duration, demand=demand))
    else:
        options = CityFlowOptions(
            name=name,
            duration=duration,
            derive_connections=derive_connections,
            keep_all_phases=keep_all_phases,
        )
        result = load(config=config, roadnet=roadnet, flow=flow, options=options)
    if result.warnings:
        err_console.print(f"{len(result.warnings)} conversion warning(s):", markup=False)
        for w in result.warnings:
            err_console.print(f"  - {w.path}: {w.message}", markup=False, soft_wrap=True)
    result.scenario.save(output)
    counts = result.scenario.summary()
    console.print(
        f"Wrote {output} ({counts['intersections']} intersections, {counts['roads']} roads, "
        f"{counts['flows']} flows; hash {result.scenario.short_hash})",
        markup=False,
        soft_wrap=True,
    )


def register(app: typer.Typer) -> None:
    app.command("import", rich_help_panel="Scenarios")(import_cmd)
