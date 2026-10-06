"""``urbanflow run``: run a scenario headless, print the results and save the run (AC 7.1)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from urbanflow.core.config import SimulationConfig
    from urbanflow.scenario.scenario import Scenario
    from urbanflow.simulation import Simulation

EXIT_INTERRUPTED = 130


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _config(
    scenario: Scenario, file: Path | None, assignments: list[str], flags: dict[str, Any]
) -> SimulationConfig:
    """defaults < scenario.simulation < config file [simulation] < --set < flags (AB 6.1)."""
    from urbanflow.cli.params import parse_assignments
    from urbanflow.core.config import SimulationConfig, resolve_config

    return resolve_config(
        ("scenario", scenario.spec.simulation),
        (str(file), None if file is None else SimulationConfig.from_toml(file)),
        ("--set", parse_assignments(assignments)),
        ("options", flags),
    )


def _controllers(items: list[str]) -> dict[str, str]:
    """``--controller NAME`` (every signalised intersection, key ``"*"``) and
    ``--controller J=NAME`` (intersection ``J``); a later option wins for the same key."""
    from urbanflow.core.errors import ConfigError

    out: dict[str, str] = {}
    for item in items:
        key, sep, name = item.partition("=")
        if not sep:
            key, name = "*", key
        if not key.strip() or not name.strip():
            raise ConfigError(
                f'invalid --controller "{item}": expected NAME or INTERSECTION=NAME '
                "(e.g. --controller external or --controller J=fixed_time)"
            )
        out[key.strip()] = name.strip()
    return out


def _loaded_line(sim: Simulation) -> str:
    from urbanflow.core.types import IntersectionKind

    net = sim.network.compiled
    junctions = int((net.int_kind != IntersectionKind.boundary.code).sum())
    counts = (
        (junctions, "intersection"),
        (net.n_roads, "road"),
        (net.n_lanes, "lane"),
        (net.n_conn, "connector"),
    )
    shown = ", ".join(f"{n} {word}{'' if n == 1 else 's'}" for n, word in counts)
    return f"Loaded {sim.scenario.name}: {shown}  (hash {sim.scenario.short_hash})"


def run(
    ctx: typer.Context,
    scenario: Annotated[Path, typer.Argument(help="Scenario JSON file.")],
    duration: Annotated[
        float | None, typer.Option("--duration", help="Time limit, s (config duration).")
    ] = None,
    until: Annotated[
        float | None, typer.Option("--until", help="Stop at this simulated time, s.")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Root random seed.")] = None,
    dt: Annotated[float | None, typer.Option("--dt", help="Step length, s.")] = None,
    controller: Annotated[
        list[str] | None,
        typer.Option(
            "--controller",
            help="Signal controller: NAME for every signalized intersection, or "
            "INTERSECTION=NAME (repeatable; per-intersection options win).",
        ),
    ] = None,
    router: Annotated[str | None, typer.Option("--router", help="Registered router.")] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Run directory [default: <workspace>/runs/<run_id>]."),
    ] = None,
    no_save: Annotated[
        bool, typer.Option("--no-save", help="Do not write a run directory.")
    ] = False,
    progress: Annotated[
        bool | None,
        typer.Option(
            "--progress/--no-progress",
            help="Progress bar on stderr.",
            show_default="when stderr is a terminal",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="TOML file whose [simulation] table sets defaults "
            "[default: the workspace urbanflow.toml, if any].",
        ),
    ] = None,
    assign: Annotated[
        list[str] | None,
        typer.Option(
            "--set", help="KEY=VALUE config override (repeatable; dotted keys, JSON values)."
        ),
    ] = None,
    export: Annotated[
        Path | None,
        typer.Option(
            "--export",
            help="Also write the timeseries table here (.csv, .json or .parquet).",
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print only the result as JSON on stdout.")
    ] = False,
    debug_checks: Annotated[
        bool, typer.Option("--debug-checks", help="Check every invariant each step (slow).")
    ] = False,
) -> None:
    """Run a scenario headless and save results."""
    from urbanflow.cli.main import state
    from urbanflow.cli.output import console, emit_json, err_console, print_summary
    from urbanflow.experiments.artifacts import RunArtifacts
    from urbanflow.scenario.scenario import Scenario
    from urbanflow.simulation import Simulation

    settings = state(ctx).settings()
    loaded = Scenario.load(scenario)
    if config is None and settings.resolved_config_file.is_file():
        config = settings.resolved_config_file
    flags = {"duration": duration, "seed": seed, "dt": dt, "router": router}
    flags = {k: v for k, v in flags.items() if v is not None}
    if debug_checks:
        flags["debug_checks"] = True
    controllers = _controllers(controller or [])
    sim = Simulation(loaded, _config(loaded, config, assign or [], flags), controllers=controllers)
    if not json_output:
        console.print(_loaded_line(sim), markup=False, soft_wrap=True)
    artifacts = None
    if not no_save:
        artifacts = RunArtifacts.create(
            settings.runs_dir, name=loaded.name, seed=sim.seed, directory=out
        )
        artifacts.write_spec(
            {
                "command": "run",
                "scenario": {
                    "name": loaded.name,
                    "hash": loaded.content_hash,
                    "path": str(loaded.path),
                },
                "seed": sim.seed,
                "until": until,
                "controllers": controllers,
                "config": sim.config.model_dump(mode="json"),
            }
        )
        artifacts.write_scenario(loaded)
        artifacts.write_env(sim.config.accel)
    show_progress = sys.stderr.isatty() if progress is None else progress
    interrupted = False
    try:
        with sim:
            result = sim.run(until, progress=show_progress)
    except KeyboardInterrupt:
        interrupted = True
        result = sim.get_results()
    if artifacts is not None:
        result = artifacts.write_result(result)
    if export is not None:
        from urbanflow.metrics.export import write_table

        write_table(result.tables["timeseries"], export)
    if json_output:
        run_dir = None if artifacts is None else str(artifacts.directory)
        emit_json({**result.to_dict(), "run_dir": run_dir})
    else:
        print_summary(result.title, dict(result.rows()))
        console.print(f"State digest: {result.state_digest}", markup=False)
        if artifacts is not None:
            console.print(f"Run saved: {_shown(artifacts.directory)}", markup=False)
        if export is not None:
            console.print(f"Exported: {_shown(export)}", markup=False)
    if interrupted:
        where = "" if artifacts is None else f"; partial results in {_shown(artifacts.directory)}"
        err_console.print(
            f"Interrupted at t={result.sim_time:g} s{where}.", markup=False, soft_wrap=True
        )
        raise typer.Exit(EXIT_INTERRUPTED)


def register(app: typer.Typer) -> None:
    app.command("run", rich_help_panel="Simulation")(run)
