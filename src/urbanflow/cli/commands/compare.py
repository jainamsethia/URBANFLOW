"""``urbanflow compare``: controllers x seeds on one scenario, with paired statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer


def compare(
    ctx: typer.Context,
    scenario: Annotated[str, typer.Argument(help="Bundled scenario name or a scenario file.")],
    controller: Annotated[
        list[str],
        typer.Option(
            "--controller", "-c", help="Controller to compare (repeatable; first = baseline)."
        ),
    ],
    seeds: Annotated[int, typer.Option("--seeds", min=1, help="Seeds per controller.")] = 3,
    duration: Annotated[
        float | None, typer.Option("--duration", help="Simulated duration, s.")
    ] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Parallel processes.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON only.")] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Also write the JSON here.")] = None,
) -> None:
    """Compare signal controllers on one scenario (paired over common seeds)."""
    import json

    from urbanflow.cli.main import state
    from urbanflow.cli.output import console, emit_json
    from urbanflow.core.errors import ConfigError
    from urbanflow.experiments.compare import compare_controllers
    from urbanflow.scenario.io import bundled, bundled_names
    from urbanflow.scenario.scenario import Scenario

    if len(set(controller)) != len(controller):
        raise ConfigError("--controller values must be distinct")
    settings = state(ctx).settings()
    path = bundled(scenario) if scenario in bundled_names() else Path(scenario)
    loaded = Scenario.load(path)
    if not json_output:
        console.print(
            f"Running {len(controller)} controller(s) x {seeds} seed(s) on {loaded.name} ...",
            markup=False,
        )
    result = compare_controllers(
        loaded,
        controller,
        seeds=seeds,
        duration=duration,
        max_workers=workers or settings.max_workers,
    )
    if out is not None:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if json_output:
        emit_json(result)
        return
    _print_table(result, console)
    if out is not None:
        console.print(f"Wrote {out}", markup=False)


def _print_table(result: dict[str, Any], console: Any) -> None:
    from rich.table import Table

    base = result["baseline"]
    others = [c for c in result["controllers"] if c != base]
    table = Table(title=f"{result['scenario']}: {len(result['seeds'])} seed(s), baseline {base}")
    table.add_column("Metric")
    for c in result["controllers"]:
        table.add_column(c, justify="right")
    for c in others:
        table.add_column(f"{c} vs {base}", justify="right")
    for row in result["metrics"]:
        cells = [_fmt(**row["by_controller"][c]) for c in result["controllers"]]
        for c in others:
            d = row["vs_baseline"][c]
            pct = "" if d["delta_pct"] is None else f" ({d['delta_pct']:+.1f}%)"
            mark = " *" if d["significant"] else ""
            cells.append("n/a" if d["delta"] is None else f"{d['delta']:+.2f}{pct}{mark}")
        table.add_row(row["metric"], *cells)
    console.print(table)
    console.print("mean [95% CI]; * = paired 95% CI excludes 0", markup=False)


def _fmt(mean: float | None, ci95: list[float | None], **_: Any) -> str:
    if mean is None:
        return "n/a"
    if ci95[0] is None or ci95[1] is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} [{ci95[0]:.2f}, {ci95[1]:.2f}]"


def register(app: typer.Typer) -> None:
    app.command("compare", rich_help_panel="Research")(compare)
