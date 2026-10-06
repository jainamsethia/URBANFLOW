"""``urbanflow replay FILE``: inspect a replay or open it in the web workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def replay(
    ctx: typer.Context,
    file: Annotated[Path, typer.Argument(help="A .ufr replay (or an unpacked .ufr.d directory).")],
    view: Annotated[
        bool, typer.Option("--view", help="Serve the workbench and open the replay viewer.")
    ] = False,
    port: Annotated[int, typer.Option("--port", help="Port for --view.")] = 8000,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the manifest as JSON.")
    ] = False,
) -> None:
    """Inspect a replay (frames, vehicles, summary) or view it in the browser."""
    import shutil

    from urbanflow.cli.main import state
    from urbanflow.cli.output import console, emit_json, print_summary
    from urbanflow.replay import ReplayReader

    reader = ReplayReader(file)
    m = reader.manifest
    if json_output:
        emit_json({k: v for k, v in m.items() if k != "chunks"})
        return
    dt = float(m.get("dt", 1.0))
    print_summary(
        f"Replay: {file.name}",
        {
            "Scenario": f"{m['scenario']['name']} ({m['scenario']['hash'][:12]})",
            "Complete": "yes" if m.get("complete") else "no (unpacked recording)",
            "Frames": f"{reader.n_frames:,} (steps {reader.steps[0]}-{reader.steps[-1]})"
            if reader.n_frames
            else "0",
            "Simulated time": f"{reader.steps[-1] * dt:g} s" if reader.n_frames else "0 s",
            "Vehicles": f"{m.get('n_vehicles', 0):,}",
            "Seed / dt": f"{m.get('seed')} / {dt:g} s",
            "Controllers": ", ".join(sorted(set(m.get("controllers", {}).values()))) or "none",
            "UrbanFlow": str(m.get("urbanflow_version")),
        },
    )
    reader.close()
    if not view:
        console.print("Open it in the workbench with --view.", markup=False)
        return
    settings = state(ctx).settings()
    settings.replays_dir.mkdir(parents=True, exist_ok=True)
    name = file.stem if file.suffix == ".ufr" else file.name.removesuffix(".ufr.d")
    target = settings.replays_dir / f"{name}.ufr"
    if file.is_file() and file.resolve() != target.resolve():
        shutil.copyfile(file, target)  # the server only opens workspace replays by name
    elif file.is_dir():
        raise typer.BadParameter("pack the recording first (it is an unpacked directory)")
    from urbanflow.cli.commands.serve import serve

    console.print(f"Viewer: http://127.0.0.1:{port}/?replay={name}", markup=False)
    serve(ctx, port=port, open_browser=True, replay=name)


def register(app: typer.Typer) -> None:
    app.command("replay", rich_help_panel="Simulation")(replay)
