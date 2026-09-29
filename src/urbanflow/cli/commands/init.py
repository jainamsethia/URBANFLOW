"""``urbanflow init``: scaffold a workspace from a bundled scenario (plan AC 7.1, B.2 #21)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Annotated, Final

import typer

DEFAULT_TEMPLATE: Final = "single_intersection"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")

CONFIG_TOML: Final = """\
# UrbanFlow workspace settings. Environment variables (URBANFLOW_*) override [server]
# and [logging]; they never change [simulation].

[simulation]
# Defaults for every run in this workspace (overridden by run flags and --set).
# dt = 1.0
# duration = 3600
# seed = 0

[server]
host = "127.0.0.1"
port = 8000

[logging]
level = "INFO"
format = "text"
"""

GITIGNORE: Final = "runs/\n.urbanflow/\n"

README: Final = """\
# UrbanFlow workspace

- `scenarios/`: scenario files
- `experiments/`: experiment specs
- `runs/`: run outputs (git-ignored)
- `urbanflow.toml`: workspace settings

Useful commands:

    urbanflow validate scenarios/{name}.json
    urbanflow generate --list
    urbanflow schema scenario -o scenario.schema.json
"""


def _experiment(name: str) -> str:
    """An example experiment spec (format ``urbanflow.experiment`` 1.0, plan Q.1)."""
    spec = {
        "format": "urbanflow.experiment",
        "version": "1.0",
        "name": "example",
        "description": f"Lane changing on and off on scenarios/{name}.json, 3 seeds.",
        "scenario": {"path": f"scenarios/{name}.json"},
        "variants": [
            {"name": "baseline"},
            {"name": "no_lane_change", "config": {"lane_changing": False}},
        ],
        "seeds": {"start": 0, "count": 3},
    }
    return json.dumps(spec, indent=2) + "\n"


def init(
    ctx: typer.Context,
    directory: Annotated[
        Path | None,
        typer.Argument(help="Workspace directory [default: --workspace or the current one]."),
    ] = None,
    template: Annotated[
        str, typer.Option("--template", "-t", help="Bundled scenario to copy.")
    ] = DEFAULT_TEMPLATE,
    name: Annotated[
        str | None, typer.Option("--name", help="Scenario file name [default: the template].")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Scaffold into a non-empty directory.")
    ] = False,
) -> None:
    """Scaffold a workspace (config, example scenario, dirs)."""
    from urbanflow.cli.main import state
    from urbanflow.cli.output import console
    from urbanflow.core.errors import ConfigError
    from urbanflow.scenario.io import bundled

    source = bundled(template)  # NotFoundError lists the bundled names
    stem = name or template
    if not _SAFE_NAME.fullmatch(stem):
        raise ConfigError(f"--name must be a plain file name (letters, digits, _ . -): {stem!r}")
    root = (directory or state(ctx).workspace or Path.cwd()).resolve()
    if root.is_dir() and any(root.iterdir()) and not force:
        raise ConfigError(
            f"{root} is not empty; use --force to scaffold into it (existing files are kept)"
        )
    files: dict[str, str | Path] = {
        "urbanflow.toml": CONFIG_TOML,
        f"scenarios/{stem}.json": source,
        "experiments/example.json": _experiment(stem),
        ".gitignore": GITIGNORE,
        "README.md": README.format(name=stem),
    }
    (root / "runs").mkdir(parents=True, exist_ok=True)
    lines = [f"Initialised workspace {root}"]
    for rel, content in files.items():
        target = root / rel
        if target.exists():
            lines.append(f"  {rel}  (exists, kept)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            shutil.copyfile(content, target)
        else:
            target.write_text(content, encoding="utf-8", newline="\n")
        lines.append(f"  {rel}")
    lines.append("  runs/")
    lines += [
        "Next steps:",
        f"  cd {root}",
        f"  urbanflow validate scenarios/{stem}.json",
        "  urbanflow generate --list",
    ]
    console.print("\n".join(lines), markup=False, highlight=False, soft_wrap=True)


def register(app: typer.Typer) -> None:
    app.command("init", rich_help_panel="Scenarios")(init)
