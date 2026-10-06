"""``urbanflow serve``: start the web workbench (API + UI)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def serve(
    ctx: typer.Context,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="TCP port.")] = 8000,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the UI in a browser.")
    ] = False,
    dev: Annotated[
        bool, typer.Option("--dev", help="Allow the Vite dev server (localhost:5173).")
    ] = False,
    token: Annotated[str | None, typer.Option("--token", help="Require this API token.")] = None,
    token_file: Annotated[
        Path | None, typer.Option("--token-file", help="Read the API token from a file.")
    ] = None,
    replay: Annotated[str | None, typer.Option(hidden=True)] = None,
) -> None:
    """Start the web workbench (API + UI)."""
    import socket
    import webbrowser

    import uvicorn

    from urbanflow.cli.main import state
    from urbanflow.cli.output import console
    from urbanflow.core.errors import ConfigError
    from urbanflow.server.app import _is_loopback, create_app

    if token and token_file:
        raise ConfigError("give --token or --token-file, not both")
    if token_file is not None:
        token = token_file.read_text(encoding="utf-8").strip()
    if not _is_loopback(host) and not token:
        raise ConfigError(
            f"refusing to listen on {host} without an API token; use --token/--token-file "
            "or bind to 127.0.0.1"
        )
    with socket.socket() as probe:
        if probe.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0:  # noqa: S104
            raise ConfigError(f"port {port} is already in use; choose another with --port")
    base = state(ctx).settings()
    overrides: dict[str, object] = {"host": host, "port": port}
    if token:
        overrides["api_token"] = token
    if dev:
        overrides["cors_origins"] = [*base.cors_origins, "http://localhost:5173"]
    settings = base.model_copy(update=overrides)
    if token:
        from pydantic import SecretStr

        settings = settings.model_copy(update={"api_token": SecretStr(token)})
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"  # noqa: S104
    if replay:
        url += f"?replay={replay}"
    console.print(f"UrbanFlow workbench on {url}  (Ctrl+C to stop)", markup=False)
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(
        create_app(settings),
        host=host,
        port=port,
        ws_max_size=65536,
        ws_per_message_deflate=False,
        log_level="warning",
    )


def register(app: typer.Typer) -> None:
    app.command("serve", rich_help_panel="Workbench")(serve)
