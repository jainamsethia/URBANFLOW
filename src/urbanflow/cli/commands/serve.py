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
    import secrets
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
    base = state(ctx).settings()
    if not token and base.api_token is not None:  # URBANFLOW_API_TOKEN or urbanflow.toml
        token = base.api_token.get_secret_value()
    if not _is_loopback(host) and not token:
        token = secrets.token_urlsafe(16)  # never open to the network without a token
    with socket.socket() as probe:
        if probe.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0:  # noqa: S104
            raise ConfigError(f"port {port} is already in use; choose another with --port")
    overrides: dict[str, object] = {"host": host, "port": port}
    if token:
        from pydantic import SecretStr

        overrides["api_token"] = SecretStr(token)
    if dev:
        overrides["cors_origins"] = [*base.cors_origins, "http://localhost:5173"]
    settings = base.model_copy(update=overrides)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"  # noqa: S104
    if replay:
        url += f"?replay={replay}"
    if token:
        url += f"#token={token}"  # a fragment never reaches the server or its logs
    console.print("UrbanFlow workbench is running (Ctrl+C to stop). Open:", markup=False)
    console.print(f"  {url}", markup=False, soft_wrap=True)  # never wrap the link
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
