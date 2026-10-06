"""Typer root of the ``urbanflow`` command and the error → exit-code mapping.

Exit codes: 0 success, 1 internal/doctor failure, 2 usage error, 3 scenario invalid,
4 config error, 5 not found, 6 simulation/command error, 7 invariant violation,
8 replay format error, 9 missing optional dependency, 10 benchmark regression,
130 interrupted.
"""

from __future__ import annotations

import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import click
import typer
from rich.markup import escape

from urbanflow._version import __version__
from urbanflow.cli.output import err_console
from urbanflow.core.errors import UrbanFlowError

if TYPE_CHECKING:
    from urbanflow.core.settings import AppSettings

__all__ = ["app", "main"]

EXIT_INTERRUPTED = 130


@dataclass
class CliState:
    """Per-invocation global options, stored on the Click context (no module globals)."""

    workspace: Path | None = None
    log_level: str = "WARNING"
    log_format: str = "text"
    debug: bool = False
    _settings: AppSettings | None = field(default=None, repr=False)

    def settings(self, **overrides: Any) -> AppSettings:
        """Load settings once per invocation (overrides are applied on top)."""
        from urbanflow.core.settings import load_settings

        if overrides:
            return load_settings(workspace=self.workspace, **overrides)
        if self._settings is None:
            self._settings = load_settings(workspace=self.workspace)
        return self._settings


def platform_tag() -> str:
    """Short platform tag such as ``win-arm64``, ``linux-x86_64`` or ``macos-arm64``."""
    system = {"win32": "win", "darwin": "macos"}.get(sys.platform, sys.platform)
    return f"{system}-{platform.machine().lower() or 'unknown'}"


def _version_string() -> str:
    try:
        from importlib.metadata import version

        numpy_version = version("numpy")
    except Exception:  # numpy missing is reported by `doctor`, not here
        numpy_version = "n/a"
    impl = platform.python_implementation()
    return (
        f"urbanflow {__version__} ({impl} {platform.python_version()}, "
        f"numpy {numpy_version}, {platform_tag()})"
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(_version_string())
        raise typer.Exit()


app = typer.Typer(
    name="urbanflow",
    help="UrbanFlow - microscopic city traffic simulation and research workbench.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def _root(
    ctx: typer.Context,
    version: Annotated[  # noqa: ARG001 - eager option handled by its callback
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            envvar="URBANFLOW_WORKSPACE",
            help="Workspace directory (scenarios, runs, replays).",
            show_default="current directory",
        ),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="-v INFO, -vv DEBUG (default WARNING)."),
    ] = 0,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="DEBUG|INFO|WARNING|ERROR (overrides -v)."),
    ] = None,
    log_format: Annotated[
        str,
        typer.Option("--log-format", help="text|json."),
    ] = "text",
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks.")] = False,
) -> None:
    """UrbanFlow - microscopic city traffic simulation and research workbench."""
    from urbanflow.core.errors import ConfigError
    from urbanflow.core.logging import configure_logging

    level = (
        log_level or ("DEBUG" if verbose >= 2 else "INFO" if verbose == 1 else "WARNING")
    ).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError(f"--log-level must be DEBUG, INFO, WARNING or ERROR (got {log_level!r})")
    if log_format not in {"text", "json"}:
        raise ConfigError(f"--log-format must be text or json (got {log_format!r})")
    configure_logging(level, log_format)  # type: ignore[arg-type]
    ctx.obj = CliState(workspace=workspace, log_level=level, log_format=log_format, debug=debug)


def state(ctx: typer.Context) -> CliState:
    """Return the per-invocation :class:`CliState` (creating a default one in tests)."""
    root = ctx.find_root()
    if not isinstance(root.obj, CliState):
        root.obj = CliState()
    return root.obj


def _register_commands() -> None:
    from urbanflow.cli.commands import (
        compare,
        doctor,
        generate,
        init,
        run,
        schema,
        serve,
        validate,
    )

    for module in (init, validate, generate, schema, run, compare, serve, doctor):
        module.register(app)


_register_commands()


def _report(exc: BaseException, debug: bool) -> None:
    if debug:
        err_console.print("".join(traceback.format_exception(exc)), markup=False, highlight=False)
    message = str(exc).strip() or type(exc).__name__
    first, _, rest = message.partition("\n")
    err_console.print(f"[bold red]Error:[/] {escape(first)}", highlight=False, soft_wrap=True)
    if rest:
        err_console.print(rest, markup=False, highlight=False, soft_wrap=True)


def _is_click_exception(exc: BaseException) -> bool:
    """Usage/parameter errors from click or from Typer's vendored copy (``typer._click``)."""
    root = type(exc).__module__.split(".")[0]
    return (
        root in {"click", "typer"}
        and hasattr(exc, "show")
        and isinstance(getattr(exc, "exit_code", None), int)
    )


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point with friendly errors and stable exit codes."""
    args = list(sys.argv[1:] if argv is None else argv)
    debug = "--debug" in args
    try:
        result = app(args=args, prog_name="urbanflow", standalone_mode=False)
    except (click.exceptions.Abort, typer.Abort):
        err_console.print("Aborted.")
        code = EXIT_INTERRUPTED
    except KeyboardInterrupt:
        err_console.print("Interrupted.")
        code = EXIT_INTERRUPTED
    except UrbanFlowError as exc:
        _report(exc, debug)
        code = exc.exit_code
    except Exception as exc:
        if _is_click_exception(exc):
            if type(exc).__name__ != "NoArgsIsHelpError":  # help text was already printed
                exc.show()  # type: ignore[attr-defined]
                code = int(exc.exit_code)  # type: ignore[attr-defined]
            else:
                code = 0
        elif debug:
            _report(exc, True)
            code = 1
        else:
            err_console.print(
                f"[bold red]Internal error:[/] {type(exc).__name__}: {exc} "
                "(re-run with --debug and report this)",
                highlight=False,
            )
            code = 1
    else:
        code = result if isinstance(result, int) else 0
    raise SystemExit(code)
