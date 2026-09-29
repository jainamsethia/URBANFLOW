"""``urbanflow validate``: check scenario files and print friendly, path-addressed issues."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from urbanflow.core.errors import NotFoundError
    from urbanflow.scenario.validate import ValidationReport

EXIT_INVALID = 3
FORMATS = ("text", "json")


def _check(path: Path, deep: bool) -> ValidationReport | NotFoundError:
    """The report of one file, or the NotFoundError of a missing one (the others still run).

    ``deep`` runs :func:`urbanflow.check` (registry and network-compile checks) on top of
    the file-level validation.
    """
    from urbanflow.checks import check
    from urbanflow.core.errors import NotFoundError, ScenarioValidationError
    from urbanflow.scenario.io import parse_json, read_text
    from urbanflow.scenario.validate import ValidationReport, validate_data

    try:
        if deep:
            return check(path)
        return validate_data(parse_json(read_text(path)))
    except ScenarioValidationError as exc:  # load stage: size, encoding, JSON syntax
        return ValidationReport(exc.issues)
    except NotFoundError as exc:
        return exc


def _ok_line(report: ValidationReport) -> str:
    from urbanflow.scenario.scenario import Scenario

    if report.spec is None or report.resolved is None:  # pragma: no cover - ok implies a spec
        return "OK"
    scenario = Scenario(report.spec, report.resolved)
    counts = scenario.summary()
    shown = ", ".join(
        f"{counts[k]} {k}" for k in ("intersections", "roads", "lanes", "movements", "flows")
    )
    return f"OK: {scenario.name} (hash {scenario.short_hash}; {shown})"


def validate(
    ctx: typer.Context,
    scenarios: Annotated[list[Path], typer.Argument(help="Scenario JSON files.")],
    strict: Annotated[bool, typer.Option("--strict", help="Warnings also fail.")] = False,
    output_format: Annotated[str, typer.Option("--format", help="text|json.")] = "text",
    deep: Annotated[
        bool,
        typer.Option(
            "--deep/--no-deep",
            help="Also run the deep checks (urbanflow.check): registered model and router "
            "names and parameters, and the network compile (short or degenerate lanes, "
            "conflicting protected movements, connectors crossing twice).",
        ),
    ] = True,
) -> None:
    """Validate scenario files (schema, references, topology)."""
    from urbanflow.cli.main import state
    from urbanflow.cli.output import console, emit_json, err_console, print_issues
    from urbanflow.core.errors import ConfigError, NotFoundError, ScenarioValidationError

    if output_format not in FORMATS:
        raise ConfigError(f"--format must be text or json (got {output_format!r})")
    debug = state(ctx).debug
    reports = [(path, _check(path, deep)) for path in scenarios]
    failed = missing = False
    files: list[dict[str, Any]] = []
    for path, report in reports:
        if isinstance(report, NotFoundError):
            missing = True
            if output_format == "json":
                gone = {"path": str(path), "ok": False, "error": str(report)}
                files.append(gone | {"errors": [], "warnings": []})
            else:
                err_console.print(f"Error: {report}", markup=False, highlight=False, soft_wrap=True)
            continue
        bad = bool(report.errors or (strict and report.warnings))
        failed |= bad
        if output_format == "json":
            entry: dict[str, Any] = {
                "path": str(path),
                "ok": not bad,
                "errors": [i.to_dict() for i in report.errors],
                "warnings": [i.to_dict() for i in report.warnings],
            }
            if not bad and report.spec is not None and report.resolved is not None:
                from urbanflow.scenario.scenario import Scenario

                scenario = Scenario(report.spec, report.resolved)
                entry |= {"name": scenario.name, "hash": scenario.content_hash}
                entry["counts"] = scenario.summary()
            files.append(entry)
            continue
        if len(reports) > 1:
            err_console.print(f"{path}:", markup=False, highlight=False, soft_wrap=True)
        if bad:
            try:
                report.raise_for_errors(strict=strict, source=str(path))
            except ScenarioValidationError as exc:
                if debug:
                    err_console.print("".join(traceback.format_exception(exc)), markup=False)
                print_issues(exc.errors)
                if not strict and report.warnings:
                    print_issues(report.warnings, header="Warnings:")
            continue
        if report.warnings:
            print_issues(report.warnings, header="Warnings:")
        console.print(_ok_line(report), markup=False, soft_wrap=True)
    if output_format == "json":
        emit_json({"ok": not (failed or missing), "files": files})
    if missing:
        raise typer.Exit(NotFoundError.exit_code)
    if failed:
        raise typer.Exit(EXIT_INVALID)


def register(app: typer.Typer) -> None:
    app.command("validate", rich_help_panel="Scenarios")(validate)
