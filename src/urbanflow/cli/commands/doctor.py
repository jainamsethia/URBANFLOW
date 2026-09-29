"""``urbanflow doctor``: check installation, optional extras, frontend, port and workspace."""

from __future__ import annotations

import json
import platform
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

if TYPE_CHECKING:
    from urbanflow.core.settings import AppSettings

Status = Literal["OK", "WARN", "FAIL"]

CORE_DISTRIBUTIONS = (
    "numpy",
    "pydantic",
    "pydantic-settings",
    "networkx",
    "typer",
    "rich",
    "sqlalchemy",
    "psutil",
    "fastapi",
    "uvicorn",
    "websockets",
)

EXTRAS: dict[str, tuple[str, ...]] = {
    "rl": ("gymnasium", "pettingzoo"),
    "rl-train": ("stable-baselines3", "torch"),
    "data": ("polars", "pandas"),
    "accel": ("numba",),
    "postgres": ("psycopg",),
}

_WIN_ARM64 = sys.platform == "win32" and platform.machine().lower() in {"arm64", "aarch64"}
_EXTRA_HINTS = {
    ("rl-train", True): "no win_arm64 torch wheel; use an x64 Python env (see docs)",
    ("accel", True): "numba win_arm64 wheels exist for Python 3.14 only",
    ("postgres", True): "install psycopg and put libpq.dll on PATH",
}


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    details: str


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def check_python() -> Check:
    ok = sys.version_info >= (3, 12)
    impl = f"{platform.python_implementation()} {platform.python_version()}"
    return Check("Python", "OK" if ok else "FAIL", impl if ok else f"{impl} (need >= 3.12)")


def check_platform() -> Check:
    return Check("Platform", "OK", f"{platform.platform(terse=True)}, {platform.machine()}")


def check_urbanflow() -> Check:
    from urbanflow._version import __version__

    detail = __version__
    try:
        direct = metadata.distribution("urbanflow").read_text("direct_url.json")
        if direct and json.loads(direct).get("dir_info", {}).get("editable"):
            detail += " (editable)"
    except metadata.PackageNotFoundError:
        detail += " (not installed; running from source)"
    return Check("urbanflow", "OK", detail)


def check_core() -> Check:
    found: list[str] = []
    missing: list[str] = []
    for dist in CORE_DISTRIBUTIONS:
        v = _version(dist)
        (found if v else missing).append(f"{dist} {v}" if v else dist)
    if missing:
        return Check("Core dependencies", "FAIL", "missing: " + ", ".join(missing))
    return Check("Core dependencies", "OK", ", ".join(found))


def check_extras() -> list[Check]:
    checks = []
    for extra, dists in EXTRAS.items():
        versions = {d: _version(d) for d in dists}
        missing = [d for d, v in versions.items() if v is None]
        present = ", ".join(f"{d} {v}" for d, v in versions.items() if v)
        if not missing:
            checks.append(Check(f"Extra {extra}", "OK", present))
            continue
        hint = _EXTRA_HINTS.get((extra, _WIN_ARM64), f'pip install "urbanflow[{extra}]"')
        detail = f"{', '.join(missing)} missing ({hint})"
        if present:
            detail = f"{present}; {detail}"
        checks.append(Check(f"Extra {extra}", "WARN", detail))
    return checks


def _frontend_candidates(settings: AppSettings) -> list[Path]:
    candidates = []
    if settings.frontend_dir:
        candidates.append(settings.frontend_dir)
    spec = find_spec("urbanflow")
    if spec and spec.submodule_search_locations:
        candidates += [Path(p) / "_frontend" for p in spec.submodule_search_locations]
    return candidates


def check_frontend(settings: AppSettings) -> Check:
    for directory in _frontend_candidates(settings):
        index = directory / "index.html"
        if index.is_file():
            size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
            return Check("Frontend bundle", "OK", f"{index} ({size / 1e6:.1f} MB)")
    return Check(
        "Frontend bundle",
        "WARN",
        "not built (run: npm --prefix frontend ci && npm --prefix frontend run build)",
    )


def check_port(host: str, port: int) -> Check:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return Check(f"Port {port}", "WARN", f"in use on {host} (use --port)")
    return Check(f"Port {port}", "OK", f"free on {host}")


def check_workspace(settings: AppSettings) -> Check:
    ws = settings.workspace
    try:
        ws.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=ws, prefix=".urbanflow-doctor-", delete=True):
            pass
    except OSError as exc:
        return Check("Workspace", "FAIL", f"{ws} not writable: {exc}")
    return Check("Workspace", "OK", f"{ws} writable")


def check_database(settings: AppSettings) -> Check:
    url = settings.resolved_database_url
    if url.startswith("sqlite"):
        return Check("Database", "OK", f"sqlite ({settings.state_dir / 'urbanflow.db'})")
    if url.startswith("postgresql"):
        if _version("psycopg") is None:
            hint = _EXTRA_HINTS.get(("postgres", _WIN_ARM64), 'pip install "urbanflow[postgres]"')
            return Check("Database", "WARN", f"postgresql configured but psycopg missing ({hint})")
        return Check("Database", "OK", "postgresql (psycopg installed)")
    return Check("Database", "WARN", f"unrecognised database URL scheme: {url.split(':', 1)[0]}")


def _ping() -> int:
    """Target for the spawn round-trip check (must be importable at module top level)."""
    return 42


def check_spawn() -> Check:
    import multiprocessing as mp

    start = time.perf_counter()
    try:
        with mp.get_context("spawn").Pool(1) as pool:
            ok = pool.apply(_ping) == 42
    except Exception as exc:  # report, don't crash doctor
        return Check("Process spawn", "FAIL", f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - start
    return Check("Process spawn", "OK" if ok else "FAIL", f"round trip {elapsed:.2f} s")


def run_checks(settings: AppSettings, *, port: int, deep: bool = False) -> list[Check]:
    checks = [check_python(), check_platform(), check_urbanflow(), check_core()]
    checks += check_extras()
    checks += [
        check_frontend(settings),
        check_port(settings.host, port),
        check_workspace(settings),
        check_database(settings),
    ]
    if deep:
        checks.append(check_spawn())
    return checks


def doctor(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON only.")] = False,
    port: Annotated[int | None, typer.Option("--port", help="Port to check.")] = None,
    deep: Annotated[
        bool, typer.Option("--deep", help="Also test a multiprocessing spawn round trip.")
    ] = False,
) -> None:
    """Check installation, extras, frontend, ports and workspace."""
    from urbanflow.cli.main import state
    from urbanflow.cli.output import console, emit_json, print_table

    settings = state(ctx).settings()
    checks = run_checks(settings, port=port or settings.port, deep=deep)
    warnings = sum(c.status == "WARN" for c in checks)
    failures = sum(c.status == "FAIL" for c in checks)
    if as_json:
        emit_json(
            {"checks": [asdict(c) for c in checks], "warnings": warnings, "failures": failures}
        )
    else:
        style = {"OK": "[green]OK[/]", "WARN": "[yellow]WARN[/]", "FAIL": "[bold red]FAIL[/]"}
        print_table(
            "UrbanFlow doctor",
            ("Check", "Status", "Details"),
            [(c.name, style[c.status], c.details) for c in checks],
        )
        console.print(f"{warnings} warning(s), {failures} failure(s).")
    if failures:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    app.command("doctor", rich_help_panel="Workbench")(doctor)
