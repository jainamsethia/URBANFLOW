"""``urbanflow benchmark``: measure simulation speed on fixed workloads (plan S.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

EXIT_REGRESSION = 10


def benchmark(
    target: Annotated[
        list[str] | None,
        typer.Argument(
            help="Suites (grid_scaling, vehicle_sweep, overhead) or scenario files; "
            "default grid_scaling."
        ),
    ] = None,
    list_: Annotated[bool, typer.Option("--list", help="List the suites and cases.")] = False,
    quick: Annotated[
        bool, typer.Option("--quick", help="60 s warm-up and 300 s timed (else 300 + 1500 s).")
    ] = False,
    repeat: Annotated[int, typer.Option("--repeat", min=1, max=50, help="Repeats per case.")] = 5,
    case: Annotated[
        list[str] | None, typer.Option("--case", help="Only this case (repeatable).")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the JSON report.")
    ] = None,
    compare_to: Annotated[
        Path | None,
        typer.Option("--compare", help="Baseline report: exit 10 if a case got slower."),
    ] = None,
    threshold: Annotated[
        float, typer.Option("--threshold", min=0.01, max=0.99, help="Slowdown that counts.")
    ] = 0.10,
    profile: Annotated[
        Path | None, typer.Option("--profile", help="Write a cProfile file per case here.")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the JSON report only.")
    ] = False,
) -> None:
    """Measure steps/s, vehicle-steps/s, real-time factor and memory on fixed workloads."""
    from urbanflow.benchmark import (
        SUITES,
        BenchmarkCase,
        compare,
        load_report,
        machine_key,
        run_suites,
        save_report,
        scenario_case,
    )
    from urbanflow.cli.output import console, emit_json
    from urbanflow.core.errors import ConfigError, NotFoundError

    if list_:
        for suite, members in SUITES.items():
            console.print(f"{suite}: {', '.join(c.name for c in members)}", markup=False)
        console.print(f"This machine's key: {machine_key()}", markup=False)
        return
    targets = target or ["grid_scaling"]
    cases: list[BenchmarkCase] = []
    for t in targets:
        if t in SUITES:
            cases += SUITES[t]
        elif Path(t).is_file():
            cases.append(scenario_case(t))
        else:
            raise NotFoundError(f'"{t}" is neither a suite ({", ".join(SUITES)}) nor a file')
    if case:
        unknown = sorted(set(case) - {c.name for c in cases})
        if unknown:
            raise ConfigError(f"unknown case(s) {unknown}; choose from {[c.name for c in cases]}")
        cases = [c for c in cases if c.name in case]
    baseline = load_report(compare_to) if compare_to is not None else None
    if baseline is not None and baseline.machine_key != machine_key():
        console.print(
            f"Note: the baseline was measured on {baseline.machine_key}, this is {machine_key()}.",
            markup=False,
        )
    if not json_output:
        console.print(
            f"{' + '.join(targets)}: {len(cases)} case(s) x {repeat} repeat(s)"
            f"{' (quick)' if quick else ''}, "
            "one fresh process per repeat ...",
            markup=False,
        )
    report = run_suites(
        cases,
        repeats=repeat,
        quick=quick,
        profile_dir=profile,
        on_case=None if json_output else lambda r: console.print(_line(r), markup=False),
    )
    if output is not None:
        save_report(report, output)
    regressions = compare(baseline, report, threshold) if baseline is not None else []
    if json_output:
        emit_json(report.model_dump(mode="json"))
    else:
        _print_table(report, console)
        if output is not None:
            console.print(f"Wrote {output}", markup=False)
        for r in regressions:
            console.print(
                f"REGRESSION {r.suite}/{r.case}: {r.metric} {r.baseline:,.0f} -> "
                f"{r.current:,.0f} ({r.change_pct:+.1f} %)",
                markup=False,
            )
        if baseline is not None and not regressions:
            console.print(f"No case is more than {threshold:.0%} slower than the baseline.")
    if regressions:
        raise typer.Exit(EXIT_REGRESSION)


def _line(r: Any) -> str:
    flag = "  NONDETERMINISTIC" if r.nondeterministic else ""
    return (
        f"  {r.suite}/{r.case}: {r.steps_per_s.median:,.1f} steps/s, "
        f"{r.mean_active:,.0f} vehicles{flag}"
    )


def _print_table(report: Any, console: Any) -> None:
    from rich.table import Table

    table = Table(title=f"UrbanFlow benchmark ({report.machine_key}, numpy backend)")
    for name in ("case", "junctions", "vehicles", "steps/s", "veh-steps/s", "x real time",
                 "compile s", "peak MB", "same digest"):  # fmt: skip
        table.add_column(name, justify="left" if name == "case" else "right", overflow="fold")
    for r in report.cases:
        table.add_row(
            f"{r.suite}/{r.case}",
            str(r.n_intersections),
            f"{r.mean_active:,.0f}",
            f"{r.steps_per_s.median:,.1f}",
            f"{r.vehicle_steps_per_s.median:,.0f}",
            f"{r.rtf.median:,.1f}",
            f"{r.compile_s.median:.2f}",
            f"{r.peak_rss_mb.median:,.0f}",
            "no" if r.nondeterministic else "yes",
        )
    console.print(table)
    console.print("Medians over the repeats (quartiles and every value in the JSON report).")


def register(app: typer.Typer) -> None:
    app.command("benchmark", rich_help_panel="Research")(benchmark)
