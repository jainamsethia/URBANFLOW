"""``urbanflow experiment``: run, list, show, compare and export experiments (plan AC 7.1)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer

if TYPE_CHECKING:
    from urbanflow.experiments.store import ExperimentRecord, ExperimentStore

app = typer.Typer(no_args_is_help=True, help="Run, list, show, compare and export experiments.")


def _store(ctx: typer.Context) -> tuple[ExperimentStore, Any]:
    from urbanflow.cli.main import state
    from urbanflow.experiments.store import ExperimentStore

    settings = state(ctx).settings()
    return ExperimentStore.open(settings.resolved_database_url), settings


def _counts(rec: ExperimentRecord) -> str:
    total = sum(rec.counts.values())
    done = rec.counts.get("completed", 0)
    failed = rec.counts.get("failed", 0)
    return f"{done}/{total}" + (f" ({failed} failed)" if failed else "")


@app.command("run")
def run(
    ctx: typer.Context,
    spec_file: Annotated[Path, typer.Argument(help="Experiment file (JSON).")],
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, max=61, help="Parallel processes.")
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Continue the latest experiment with this spec."),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and list the runs; run nothing.")
    ] = False,
) -> None:
    """Run every variant x seed of an experiment in parallel processes."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from urbanflow.cli.output import console, print_table
    from urbanflow.experiments.runner import ExperimentRunner, ProgressEvent
    from urbanflow.experiments.spec import expand, load_experiment

    spec = load_experiment(spec_file)
    store, settings = _store(ctx)
    if dry_run:
        runs = expand(spec, settings.workspace, "dry-run", settings.runs_dir)
        print_table(
            f"{spec.name}: {len(runs)} run(s)",
            ["Run", "Variant", "Seed", "Policy"],
            [(r.run_id, r.variant, r.seed, r.policy.controller or "scenario") for r in runs],
        )
        return
    runner = ExperimentRunner(
        settings.workspace, store, runs_dir=settings.runs_dir, max_workers=workers
    )
    experiment_id = None
    if resume:
        same = [e for e in store.list_experiments(limit=200) if e.spec == spec]
        experiment_id = same[0].id if same else None  # the latest; done runs are skipped
    if experiment_id is None:
        experiment_id = runner.submit(spec)
    total = sum(store.get_experiment(experiment_id).counts.values())
    console.print(f"Experiment {experiment_id} ({spec.name}): {total} run(s)", markup=False)
    with Progress(
        TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console
    ) as bar:
        task = bar.add_task("runs", total=total)
        done = sum(1 for r in store.list_runs(experiment_id) if r.status.value == "completed")
        bar.update(task, completed=done if resume else 0)

        def on_progress(event: ProgressEvent) -> None:
            if event.kind == "finished":
                bar.advance(task)

        try:
            status = runner.run(experiment_id, on_progress=on_progress)
        except KeyboardInterrupt:
            console.print(
                f"Interrupted; continue with: urbanflow experiment run {spec_file} --resume",
                markup=False,
            )
            raise
    rec = store.get_experiment(experiment_id)
    console.print(f"Experiment {experiment_id}: {status.value}, {_counts(rec)} runs", markup=False)
    if rec.counts.get("completed"):
        _show_comparison(store, experiment_id, None, None)


@app.command("list")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Most recent first.")] = 20,
    status: Annotated[str | None, typer.Option("--status", help="Only this status.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON only.")] = False,
) -> None:
    """List experiments, most recent first."""
    from urbanflow.cli.output import emit_json, print_table

    store, _ = _store(ctx)
    records = store.list_experiments(limit=limit, status=status)
    if json_output:
        emit_json([_record_json(r) for r in records])
        return
    print_table(
        "Experiments",
        ["Id", "Name", "Status", "Runs done", "Created (UTC)"],
        [
            (r.id, r.name, r.status.value, _counts(r), r.created_at.strftime("%Y-%m-%d %H:%M"))
            for r in records
        ],
    )


@app.command("show")
def show(
    ctx: typer.Context,
    experiment: Annotated[str, typer.Argument(help="Experiment id (or unique prefix, or name).")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON only.")] = False,
) -> None:
    """Show an experiment and its runs."""
    from urbanflow.cli.output import console, emit_json, print_table

    store, _ = _store(ctx)
    rec = store.find_experiment(experiment)
    runs = store.list_runs(rec.id)
    if json_output:
        emit_json({**_record_json(rec), "runs": [_run_json(r) for r in runs]})
        return
    console.print(
        f"{rec.id}  {rec.name}  {rec.status.value}  {_counts(rec)} runs  "
        f"baseline {rec.spec.baseline_name}",
        markup=False,
    )
    print_table(
        None,
        ["Run", "Variant", "Seed", "Status", "Wall (s)", "Error"],
        [
            (
                r.id,
                r.variant,
                r.seed,
                r.status.value,
                "" if r.wall_time_s is None else f"{r.wall_time_s:.1f}",
                r.error or "",
            )
            for r in runs
        ],
    )


@app.command("compare")
def compare(
    ctx: typer.Context,
    experiment: Annotated[str, typer.Argument(help="Experiment id (or unique prefix, or name).")],
    metric: Annotated[
        list[str] | None, typer.Option("--metric", "-m", help="Metric (repeatable).")
    ] = None,
    baseline: Annotated[
        str | None, typer.Option("--baseline", help="Variant to compare against.")
    ] = None,
    fmt: Annotated[
        Literal["table", "csv", "json"], typer.Option("--format", help="Output format.")
    ] = "table",
) -> None:
    """Means with 95 % CIs per variant and paired differences against the baseline."""
    from urbanflow.cli.output import console, emit_json
    from urbanflow.experiments.compare import compare_experiment, comparison_csv

    store, _ = _store(ctx)
    rec = store.find_experiment(experiment)
    if fmt == "table":
        _show_comparison(store, rec.id, metric, baseline)
        return
    report = compare_experiment(store, rec.id, metrics=metric or None, baseline=baseline)
    if fmt == "json":
        emit_json(report)
    else:
        console.print(comparison_csv(report), end="", markup=False, soft_wrap=True)


@app.command("export")
def export(
    ctx: typer.Context,
    experiment: Annotated[str, typer.Argument(help="Experiment id (or unique prefix, or name).")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path(),
    fmt: Annotated[Literal["csv", "json"], typer.Option("--format", help="File format.")] = "csv",
) -> None:
    """Write the runs table and the comparison (``runs.<fmt>``, ``comparison.<fmt>``)."""
    import csv
    import json

    from urbanflow.cli.output import console
    from urbanflow.experiments.compare import compare_experiment, comparison_csv

    store, _ = _store(ctx)
    rec = store.find_experiment(experiment)
    runs = store.list_runs(rec.id)
    report = compare_experiment(store, rec.id)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{rec.spec.name}-{rec.id}"
    if fmt == "json":
        runs_path = out / f"{stem}-runs.json"
        cmp_path = out / f"{stem}-comparison.json"
        runs_path.write_text(json.dumps([_run_json(r) for r in runs], indent=2), encoding="utf-8")
        cmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        runs_path = out / f"{stem}-runs.csv"
        cmp_path = out / f"{stem}-comparison.csv"
        metrics = sorted({k for r in runs for k in (r.summary or {})})
        with runs_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["run_id", "variant", "seed", "status", "wall_time_s", *metrics])
            for r in runs:
                values = [(r.summary or {}).get(m) for m in metrics]
                writer.writerow(
                    [r.id, r.variant, r.seed, r.status.value, r.wall_time_s]
                    + ["" if v is None else v for v in values]
                )
        cmp_path.write_text(comparison_csv(report), encoding="utf-8")
    console.print(f"Wrote {runs_path} and {cmp_path}", markup=False)


def _show_comparison(
    store: ExperimentStore, experiment_id: str, metrics: list[str] | None, baseline: str | None
) -> None:
    from urbanflow.cli.commands.compare import _print_table
    from urbanflow.cli.output import console
    from urbanflow.experiments.compare import compare_experiment

    report = compare_experiment(store, experiment_id, metrics=metrics or None, baseline=baseline)
    _print_table(
        {
            "scenario": report["name"],
            "controllers": report["variants"],
            "baseline": report["baseline"],
            "seeds": report["seeds"],
            "metrics": report["metrics"],
        },
        console,
    )
    if report["missing"]:
        console.print(f"Missing (not completed): {', '.join(report['missing'])}", markup=False)


def _record_json(r: ExperimentRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "name": r.name,
        "status": r.status.value,
        "counts": r.counts,
        "created_at": r.created_at.isoformat(),
        "spec": r.spec.model_dump(mode="json"),
    }


def _run_json(r: Any) -> dict[str, Any]:
    return {
        "id": r.id,
        "variant": r.variant,
        "seed": r.seed,
        "status": r.status.value,
        "wall_time_s": r.wall_time_s,
        "artifacts_dir": r.artifacts_dir,
        "summary": r.summary,
        "error": r.error,
    }


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="experiment", rich_help_panel="Research")
