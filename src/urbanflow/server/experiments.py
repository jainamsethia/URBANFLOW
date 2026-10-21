"""Experiments over REST: create, queue, run, cancel, compare, export, and run artifacts
(plan M.1, N.2.6, B.2 #16).

:class:`ExperimentJobs` runs queued experiments **one at a time** (FIFO) on a coordinator
thread, each with its own :class:`~urbanflow.experiments.runner.ExperimentRunner` process
pool; only this process writes to the store. Clients poll ``GET /experiments/{id}``
(run statuses and progress, updated about once a second).

Endpoints (under ``/api/v1``):

* ``GET /experiments`` (``status``, ``limit``, ``offset``): a page of summaries
* ``POST /experiments`` ``{"spec": {...}, "run": false}``: validate (422 with JSON paths),
  expand and store as ``draft``, or queue it right away with ``"run": true``
* ``GET /experiments/{id}``, ``DELETE /experiments/{id}?purge=false`` (409 while active)
* ``POST /experiments/{id}/run`` ``{"resume": true}`` (202; 409 while active),
  ``POST /experiments/{id}/cancel`` (202; 409 unless active)
* ``GET /experiments/{id}/compare`` (``baseline``, ``metric``...),
  ``GET /experiments/{id}/timeseries`` (``column``): mean and 95 % CI over seeds per variant,
  ``GET /experiments/{id}/export?what=runs|comparison&format=csv|json``
* ``GET /runs?experiment_id=...``, ``GET /runs/{run_id}``,
  ``GET /runs/{run_id}/artifacts``, ``GET /runs/{run_id}/artifacts/{name}``: files are
  served only by matching a listed name (no path from the client is ever joined).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from urbanflow.core.errors import ConfigError
from urbanflow.core.settings import AppSettings
from urbanflow.experiments.compare import compare_experiment, comparison_csv, timeseries_overlay
from urbanflow.experiments.runner import ExperimentRunner
from urbanflow.experiments.spec import (
    ExperimentStatus,
    RunStatus,
    expand,
    new_experiment_id,
    parse_experiment,
)
from urbanflow.experiments.store import ExperimentRecord, ExperimentStore, RunRecord

__all__ = ["ExperimentJobs", "register_experiments"]

_log = logging.getLogger(__name__)

MAX_RUNS_PER_EXPERIMENT: Final = 2000
"""An experiment created over REST expands to at most this many runs."""
MAX_ARTIFACTS: Final = 500
_MEDIA: Final = {
    ".json": "application/json",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".ufr": "application/zip",
    ".parquet": "application/vnd.apache.parquet",
}


class ExperimentJobs:
    """FIFO of experiments run one at a time by a coordinator thread."""

    def __init__(self, store: ExperimentStore, settings: AppSettings) -> None:
        self.store = store
        self.settings = settings
        self._queue: deque[tuple[str, bool]] = deque()
        self._cond = threading.Condition()
        self._current: str | None = None
        self._runner: ExperimentRunner | None = None
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, name="uf-experiments", daemon=True)
        self._thread.start()

    def is_active(self, experiment_id: str) -> bool:
        with self._cond:
            return self._current == experiment_id or any(e == experiment_id for e, _ in self._queue)

    def start(self, experiment_id: str, *, resume: bool = True) -> None:
        """Queue ``experiment_id`` (``ConfigError`` if it already is queued or running)."""
        with self._cond:
            if self._current == experiment_id or any(e == experiment_id for e, _ in self._queue):
                raise ConfigError(f'experiment "{experiment_id}" is already queued or running')
            self.store.set_experiment_status(experiment_id, ExperimentStatus.queued)
            self._queue.append((experiment_id, resume))
            self._cond.notify()

    def cancel(self, experiment_id: str) -> bool:
        """Cancel a queued or running experiment; False if it is neither."""
        with self._cond:
            if self._current == experiment_id and self._runner is not None:
                self._runner.cancel()
                return True
            for item in list(self._queue):
                if item[0] == experiment_id:
                    self._queue.remove(item)
                    for run in self.store.list_runs(experiment_id, status=[RunStatus.pending]):
                        self.store.mark_run(run.id, RunStatus.cancelled)
                    self.store.set_experiment_status(experiment_id, ExperimentStatus.cancelled)
                    return True
        return False

    def shutdown(self) -> None:
        """Cancel the running experiment, drop the queue and stop the thread."""
        with self._cond:
            self._stopping = True
            for experiment_id, _ in self._queue:
                self.store.set_experiment_status(experiment_id, ExperimentStatus.interrupted)
            self._queue.clear()
            if self._runner is not None:
                self._runner.cancel()
            self._cond.notify()
        self._thread.join(timeout=60)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                experiment_id, resume = self._queue.popleft()
                self._current = experiment_id
                self._runner = ExperimentRunner(
                    self.settings.workspace,
                    self.store,
                    runs_dir=self.settings.runs_dir,
                    max_workers=self.settings.max_workers,
                )
                runner = self._runner
            try:
                runner.run(experiment_id, resume=resume)
            except Exception:
                _log.exception("experiment %s failed", experiment_id)
                self.store.set_experiment_status(experiment_id, ExperimentStatus.failed)
            finally:
                with self._cond:
                    self._current, self._runner = None, None


# ---------------------------------------------------------------------------- payloads
class ExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]
    """An experiment document (``format: "urbanflow.experiment"``)."""
    run: bool = False
    """Queue it right away."""


class ExperimentRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resume: bool = True
    """Skip runs that already completed (False: run every run again)."""


def _progress(rec: ExperimentRecord) -> dict[str, int]:
    out = {s.value: 0 for s in RunStatus}
    out.update(rec.counts)
    out["total"] = sum(rec.counts.values())
    return out


def _summary(rec: ExperimentRecord) -> dict[str, Any]:
    ref = rec.spec.scenario
    return {
        "id": rec.id,
        "name": rec.name,
        "description": rec.description,
        "status": rec.status.value,
        "scenario": ref.bundled or ref.path or ref.generator,
        "variants": [v.name for v in rec.spec.variants],
        "seeds": rec.spec.seed_list(),
        "baseline": rec.spec.baseline_name,
        "progress": _progress(rec),
        "created_at": rec.created_at.isoformat(),
        "started_at": rec.started_at.isoformat() if rec.started_at else None,
        "finished_at": rec.finished_at.isoformat() if rec.finished_at else None,
    }


def _run(r: RunRecord, *, full: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": r.id,
        "experiment_id": r.experiment_id,
        "variant": r.variant,
        "seed": r.seed,
        "status": r.status.value,
        "progress": r.progress,
        "wall_time_s": r.wall_time_s,
        "error": r.error,
        "replay": r.status is RunStatus.completed
        and (Path(r.artifacts_dir) / "replay.ufr").is_file(),
    }
    if full:
        out["summary"] = r.summary
        out["config"] = r.spec.config
        out["policy"] = r.spec.policy.model_dump(mode="json")
        out["started_at"] = r.started_at.isoformat() if r.started_at else None
        out["finished_at"] = r.finished_at.isoformat() if r.finished_at else None
    return out


def _artifacts(run: RunRecord) -> dict[str, Path]:
    """Relative POSIX name -> file, for the files of a run directory."""
    root = Path(run.artifacts_dir)
    if not root.is_dir():
        return {}
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    return {p.relative_to(root).as_posix(): p for p in files[:MAX_ARTIFACTS]}


def _runs_csv(runs: list[RunRecord]) -> str:
    metrics = sorted({k for r in runs for k in (r.summary or {})})
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["run_id", "variant", "seed", "status", "wall_time_s", *metrics])
    for r in runs:
        values = [(r.summary or {}).get(m) for m in metrics]
        w.writerow(
            [r.id, r.variant, r.seed, r.status.value, r.wall_time_s]
            + ["" if v is None else v for v in values]
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------- routes
def register_experiments(app: FastAPI, api: str, provider: Callable[[], ExperimentJobs]) -> None:
    """Add the experiment and run endpoints to ``app`` under ``api``; ``provider`` returns
    the app's job queue (created when the app starts)."""

    def _jobs() -> ExperimentJobs:
        return provider()

    def _store() -> ExperimentStore:
        return provider().store

    def _active_conflict(experiment_id: str) -> None:
        if _jobs().is_active(experiment_id):
            raise HTTPException(409, f'experiment "{experiment_id}" is queued or running')

    @app.get(f"{api}/experiments")
    def list_experiments(
        status: ExperimentStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        state = None if status is None else status.value
        records = _store().list_experiments(limit=limit, offset=offset, status=state)
        return {
            "items": [_summary(r) for r in records],
            "total": _store().count_experiments(status=state),
            "limit": limit,
            "offset": offset,
        }

    @app.post(f"{api}/experiments", status_code=201)
    def create_experiment(body: ExperimentCreate) -> dict[str, Any]:
        spec = parse_experiment(body.spec, source="spec")
        n_runs = len(spec.variants) * len(spec.seed_list())
        if n_runs > MAX_RUNS_PER_EXPERIMENT:
            raise ConfigError(
                f"the experiment has {n_runs} runs; the server accepts at most "
                f"{MAX_RUNS_PER_EXPERIMENT} (use the CLI for larger ones)"
            )
        settings = _jobs().settings
        experiment_id = new_experiment_id()
        runs = expand(spec, settings.workspace, experiment_id, settings.runs_dir)
        _store().create_experiment(experiment_id, spec, runs, status=ExperimentStatus.draft)
        if body.run:
            _jobs().start(experiment_id)
        return get_experiment(experiment_id)

    @app.get(f"{api}/experiments/{{experiment_id}}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        rec = _store().get_experiment(experiment_id)
        return {
            **_summary(rec),
            "spec": rec.spec.model_dump(mode="json"),
            "runs": [_run(r) for r in _store().list_runs(experiment_id)],
        }

    @app.delete(f"{api}/experiments/{{experiment_id}}", status_code=204)
    def delete_experiment(experiment_id: str, purge: bool = False) -> Response:
        _store().get_experiment(experiment_id)
        _active_conflict(experiment_id)
        _store().delete_experiment(experiment_id, delete_artifacts=purge)
        return Response(status_code=204)

    @app.post(f"{api}/experiments/{{experiment_id}}/run", status_code=202)
    def run_experiment(experiment_id: str, body: ExperimentRun | None = None) -> dict[str, Any]:
        _store().get_experiment(experiment_id)
        _active_conflict(experiment_id)
        _jobs().start(experiment_id, resume=(body or ExperimentRun()).resume)
        return get_experiment(experiment_id)

    @app.post(f"{api}/experiments/{{experiment_id}}/cancel", status_code=202)
    def cancel_experiment(experiment_id: str) -> dict[str, Any]:
        _store().get_experiment(experiment_id)
        if not _jobs().cancel(experiment_id):
            raise HTTPException(409, f'experiment "{experiment_id}" is not queued or running')
        return get_experiment(experiment_id)

    @app.get(f"{api}/experiments/{{experiment_id}}/compare")
    def compare(
        experiment_id: str,
        baseline: str | None = None,
        metric: Annotated[list[str] | None, Query(max_length=32)] = None,
    ) -> dict[str, Any]:
        return compare_experiment(
            _store(), experiment_id, metrics=metric or None, baseline=baseline
        )

    @app.get(f"{api}/experiments/{{experiment_id}}/timeseries")
    def timeseries(
        experiment_id: str,
        column: Annotated[str, Query(max_length=64)] = "active",
        max_points: Annotated[int, Query(ge=10, le=2000)] = 400,
    ) -> dict[str, Any]:
        return timeseries_overlay(_store(), experiment_id, column, max_points=max_points)

    @app.get(f"{api}/experiments/{{experiment_id}}/export")
    def export(
        experiment_id: str,
        what: Literal["runs", "comparison"] = "runs",
        format: Literal["csv", "json"] = "csv",
    ) -> Response:
        rec = _store().get_experiment(experiment_id)
        if what == "comparison":
            report = compare_experiment(_store(), experiment_id)
            body = comparison_csv(report) if format == "csv" else json.dumps(report, indent=2)
        else:
            runs = _store().list_runs(experiment_id)
            body = (
                _runs_csv(runs)
                if format == "csv"
                else json.dumps([_run(r, full=True) for r in runs], indent=2)
            )
        filename = f"{rec.name}-{rec.id}-{what}.{format}"
        return Response(
            body,
            media_type="text/csv" if format == "csv" else "application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get(f"{api}/runs")
    def list_runs(
        experiment_id: Annotated[str, Query(min_length=1, max_length=32)],
        status: RunStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_RUNS_PER_EXPERIMENT)] = 500,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        _store().get_experiment(experiment_id)
        runs = _store().list_runs(experiment_id, status=None if status is None else [status])
        return {
            "items": [_run(r) for r in runs[offset : offset + limit]],
            "total": len(runs),
            "limit": limit,
            "offset": offset,
        }

    @app.get(f"{api}/runs/{{run_id}}")
    def get_run(run_id: str) -> dict[str, Any]:
        return _run(_store().get_run(run_id), full=True)

    @app.get(f"{api}/runs/{{run_id}}/artifacts")
    def list_artifacts(run_id: str) -> list[dict[str, Any]]:
        files = _artifacts(_store().get_run(run_id))
        return [
            {
                "name": name,
                "size": path.stat().st_size,
                "media_type": _MEDIA.get(path.suffix, "application/octet-stream"),
            }
            for name, path in files.items()
        ]

    @app.get(f"{api}/runs/{{run_id}}/artifacts/{{name:path}}")
    def get_artifact(run_id: str, name: str) -> FileResponse:
        path = _artifacts(_store().get_run(run_id)).get(name)
        if path is None:
            raise HTTPException(404, f'run "{run_id}" has no artifact "{name}"')
        return FileResponse(
            path,
            media_type=_MEDIA.get(path.suffix, "application/octet-stream"),
            filename=path.name,
        )
