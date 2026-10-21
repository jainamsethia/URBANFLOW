"""Run experiments in a pool of spawned worker processes (plan Q.2).

:meth:`ExperimentRunner.submit` expands an experiment into runs and stores them;
:meth:`ExperimentRunner.run` executes the runs that have not completed yet (``resume``) in a
``ProcessPoolExecutor`` with the spawn context and one run per worker process
(``max_tasks_per_child=1``: isolation, freed memory, Windows-safe). Progress arrives through
a spawn-context queue; :meth:`ExperimentRunner.cancel` (thread-safe) sets a spawn-context
event that running workers check between chunks, and pending runs are not started.

A worker that dies (a crash, the OOM killer, a kill) breaks the pool, and every run in
flight then fails the same way, so the culprit cannot be told apart. The runs that had
started are re-run one at a time in single-worker pools: a run that breaks its own pool is
marked failed (``error.txt``: the worker process died); the others complete normally. Runs
that had not started go back into a fresh pool.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Final

from urbanflow.experiments.execute import execute_run, init_worker
from urbanflow.experiments.spec import (
    ExperimentSpec,
    ExperimentStatus,
    RunStatus,
    expand,
    new_experiment_id,
)
from urbanflow.experiments.store import ExperimentStore, RunRecord

__all__ = ["ExperimentRunner", "ProgressEvent"]

POLL_S: Final = 0.2
DB_PROGRESS_INTERVAL_S: Final = 1.0
WORKER_DIED: Final = "the worker process died (crashed or was killed)"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """``kind``: ``started`` (with the worker ``pid``), ``progress`` (``fraction`` of the
    run, NaN if unknown) or ``finished`` (with the run's ``status``)."""

    run_id: str
    kind: str
    fraction: float = 0.0
    pid: int | None = None
    status: RunStatus | None = None


class ExperimentRunner:
    """Executes stored experiments; ``runs_dir`` defaults to ``<workspace>/runs``."""

    def __init__(
        self,
        workspace: str | Path,
        store: ExperimentStore,
        *,
        runs_dir: str | Path | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.store = store
        self.runs_dir = Path(runs_dir) if runs_dir is not None else self.workspace / "runs"
        self.max_workers = max_workers
        self._ctx = get_context("spawn")
        self._cancel = self._ctx.Event()
        self._cancelled = threading.Event()

    def submit(self, spec: ExperimentSpec) -> str:
        """Expand ``spec`` into runs (validating every config) and store them; its id."""
        experiment_id = new_experiment_id()
        runs = expand(spec, self.workspace, experiment_id, self.runs_dir)
        self.store.create_experiment(experiment_id, spec, runs)
        return experiment_id

    def cancel(self) -> None:
        """Stop: running workers stop at their next check, pending runs never start."""
        self._cancelled.set()
        self._cancel.set()

    # ------------------------------------------------------------------ run
    def run(
        self,
        experiment_id: str,
        *,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        resume: bool = True,
    ) -> ExperimentStatus:
        """Execute the runs of ``experiment_id`` that are not completed (all with
        ``resume=False``); blocks until they finish; the final status."""
        record = self.store.get_experiment(experiment_id)
        runs = self.store.list_runs(experiment_id)
        todo = [r for r in runs if not (resume and r.status is RunStatus.completed)]
        workers = max(1, min(self.max_workers or record.spec.max_workers or 4, len(todo) or 1))
        self._notify = on_progress
        self._last_db: dict[str, float] = {}
        self.store.set_experiment_status(experiment_id, ExperimentStatus.running)
        progress = self._ctx.Queue()
        try:
            remaining = todo
            while remaining and not self._cancelled.is_set():
                suspects, remaining = self._pool(remaining, workers, progress)
                for run in suspects:
                    if self._cancelled.is_set():
                        remaining.append(run)
                        continue
                    self._pool([run], 1, progress, isolated=True)
            for run in remaining:  # cancelled before they started
                self.store.mark_run(run.id, RunStatus.cancelled)
                self._emit(ProgressEvent(run.id, "finished", status=RunStatus.cancelled))
        except BaseException:
            for run in self.store.list_runs(experiment_id, status=[RunStatus.running]):
                self.store.mark_run(run.id, RunStatus.interrupted)
            self.store.set_experiment_status(experiment_id, ExperimentStatus.interrupted)
            raise
        finally:
            progress.close()
        status = self._final_status(experiment_id)
        self.store.set_experiment_status(experiment_id, status)
        return status

    def _pool(
        self, runs: Sequence[RunRecord], workers: int, progress: Any, *, isolated: bool = False
    ) -> tuple[list[RunRecord], list[RunRecord]]:
        """Run ``runs`` in one pool. If it breaks: ``(started, not started)`` of the runs
        left over (an isolated run that breaks its pool is failed instead); else empty."""
        started: set[str] = set()
        pool = ProcessPoolExecutor(
            workers,
            mp_context=self._ctx,
            max_tasks_per_child=1,
            initializer=init_worker,
            initargs=(progress, self._cancel),
        )
        futures: dict[Future[dict[str, Any]], RunRecord] = {}
        try:
            for run in runs:
                futures[pool.submit(execute_run, run.spec.model_dump_json())] = run
            while futures:
                done, _ = wait(futures, timeout=POLL_S, return_when=FIRST_COMPLETED)
                self._drain(progress, started)
                if self._cancelled.is_set():
                    for f in [f for f in futures if f.cancel()]:
                        run = futures.pop(f)
                        self.store.mark_run(run.id, RunStatus.cancelled)
                        self._emit(ProgressEvent(run.id, "finished", status=RunStatus.cancelled))
                for f in done:
                    run = futures.pop(f)
                    try:
                        outcome = f.result()
                    except BrokenProcessPool:
                        left = [run, *futures.values()]
                        time.sleep(POLL_S)  # late "started" messages
                        self._drain(progress, started)
                        if isolated:
                            self._died(run)
                            return [], []
                        return (
                            [r for r in left if r.id in started],
                            [r for r in left if r.id not in started],
                        )
                    self._finish(run, outcome)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return [], []

    def _drain(self, progress: Any, started: set[str]) -> None:
        while True:
            try:
                kind, run_id, value = progress.get_nowait()
            except (queue.Empty, OSError, ValueError, EOFError):
                return
            if kind == "started":
                started.add(run_id)
                self.store.mark_run(run_id, RunStatus.running, progress=0.0)
                self._emit(ProgressEvent(run_id, "started", pid=int(value)))
            else:
                now = time.monotonic()
                if now - self._last_db.get(run_id, -1e9) >= DB_PROGRESS_INTERVAL_S:
                    self.store.update_progress(run_id, float(value))
                    self._last_db[run_id] = now
                self._emit(ProgressEvent(run_id, "progress", fraction=float(value)))

    def _finish(self, run: RunRecord, outcome: dict[str, Any]) -> None:
        status = RunStatus(outcome["status"])
        self.store.mark_run(
            run.id,
            status,
            progress=1.0 if status is RunStatus.completed else run.progress,
            wall_time_s=outcome.get("wall_time_s"),
            summary=outcome.get("summary"),
            error=outcome.get("error"),
        )
        self._emit(ProgressEvent(run.id, "finished", fraction=1.0, status=status))

    def _died(self, run: RunRecord) -> None:
        out = Path(run.artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(WORKER_DIED + "\n", encoding="utf-8")
        self.store.mark_run(run.id, RunStatus.failed, error=WORKER_DIED)
        self._emit(ProgressEvent(run.id, "finished", status=RunStatus.failed))

    def _emit(self, event: ProgressEvent) -> None:
        if self._notify is not None:
            self._notify(event)

    def _final_status(self, experiment_id: str) -> ExperimentStatus:
        statuses = [r.status for r in self.store.list_runs(experiment_id)]
        if all(s is RunStatus.completed for s in statuses):
            return ExperimentStatus.completed
        if self._cancelled.is_set():
            return ExperimentStatus.cancelled
        if any(s is RunStatus.completed for s in statuses):
            return ExperimentStatus.partial
        return ExperimentStatus.failed
