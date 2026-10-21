"""Experiment and run records in SQLite (default) or PostgreSQL (plan Q.4).

Only the process that runs an experiment writes; workers send their results back to it.
SQLite runs in WAL mode with a busy timeout, so a CLI listing experiments while another
process runs one does not fail. Records come back as plain dataclasses (no ORM objects).

ponytail: ``create_all`` plus a ``schema_version`` row; migrations (alembic) arrive with the
first schema change.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from sqlalchemy import (
    JSON,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from urbanflow.core.errors import NotFoundError
from urbanflow.experiments.spec import ExperimentSpec, ExperimentStatus, RunSpec, RunStatus

__all__ = ["ExperimentRecord", "ExperimentStore", "RunRecord", "make_engine"]

SCHEMA_VERSION: Final = "1"


class _Base(DeclarativeBase):
    pass


class _Experiment(_Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))
    urbanflow_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    runs: Mapped[list[_Run]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", passive_deletes=True
    )


class _Run(_Base):
    __tablename__ = "runs"
    __table_args__ = (UniqueConstraint("experiment_id", "variant", "seed"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    variant: Mapped[str] = mapped_column(String(64))
    seed: Mapped[int] = mapped_column(Integer)
    position: Mapped[int] = mapped_column(Integer)
    """Order of the run in the experiment (seed-major)."""
    status: Mapped[str] = mapped_column(String(16), index=True)
    content_key: Mapped[str] = mapped_column(String(64), index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    wall_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifacts_dir: Mapped[str] = mapped_column(Text)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    experiment: Mapped[_Experiment] = relationship(back_populates="runs")


class _Meta(_Base):
    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


def make_engine(url: str) -> Engine:
    """A SQLAlchemy engine for ``url``; SQLite gets WAL, foreign keys and a busy timeout."""
    if url.startswith("sqlite"):
        path = url.split(":///", 1)[1] if ":///" in url else ""
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _pragmas(conn: Any, _record: Any) -> None:
            cur = conn.cursor()
            for pragma in (
                "journal_mode=WAL",
                "synchronous=NORMAL",
                "foreign_keys=ON",
                "busy_timeout=5000",
            ):
                cur.execute(f"PRAGMA {pragma}")
            cur.close()

        return engine
    return create_engine(url, pool_pre_ping=True)


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    experiment_id: str
    variant: str
    seed: int
    status: RunStatus
    content_key: str
    progress: float
    started_at: datetime | None
    finished_at: datetime | None
    wall_time_s: float | None
    artifacts_dir: str
    summary: dict[str, Any] | None
    error: str | None
    spec: RunSpec


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    id: str
    name: str
    description: str
    spec: ExperimentSpec
    status: ExperimentStatus
    urbanflow_version: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    counts: dict[str, int]
    """Runs per status."""


def _now() -> datetime:
    return datetime.now(UTC)


def _run_record(row: _Run) -> RunRecord:
    return RunRecord(
        id=row.id,
        experiment_id=row.experiment_id,
        variant=row.variant,
        seed=row.seed,
        status=RunStatus(row.status),
        content_key=row.content_key,
        progress=row.progress,
        started_at=row.started_at,
        finished_at=row.finished_at,
        wall_time_s=row.wall_time_s,
        artifacts_dir=row.artifacts_dir,
        summary=row.summary_json,
        error=row.error,
        spec=RunSpec.model_validate(row.spec_json),
    )


class ExperimentStore:
    """Repository of experiments and their runs."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        _Base.metadata.create_all(engine)
        with Session(engine) as s, s.begin():
            if s.get(_Meta, "schema_version") is None:
                s.add(_Meta(key="schema_version", value=SCHEMA_VERSION))

    @classmethod
    def open(cls, url: str) -> ExperimentStore:
        return cls(make_engine(url))

    def close(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------ experiments
    def create_experiment(
        self,
        experiment_id: str,
        spec: ExperimentSpec,
        runs: Sequence[RunSpec],
        *,
        status: ExperimentStatus = ExperimentStatus.queued,
    ) -> ExperimentRecord:
        from urbanflow._version import __version__

        with Session(self.engine) as s, s.begin():
            s.add(
                _Experiment(
                    id=experiment_id,
                    name=spec.name,
                    description=spec.description,
                    spec_json=spec.model_dump(mode="json"),
                    status=status,
                    urbanflow_version=__version__,
                    created_at=_now(),
                )
            )
            for k, run in enumerate(runs):
                s.add(
                    _Run(
                        id=run.run_id,
                        experiment_id=experiment_id,
                        variant=run.variant,
                        seed=run.seed,
                        position=k,
                        status=RunStatus.pending,
                        content_key=run.content_key,
                        artifacts_dir=run.output_dir,
                        spec_json=run.model_dump(mode="json"),
                    )
                )
        return self.get_experiment(experiment_id)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        with Session(self.engine) as s:
            row = s.get(_Experiment, experiment_id)
            if row is None:
                known = [r for (r,) in s.execute(select(_Experiment.id))]
                hint = f" (known: {', '.join(known[:10])})" if known else ""
                raise NotFoundError(f'unknown experiment "{experiment_id}"{hint}')
            return self._record(s, row)

    def find_experiment(self, prefix: str) -> ExperimentRecord:
        """By id or a unique id prefix, or by name (the most recent)."""
        with Session(self.engine) as s:
            ids = [
                r
                for (r,) in s.execute(
                    select(_Experiment.id).where(_Experiment.id.startswith(prefix))
                )
            ]
            if len(ids) == 1:
                return self._record(s, s.get(_Experiment, ids[0]))  # type: ignore[arg-type]
            named = s.scalars(
                select(_Experiment)
                .where(_Experiment.name == prefix)
                .order_by(_Experiment.created_at.desc())
            ).first()
            if named is not None:
                return self._record(s, named)
        if len(ids) > 1:
            raise NotFoundError(f'"{prefix}" matches several experiments: {", ".join(ids)}')
        raise NotFoundError(f'unknown experiment "{prefix}"')

    def list_experiments(
        self, *, limit: int = 50, offset: int = 0, status: str | None = None
    ) -> list[ExperimentRecord]:
        with Session(self.engine) as s:
            q = select(_Experiment).order_by(_Experiment.created_at.desc())
            if status is not None:
                q = q.where(_Experiment.status == status)
            return [self._record(s, r) for r in s.scalars(q.limit(limit).offset(offset))]

    def count_experiments(self, *, status: str | None = None) -> int:
        with Session(self.engine) as s:
            q = select(func.count()).select_from(_Experiment)
            if status is not None:
                q = q.where(_Experiment.status == status)
            return int(s.execute(q).scalar_one())

    def mark_interrupted(self) -> list[str]:
        """Experiments left running or queued by a process that is gone (server start):
        they and their running runs become ``interrupted``; their ids."""
        active = (ExperimentStatus.running, ExperimentStatus.queued)
        with Session(self.engine) as s, s.begin():
            rows = list(s.scalars(select(_Experiment).where(_Experiment.status.in_(active))))
            for row in rows:
                row.status = ExperimentStatus.interrupted
                for run in row.runs:
                    if run.status == RunStatus.running:
                        run.status = RunStatus.interrupted
            return [r.id for r in rows]

    def set_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> None:
        with Session(self.engine) as s, s.begin():
            row = s.get(_Experiment, experiment_id)
            if row is None:
                raise NotFoundError(f'unknown experiment "{experiment_id}"')
            row.status = status
            if status is ExperimentStatus.running and row.started_at is None:
                row.started_at = _now()
            if status not in (ExperimentStatus.running, ExperimentStatus.queued):
                row.finished_at = _now()

    def delete_experiment(self, experiment_id: str, *, delete_artifacts: bool = False) -> None:
        dirs = [r.artifacts_dir for r in self.list_runs(experiment_id)]
        with Session(self.engine) as s, s.begin():
            row = s.get(_Experiment, experiment_id)
            if row is None:
                raise NotFoundError(f'unknown experiment "{experiment_id}"')
            s.delete(row)
        if delete_artifacts:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def _record(s: Session, row: _Experiment) -> ExperimentRecord:
        counts: dict[str, int] = {}
        for (status,) in s.execute(select(_Run.status).where(_Run.experiment_id == row.id)):
            counts[status] = counts.get(status, 0) + 1
        return ExperimentRecord(
            id=row.id,
            name=row.name,
            description=row.description,
            spec=ExperimentSpec.model_validate(row.spec_json),
            status=ExperimentStatus(row.status),
            urbanflow_version=row.urbanflow_version,
            created_at=row.created_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
            counts=counts,
        )

    # ------------------------------------------------------------------ runs
    def list_runs(
        self, experiment_id: str, *, status: Iterable[RunStatus] | None = None
    ) -> list[RunRecord]:
        with Session(self.engine) as s:
            q = select(_Run).where(_Run.experiment_id == experiment_id).order_by(_Run.position)
            if status is not None:
                q = q.where(_Run.status.in_([str(x) for x in status]))
            return [_run_record(r) for r in s.scalars(q)]

    def get_run(self, run_id: str) -> RunRecord:
        with Session(self.engine) as s:
            row = s.get(_Run, run_id)
            if row is None:
                raise NotFoundError(f'unknown run "{run_id}"')
            return _run_record(row)

    def mark_run(self, run_id: str, status: RunStatus, **fields: Any) -> None:
        """Set a run's status and any of ``progress``, ``wall_time_s``, ``summary``,
        ``error``; ``started_at``/``finished_at`` follow the status."""
        with Session(self.engine) as s, s.begin():
            row = s.get(_Run, run_id)
            if row is None:
                raise NotFoundError(f'unknown run "{run_id}"')
            row.status = status
            if status is RunStatus.running:
                row.started_at = _now()
            elif status is not RunStatus.pending:
                row.finished_at = _now()
            if "summary" in fields:
                row.summary_json = fields.pop("summary")
            for key, value in fields.items():
                setattr(row, key, value)

    def update_progress(self, run_id: str, progress: float) -> None:
        with Session(self.engine) as s, s.begin():
            row = s.get(_Run, run_id)
            if row is not None:
                row.progress = progress
