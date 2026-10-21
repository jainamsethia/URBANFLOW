"""The experiment store: records, runs, lookups, cascade (plan Q.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from urbanflow.core.errors import NotFoundError
from urbanflow.experiments.spec import ExperimentStatus, RunStatus, expand, parse_experiment
from urbanflow.experiments.store import ExperimentStore


@pytest.fixture
def store(tmp_path: Path) -> ExperimentStore:
    return ExperimentStore.open(f"sqlite:///{(tmp_path / 'db' / 'uf.db').as_posix()}")


def _create(store: ExperimentStore, tmp_path: Path, eid: str = "abc123") -> None:
    spec = parse_experiment(
        {
            "name": "ctrl",
            "scenario": {"bundled": "single_intersection"},
            "variants": [{"name": "a"}, {"name": "b"}],
            "seeds": [0, 1],
        }
    )
    store.create_experiment(eid, spec, expand(spec, tmp_path, eid, tmp_path / "runs"))


def test_create_list_and_mark(store: ExperimentStore, tmp_path: Path) -> None:
    _create(store, tmp_path)
    rec = store.get_experiment("abc123")
    assert rec.status is ExperimentStatus.queued and rec.counts == {"pending": 4}
    runs = store.list_runs("abc123")
    assert [(r.variant, r.seed) for r in runs] == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]
    store.mark_run(runs[0].id, RunStatus.running)
    store.mark_run(runs[0].id, RunStatus.completed, wall_time_s=1.5, summary={"x": 1.0})
    done = store.get_run(runs[0].id)
    assert done.status is RunStatus.completed and done.summary == {"x": 1.0}
    assert done.started_at is not None and done.finished_at is not None
    assert [r.id for r in store.list_runs("abc123", status=[RunStatus.completed])] == [done.id]
    store.set_experiment_status("abc123", ExperimentStatus.running)
    assert store.get_experiment("abc123").started_at is not None


def test_lookup_by_prefix_or_name(store: ExperimentStore, tmp_path: Path) -> None:
    _create(store, tmp_path, "abc123")
    _create(store, tmp_path, "abd456")
    assert store.find_experiment("abc").id == "abc123"
    assert store.find_experiment("ctrl").name == "ctrl"
    with pytest.raises(NotFoundError, match="several"):
        store.find_experiment("ab")
    with pytest.raises(NotFoundError):
        store.get_experiment("zzz")


def test_delete_cascades(store: ExperimentStore, tmp_path: Path) -> None:
    _create(store, tmp_path)
    store.delete_experiment("abc123")
    assert store.list_runs("abc123") == [] and store.list_experiments() == []
