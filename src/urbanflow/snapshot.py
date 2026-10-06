"""Simulation snapshots: ``sim.snapshot()`` / ``sim.restore(snapshot)`` (plan AD.1 8.9, F.5).

A :class:`Snapshot` wraps the engine's :class:`~urbanflow.engine.state.EngineState` (deep
copies of every runtime array, controller, spawner and RNG stream) plus, in memory only, a
deep copy of the run-summary accumulators. :meth:`Snapshot.save` writes the engine state
alone (a zip of ``state.npz`` and ``state.json``; nothing is pickled);
:meth:`Snapshot.load` reads it back, and restoring a loaded snapshot restarts the summary
accumulators (a WARNING is logged).

ponytail: the summary accumulators are not written to files, so a snapshot loaded from
disk restarts travel-time statistics, event counts and the arrived-id list; the upgrade
path is a ``state_dict`` per metrics collector, saved beside the engine state.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from urbanflow.core.config import SimulationConfig
from urbanflow.core.constants import SHORT_HASH_LENGTH
from urbanflow.core.errors import SimulationError
from urbanflow.engine.state import EngineState, load_state, save_state

__all__ = ["Snapshot"]


@dataclass(frozen=True, slots=True, eq=False)
class Snapshot:
    """The state of a simulation at one step (``Simulation.snapshot()``)."""

    state: EngineState
    """Deep copy of the engine's runtime state."""
    metrics: Mapping[str, Any] | None
    """Deep copy of the run-summary accumulators (in memory only); None when loaded from a
    file."""
    scenario_hash: str
    """``Scenario.content_hash`` of the simulation it was taken from."""
    config: SimulationConfig
    """Its resolved config; ``restore`` requires the same."""
    time: float
    """Simulated time when it was taken, s."""

    @property
    def step_count(self) -> int:
        """Steps completed when it was taken."""
        return self.state.step_count

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the engine state to ``path`` (atomically); the summary accumulators are not
        saved. Returns the path."""
        return save_state(path, self.state)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Snapshot:
        """Read a snapshot written by :meth:`save` (``NotFoundError`` if missing,
        ``SimulationError`` if it is not a valid state file)."""
        state = load_state(path)
        try:
            config = SimulationConfig.model_validate(state.config)
        except ValidationError as exc:
            raise SimulationError(f"{path}: the saved config is invalid: {exc}") from None
        return cls(state, None, state.scenario_hash, config, state.step_count * config.dt)

    def __repr__(self) -> str:
        return f"Snapshot(t={self.time:g}s, scenario={self.scenario_hash[:SHORT_HASH_LENGTH]})"
