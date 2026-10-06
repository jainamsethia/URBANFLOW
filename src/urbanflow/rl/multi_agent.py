"""PettingZoo Parallel environment: every signalised intersection is an agent (plan L.1)."""

from __future__ import annotations

import functools
from typing import Any, ClassVar

import numpy as np

from urbanflow.core.errors import require
from urbanflow.rl.task import SignalControlTask
from urbanflow.scenario.scenario import Scenario

require("pettingzoo", "rl")
from gymnasium import spaces  # noqa: E402
from gymnasium.utils import seeding  # noqa: E402
from pettingzoo import ParallelEnv  # noqa: E402

__all__ = ["TrafficSignalParallelEnv", "parallel_env"]


class TrafficSignalParallelEnv(ParallelEnv):  # type: ignore[misc]
    """Joint control of several intersections (independent or cooperative learners)."""

    metadata: ClassVar[dict[str, Any]] = {"name": "urbanflow_signals_v0", "render_modes": ["ansi"]}

    def __init__(
        self,
        scenario: Scenario | str = "grid_3x3",
        intersections: list[str] | None = None,
        *,
        render_mode: str | None = None,
        **task_kwargs: Any,
    ) -> None:
        self.task = SignalControlTask(scenario, intersections, **task_kwargs)
        self.possible_agents = list(self.task.ids)
        self.agents: list[str] = []
        self._specs = {a.id: a for a in self.task.agents}
        self.render_mode = render_mode
        self._np_random, _ = seeding.np_random(None)

    @functools.cache  # noqa: B019 - one space object per agent
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (self._specs[agent].obs_size,), np.float32)

    @functools.cache  # noqa: B019
    def action_space(self, agent: str) -> spaces.Space[Any]:
        return spaces.Discrete(self._specs[agent].n_phases)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        del options  # no reset options are defined (accepted and ignored, per the APIs)
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)
        obs, infos = self.task.reset(int(self._np_random.integers(2**31 - 1)))
        self.agents = list(self.possible_agents)
        return obs, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        obs, rewards, terminated, truncated, infos = self.task.step(actions)
        terms = dict.fromkeys(self.agents, terminated)
        truncs = dict.fromkeys(self.agents, truncated)
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def state(self) -> np.ndarray:
        obs = self.task._observe()
        return np.concatenate([obs[a] for a in self.possible_agents])

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        sim = self.task.sim
        return " ".join(f"{a}:{sim.signals[a].phase_id}" for a in self.possible_agents)

    def close(self) -> None:
        self.task.sim.close()


def parallel_env(**kwargs: Any) -> TrafficSignalParallelEnv:
    """PettingZoo-style factory."""
    return TrafficSignalParallelEnv(**kwargs)
