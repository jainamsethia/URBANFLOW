"""Gymnasium environment: one signalised intersection (plan L.1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from urbanflow.core.errors import ConfigError, require
from urbanflow.rl.task import SignalControlTask
from urbanflow.scenario.scenario import Scenario

require("gymnasium", "rl")
import gymnasium as gym  # noqa: E402 - after the friendly dependency check
from gymnasium import spaces  # noqa: E402

__all__ = ["TrafficSignalEnv"]


class TrafficSignalEnv(gym.Env[np.ndarray, int]):
    """Control one intersection's phase every ``decision_interval`` seconds.

    ``TrafficSignalEnv("single_intersection")`` controls the only signalised intersection;
    pass ``intersection=`` when there are several. Other keyword arguments go to
    :class:`~urbanflow.rl.task.SignalControlTask` (``decision_interval``, ``reward``,
    ``episode_length``, ``warmup``, and simulation overrides such as ``dt``).
    """

    metadata = {"render_modes": ["ansi"]}  # noqa: RUF012 - gymnasium API

    def __init__(
        self,
        scenario: Scenario | str = "single_intersection",
        intersection: str | None = None,
        *,
        render_mode: str | None = None,
        **task_kwargs: Any,
    ) -> None:
        self.task = SignalControlTask(
            scenario, None if intersection is None else [intersection], **task_kwargs
        )
        if len(self.task.agents) != 1:
            raise ConfigError(
                "TrafficSignalEnv controls one intersection; choose one of "
                f"{', '.join(self.task.ids)} (or use TrafficSignalParallelEnv)"
            )
        self.agent = self.task.agents[0]
        self.observation_space = spaces.Box(0.0, 1.0, (self.agent.obs_size,), np.float32)
        self.action_space = spaces.Discrete(self.agent.n_phases)
        self.render_mode = render_mode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options  # no reset options are defined (accepted and ignored, per the APIs)
        episode_seed = int(self.np_random.integers(2**31 - 1))
        obs, infos = self.task.reset(episode_seed)
        return obs[self.agent.id], infos[self.agent.id]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, rewards, terminated, truncated, infos = self.task.step({self.agent.id: int(action)})
        info = infos[self.agent.id]
        if terminated or truncated:
            info["episode_metrics"] = self.task.summary()
        return obs[self.agent.id], rewards[self.agent.id], terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Valid actions now (sb3-contrib MaskablePPO convention)."""
        return self.task.action_masks()[self.agent.id]

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        sim = self.task.sim
        view = sim.signals[self.agent.id]
        return (
            f"t={sim.time:7.1f}s {self.agent.id}: phase {view.phase_id} ({view.stage}) "
            f"{view.state_string}  running={len(sim.vehicles)}"
        )
