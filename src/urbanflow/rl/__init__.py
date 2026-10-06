"""Reinforcement learning: Gymnasium / PettingZoo traffic-signal environments (plan L).

Needs the ``rl`` extra (``gymnasium``, ``pettingzoo``). Importing this package registers
the Gymnasium id ``urbanflow/TrafficSignal-v0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from urbanflow.rl.task import AgentSpec, SignalControlTask

__all__ = [
    "AgentSpec",
    "SignalControlTask",
    "TrafficSignalEnv",
    "TrafficSignalParallelEnv",
    "parallel_env",
]

if TYPE_CHECKING:
    from urbanflow.rl.env import TrafficSignalEnv as TrafficSignalEnv
    from urbanflow.rl.multi_agent import TrafficSignalParallelEnv as TrafficSignalParallelEnv
    from urbanflow.rl.multi_agent import parallel_env as parallel_env


def __getattr__(name: str) -> Any:
    if name == "TrafficSignalEnv":
        from urbanflow.rl.env import TrafficSignalEnv

        return TrafficSignalEnv
    if name in ("TrafficSignalParallelEnv", "parallel_env"):
        from urbanflow.rl import multi_agent

        return getattr(multi_agent, name)
    raise AttributeError(name)


def _register() -> None:
    try:
        import gymnasium
    except ImportError:  # the env itself raises MissingDependencyError with a hint
        return
    if "urbanflow/TrafficSignal-v0" not in gymnasium.registry:
        gymnasium.register(
            "urbanflow/TrafficSignal-v0", entry_point="urbanflow.rl.env:TrafficSignalEnv"
        )


_register()
