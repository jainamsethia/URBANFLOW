"""Reference policies and a tabular Q-learner (plan L.8), plus an evaluation helper.

Policies are callables ``policy(obs, info) -> action`` for one agent; ``info`` carries the
``action_mask``. Evaluation always uses seeds disjoint from training seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from urbanflow.rl.task import AgentSpec, SignalControlTask
from urbanflow.signals.controllers.max_pressure import max_pressure_choice

__all__ = [
    "FixedCyclePolicy",
    "MaxPressurePolicy",
    "RandomPolicy",
    "TabularQ",
    "evaluate",
    "phase_queue_features",
]

Policy = Callable[[np.ndarray, dict[str, Any]], int]


def phase_queue_features(agent: AgentSpec, obs: np.ndarray, *, scale: float = 2.0) -> np.ndarray:
    """A compact state for tabular learners: the phase one-hot, the min-green flag and, per
    phase, the summed normalised queue of the incoming lanes that phase serves (/ ``scale``,
    clipped to 1)."""
    p = agent.n_phases
    queue = obs[p + 1 + agent.in_lanes.size :]
    position = {lane: i for i, lane in enumerate(agent.in_lanes.tolist())}
    served = np.array(
        [
            sum(queue[position[x]] for x in set(agent.conn_from[agent.phase_conn[k]].tolist()))
            for k in range(p)
        ]
    )
    return np.concatenate([obs[: p + 1], np.minimum(1.0, served / scale)])


class RandomPolicy:
    """Uniform over the valid actions."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, _obs: np.ndarray, info: dict[str, Any]) -> int:
        return int(self.rng.choice(np.flatnonzero(info["action_mask"])))


class FixedCyclePolicy:
    """Cycle through the phases, holding each for ``green`` seconds."""

    def __init__(self, task: SignalControlTask, green: float = 30.0) -> None:
        self.task, self.green = task, green

    def __call__(self, _obs: np.ndarray, info: dict[str, Any]) -> int:
        n = int(info["action_mask"].size)
        phase = int(info["phase"])
        sim = self.task.sim
        agent = next(a for a in self.task.agents if a.n_phases == n)
        view = sim.signals[agent.id]
        if view.stage == "green" and view.green_elapsed >= self.green - 1e-9:
            nxt = (phase + 1) % n
            return nxt if info["action_mask"][nxt] else phase
        return phase


class MaxPressurePolicy:
    """Varaiya's max-pressure choice over the valid phases (same formula as the
    ``max_pressure`` controller)."""

    def __init__(self, task: SignalControlTask, agent: AgentSpec | None = None) -> None:
        self.task = task
        self.agent = agent or task.agents[0]

    def __call__(self, _obs: np.ndarray, info: dict[str, Any]) -> int:
        a = self.agent
        counts = self.task.link_counts()
        phase = int(info["phase"])
        mask = info["action_mask"].astype(bool)
        if mask.sum() == 1:
            return int(np.flatnonzero(mask)[0])
        conn = a.phase_conn & mask[:, None]
        return max_pressure_choice(counts[a.conn_from], counts[a.conn_to], conn, phase)


@dataclass
class TabularQ:
    """Tabular Q-learning over binned observations, with masked epsilon-greedy actions."""

    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    bins: tuple[float, ...] = (0.1, 0.3, 0.6)
    seed: int = 0
    q: dict[bytes, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def key(self, obs: np.ndarray) -> bytes:
        return bytes(np.digitize(obs, self.bins).astype(np.int8).tobytes())

    def values(self, obs: np.ndarray) -> np.ndarray:
        return self.q.setdefault(self.key(obs), np.zeros(self.n_actions))

    def act(self, obs: np.ndarray, mask: np.ndarray, *, greedy: bool = False) -> int:
        valid = np.flatnonzero(mask)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))
        qv = np.where(mask.astype(bool), self.values(obs), -np.inf)
        return int(np.argmax(qv))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
        terminated: bool,
    ) -> None:
        target = reward
        if not terminated:  # truncation still bootstraps
            nxt = np.where(next_mask.astype(bool), self.values(next_obs), -np.inf)
            target += self.gamma * float(nxt.max())
        q = self.values(obs)
        q[action] += self.alpha * (target - q[action])

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        return self.act(obs, info["action_mask"], greedy=True)

    def save(self, path: str) -> None:
        keys = list(self.q)
        np.savez_compressed(
            path,
            keys=np.array([np.frombuffer(k, dtype=np.int8) for k in keys]),
            values=np.array([self.q[k] for k in keys]),
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> TabularQ:
        with np.load(path, allow_pickle=False) as data:
            values = data["values"]
            q = {
                k.astype(np.int8).tobytes(): v.copy()
                for k, v in zip(data["keys"], values, strict=True)
            }
        agent = cls(n_actions=int(values.shape[1]), **kwargs)
        agent.q = q
        return agent


def evaluate(env: Any, policy: Policy, seeds: Iterable[int]) -> list[dict[str, float]]:
    """Run ``policy`` greedily for one episode per seed on a single-agent env; returns
    the episode metric summaries (travel time, delay, waiting time ...)."""
    out = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs, info))
            total += reward
            done = terminated or truncated
        summary = dict(info.get("episode_metrics", {}))
        summary["episode_return"] = total
        out.append(summary)
    return out
