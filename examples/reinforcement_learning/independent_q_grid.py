"""Multi-agent RL: one tabular Q-learner per intersection on the 3x3 grid (PettingZoo).

Each of the 9 signals learns independently from its own queue reward and compact state.
Evaluation runs on held-out seeds against fixed-time and max-pressure (same demand).

    uv run python examples/reinforcement_learning/independent_q_grid.py
"""

from __future__ import annotations

import os
import statistics

import numpy as np

from urbanflow.rl import TrafficSignalParallelEnv
from urbanflow.rl.baselines import MaxPressurePolicy, TabularQ, phase_queue_features

QUICK = os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1"
EPISODES = 3 if QUICK else 40
EPISODE_S = 240 if QUICK else 900
EVAL_SEEDS = range(20_000, 20_001 if QUICK else 20_003)


def episode(env: TrafficSignalParallelEnv, act, learners=None, seed=0):
    """Run one episode; ``act(agent, state, info) -> action``; updates ``learners`` if given."""
    specs = {a.id: a for a in env.task.agents}
    obs, infos = env.reset(seed=seed)
    states = {a: phase_queue_features(specs[a], obs[a]) for a in env.agents}
    while env.agents:
        actions = {a: act(a, states[a], infos[a]) for a in env.agents}
        obs, rewards, terms, _truncs, infos = env.step(actions)
        nxt = {a: phase_queue_features(specs[a], obs[a]) for a in obs}
        if learners is not None:
            for a, q in learners.items():
                q.update(
                    states[a], actions[a], rewards[a], nxt[a], infos[a]["action_mask"], terms[a]
                )
        states = nxt
    return env.task.summary()


def main() -> None:
    env = TrafficSignalParallelEnv("grid_3x3", episode_length=EPISODE_S, reward="queue")
    learners = {
        a.id: TabularQ(a.n_phases, alpha=0.2, gamma=0.9, epsilon_decay=0.93, seed=i)
        for i, a in enumerate(env.task.agents)
    }

    def explore(agent: str, state: np.ndarray, info: dict) -> int:  # type: ignore[type-arg]
        return learners[agent].act(state, info["action_mask"])

    for ep in range(EPISODES):
        s = episode(env, explore, learners, seed=ep)
        for q in learners.values():
            q.end_episode()
        if ep % 10 == 0 or ep == EPISODES - 1:
            print(
                f"episode {ep:3d}  mean waiting {s.get('waiting_time_mean', float('nan')):6.1f} s"
            )

    specs = {a.id: a for a in env.task.agents}
    pressure = {a: MaxPressurePolicy(env.task, specs[a]) for a in specs}
    policies = {
        "fixed_time (cycle)": lambda a, _s, info: (  # keep the scenario-like 30 s cycle
            (info["phase"] + 1) % info["action_mask"].size
            if info["action_mask"].all() and env.task.sim.signals[a].green_elapsed >= 30
            else info["phase"]
        ),
        "max_pressure": lambda a, s, info: pressure[a](s, info),
        "independent_q": lambda a, s, info: learners[a].act(s, info["action_mask"], greedy=True),
    }
    print(f"\nEvaluation on {len(EVAL_SEEDS)} held-out seed(s), {EPISODE_S} s each:")
    for name, policy in policies.items():
        runs = [episode(env, policy, seed=seed) for seed in EVAL_SEEDS]
        wait = statistics.mean(r.get("waiting_time_mean", float("nan")) for r in runs)
        travel = statistics.mean(r.get("travel_time.mean", float("nan")) for r in runs)
        print(f"  {name:20s} mean waiting {wait:6.1f} s   mean travel {travel:6.1f} s")


if __name__ == "__main__":
    main()
