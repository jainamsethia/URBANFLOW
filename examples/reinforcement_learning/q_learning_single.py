"""Tabular Q-learning for one signalised intersection, compared with baselines.

The north-south approaches carry 3x the east-west demand, so a fixed 50/50 split wastes
green time. The agent sees a compact state (current phase, whether min-green has passed,
and the binned queues served by each phase) and learns when to switch. Evaluation uses
seeds disjoint from training; every policy is evaluated on the same seeds (same demand).

Run:  uv run python examples/reinforcement_learning/q_learning_single.py
Quick (CI): set URBANFLOW_EXAMPLE_QUICK=1 before the same command.
"""

from __future__ import annotations

import os
import statistics

import numpy as np

from urbanflow import generate
from urbanflow.rl import TrafficSignalEnv
from urbanflow.rl.baselines import (
    FixedCyclePolicy,
    MaxPressurePolicy,
    RandomPolicy,
    TabularQ,
    evaluate,
)

QUICK = os.environ.get("URBANFLOW_EXAMPLE_QUICK") == "1"
EPISODES = 4 if QUICK else 60
EPISODE_S = 300 if QUICK else 900
EVAL_SEEDS = range(10_000, 10_002 if QUICK else 10_005)


def make_env() -> TrafficSignalEnv:
    scenario = generate(
        "single_intersection",
        approach_rates={"N": 900, "S": 900, "E": 300, "W": 300},
        duration=EPISODE_S,
    )
    return TrafficSignalEnv(scenario, episode_length=EPISODE_S, reward="queue")


def compact(env: TrafficSignalEnv, obs: np.ndarray) -> np.ndarray:
    """(phase one-hot, min-green ok, queue served by each phase) - small enough for a table."""
    a = env.agent
    p = a.n_phases
    queue = obs[p + 1 + a.in_lanes.size :]
    from_index = {lane: i for i, lane in enumerate(a.in_lanes.tolist())}
    served = [
        sum(queue[from_index[x]] for x in set(a.conn_from[a.phase_conn[k]].tolist()))
        for k in range(p)
    ]
    return np.concatenate([obs[: p + 1], np.minimum(1.0, np.array(served) / 2.0)])


def train(env: TrafficSignalEnv) -> TabularQ:
    agent = TabularQ(env.agent.n_phases, alpha=0.2, gamma=0.9, epsilon_decay=0.95, seed=0)
    for episode in range(EPISODES):
        obs, info = env.reset(seed=episode)
        state = compact(env, obs)
        done, total = False, 0.0
        while not done:
            action = agent.act(state, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
            nxt = compact(env, obs)
            agent.update(state, action, reward, nxt, info["action_mask"], terminated)
            state, total = nxt, total + reward
            done = terminated or truncated
        agent.end_episode()
        if episode % 10 == 0 or episode == EPISODES - 1:
            print(f"episode {episode:3d}  return {total:9.1f}  epsilon {agent.epsilon:.2f}")
    return agent


def main() -> None:
    env = make_env()
    q = train(env)
    policies = {
        "random": RandomPolicy(seed=1),
        "fixed_cycle_30s": FixedCyclePolicy(env.task, green=30),
        "max_pressure": MaxPressurePolicy(env.task),
        "q_learning": lambda obs, info: q.act(compact(env, obs), info["action_mask"], greedy=True),
    }
    print(f"\nEvaluation on {len(EVAL_SEEDS)} held-out seeds ({EPISODE_S} s each):")
    print(f"{'policy':16s} {'mean waiting (s)':>18s} {'mean travel (s)':>16s}")
    for name, policy in policies.items():
        runs = evaluate(env, policy, EVAL_SEEDS)
        wait = statistics.mean(r.get("waiting_time_mean", float("nan")) for r in runs)
        travel = statistics.mean(r.get("travel_time.mean", float("nan")) for r in runs)
        print(f"{name:16s} {wait:18.1f} {travel:16.1f}")


if __name__ == "__main__":
    main()
