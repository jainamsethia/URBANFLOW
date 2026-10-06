"""Gymnasium / PettingZoo environments (plan L, AT-32..AT-34 core)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")

import gymnasium as gym

import urbanflow.rl  # noqa: F401 - registers the gym id
from urbanflow.core.errors import ConfigError
from urbanflow.rl import TrafficSignalEnv, TrafficSignalParallelEnv
from urbanflow.rl.baselines import MaxPressurePolicy, RandomPolicy, TabularQ, evaluate


def test_check_env_and_registration() -> None:
    from gymnasium.utils.env_checker import check_env

    env = TrafficSignalEnv("single_intersection", episode_length=120)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, skip_render_check=True)
    made = gym.make("urbanflow/TrafficSignal-v0", episode_length=60)
    obs, info = made.reset(seed=0)
    assert obs.shape == env.observation_space.shape and info["action_mask"].dtype == np.int8


def test_same_seed_same_episode_and_reseed_changes_demand() -> None:
    env = TrafficSignalEnv("single_intersection", episode_length=300)

    def rollout(seed: int) -> tuple[list[float], float]:
        obs, info = env.reset(seed=seed)
        rewards, done = [], False
        policy = RandomPolicy(seed=7)
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, info))
            rewards.append(r)
            done = term or trunc
        return rewards, info["episode_metrics"]["vehicles.generated"]

    a, b, c = rollout(1), rollout(1), rollout(2)
    assert a == b
    assert a != c


def test_masks_and_overrides() -> None:
    env = TrafficSignalEnv("single_intersection", episode_length=120, decision_interval=1.0)
    _obs, info = env.reset(seed=0)
    current = info["phase"]
    other = 1 - current
    # immediately after reset the green is younger than min-green: only the current phase
    assert info["action_mask"].tolist() == [int(p == current) for p in range(2)]
    _obs, _r, _t, _tr, info = env.step(other)
    assert info["action_overridden"] is True and info["phase"] == current
    for _ in range(6):
        _obs, _r, _t, _tr, info = env.step(current)
    assert info["action_mask"].all()  # min-green passed
    _obs, _r, _t, _tr, info = env.step(other)
    assert info["action_overridden"] is False
    _obs, _r, _t, _tr, info = env.step(other)
    assert info["action_mask"].tolist() == [int(p == other) for p in range(2)]  # transition


def test_bad_configuration() -> None:
    with pytest.raises(ConfigError, match="multiple of dt"):
        TrafficSignalEnv("single_intersection", decision_interval=2.5, dt=1.0)
    with pytest.raises(ConfigError, match="controls one intersection"):
        TrafficSignalEnv("grid_3x3")
    with pytest.raises(ConfigError, match="unknown reward"):
        TrafficSignalEnv("single_intersection", reward="speedy")


def test_parallel_env_api() -> None:
    from pettingzoo.test import parallel_api_test

    env = TrafficSignalParallelEnv("grid_3x3", episode_length=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel_api_test(env, num_cycles=15)
    obs, _ = env.reset(seed=3)
    assert set(obs) == {f"r{r}c{c}" for r in range(3) for c in range(3)}
    assert env.state().shape == (sum(o.size for o in obs.values()),)


def test_max_pressure_policy_beats_random() -> None:
    env = TrafficSignalEnv("single_intersection", episode_length=600, reward="queue")
    seeds = [11, 12]
    rand = evaluate(env, RandomPolicy(0), seeds)
    mp = evaluate(env, MaxPressurePolicy(env.task), seeds)
    mean = lambda rs: float(np.mean([r["waiting_time_mean"] for r in rs]))  # noqa: E731
    assert mean(mp) < mean(rand)


def test_tabular_q_update_and_save_load(tmp_path: object) -> None:
    from pathlib import Path

    q = TabularQ(n_actions=2, alpha=0.5, gamma=0.9, seed=0)
    s, s2 = np.array([0.0, 0.5]), np.array([0.2, 0.9])
    mask = np.array([1, 1], dtype=np.int8)
    q.update(s, 1, reward=2.0, next_obs=s2, next_mask=mask, terminated=False)
    assert q.values(s)[1] == pytest.approx(1.0)  # 0 + 0.5 * (2 + 0.9 * 0 - 0)
    q.update(s, 1, reward=2.0, next_obs=s2, next_mask=mask, terminated=True)
    assert q.values(s)[1] == pytest.approx(1.5)
    assert q.act(s, np.array([1, 0], dtype=np.int8), greedy=True) == 0  # masked
    path = Path(str(tmp_path)) / "q.npz"
    q.save(str(path))
    again = TabularQ.load(str(path))
    assert again.values(s)[1] == pytest.approx(1.5)
