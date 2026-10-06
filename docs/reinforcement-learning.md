# Reinforcement learning

UrbanFlow ships first-party environments for traffic-signal control (extra `rl`:
gymnasium, pettingzoo).

```python
import gymnasium as gym
import urbanflow.rl  # registers urbanflow/TrafficSignal-v0

env = gym.make(
    "urbanflow/TrafficSignal-v0", scenario="single_intersection", episode_length=900, reward="queue"
)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(
    env.action_space.sample(mask=info["action_mask"])
)
```

- **`TrafficSignalEnv`** (Gymnasium) controls one intersection;
  **`TrafficSignalParallelEnv`** (PettingZoo Parallel API) makes every signalised
  intersection an agent (`state()` is the concatenated observation).
- **Actions**: `Discrete(n_phases)`, the phase wanted next, every `decision_interval`
  (5 s). The engine inserts yellow and all-red and enforces min-green; phases that cannot
  be honoured now are masked (`info["action_mask"]`, `env.action_masks()`); a masked
  choice keeps the current phase and sets `info["action_overridden"]`.
- **Observations** (in [0, 1]): phase one-hot, min-green-passed flag, per incoming lane
  density `min(1, n/c)` and stop-line queue `min(1, Q/c)` with `c = max(1, L/7.5 m)`.
- **Rewards**: `diff_waiting_time` (drop in summed waiting time / 100), `queue` (minus
  halting vehicles), `pressure` (PressLight); or a weighted dict.
- **Episodes**: `terminated` when demand is exhausted and the network is empty,
  `truncated` at `episode_length`; `reset(seed=...)` re-seeds all demand, so a seed is a
  reproducible episode. `warmup` runs the scenario's own controllers first.
- **Baselines** (`urbanflow.rl.baselines`): `RandomPolicy`, `FixedCyclePolicy`,
  `MaxPressurePolicy`, `TabularQ` (masked epsilon-greedy, npz save/load) and `evaluate`.

Examples: `examples/reinforcement_learning/q_learning_single.py` (tabular Q on one
junction) and `independent_q_grid.py` (nine learners on the 3x3 grid). Evaluate on seeds
disjoint from the training seeds, and compare policies on the same seeds.
