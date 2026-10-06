# Reinforcement learning

Traffic-signal control as Gymnasium (single intersection) and PettingZoo (all
intersections) environments.

- `q_learning_single.py`: tabular Q-learning on an intersection with 3:1 asymmetric
  demand, evaluated on held-out seeds against random, fixed-cycle and max-pressure
  policies (all on identical demand).
- `independent_q_grid.py`: nine independent tabular Q-learners (one per intersection)
  on the 3x3 grid through the PettingZoo environment, against fixed-time and max-pressure.

```bash
uv run python examples/reinforcement_learning/q_learning_single.py
```

The environments: `urbanflow.rl.TrafficSignalEnv` (`gymnasium.make("urbanflow/TrafficSignal-v0")`)
and `urbanflow.rl.TrafficSignalParallelEnv` (PettingZoo Parallel API). Observations are
phase one-hot, min-green flag, per-lane density and queue; actions choose the next phase
(masked during min-green and transitions; the engine enforces yellow and all-red).
