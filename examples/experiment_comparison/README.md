# Experiment comparison

- `run_comparison.py`: 3 controllers x 5 seeds on the 4x4 grid, 95% confidence intervals and paired differences, saved as JSON (the same as `urbanflow compare`).
- `controllers.json` + `run_experiment.py`: an experiment file (5 variants, including a heavier-demand one, x 5 seeds) run by the experiment runner in parallel worker processes, stored in a database (resumable, crash-isolated) and compared against the fixed-time baseline. From the CLI: `urbanflow experiment run examples/experiment_comparison/controllers.json`, then `urbanflow experiment compare grid-controllers`.
