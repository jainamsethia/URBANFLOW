# Benchmarks

Speed baselines for `urbanflow benchmark` (see [docs/benchmarking.md](../docs/benchmarking.md)).

- `baselines/<machine key>.json`: the quick suites measured on one machine type, used by
  `pytest -m performance` (fails on a slowdown of more than 30 %). `urbanflow benchmark
  --list` prints this machine's key, for example `windows-arm64-py313`.
- `results/`: your own runs (ignored by git).

To (re)create the baseline for this machine, on mains power with nothing else running:

```bash
urbanflow benchmark --list
urbanflow benchmark grid_scaling overhead --quick --repeat 3 -o benchmarks/baselines/<machine key>.json
```
