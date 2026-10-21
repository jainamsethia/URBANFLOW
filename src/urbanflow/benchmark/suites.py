"""Benchmark workloads (plan S.2): data only, so a suite is a list of cases.

* ``grid_scaling``: square grids of 1, 4, 16, 64 and 144 signalised intersections with
  3 lanes per direction; total inflow 360 veh/h per intersection, split evenly over the
  boundary entries (90 x n veh/h per entry of an n x n grid).
* ``vehicle_sweep``: the 8 x 8 grid at 0.25, 0.5, 1, 2 and 4 times that demand.
* ``overhead``: the 4 x 4 grid as is, without lane changes, recording a replay, with the
  debug invariant checks on, and with (almost) no traffic (the fixed cost of a step).

Every case simulates ``warmup`` seconds untimed, then ``timed`` seconds timed (quick mode:
60 s and 300 s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = ["SUITES", "BenchmarkCase", "scenario_case"]

FULL_WARMUP: Final = 300.0
FULL_TIMED: Final = 1500.0
QUICK_WARMUP: Final = 60.0
QUICK_TIMED: Final = 300.0
INFLOW_PER_INTERSECTION: Final = 360.0
"""veh/h of demand per signalised intersection (split over the grid's boundary entries)."""


@dataclass(frozen=True)
class BenchmarkCase:
    """One workload: a generated scenario (or a scenario file) plus config overrides."""

    suite: str
    name: str
    generator: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    """A scenario file instead of a generator."""
    config: dict[str, Any] = field(default_factory=dict)
    """Simulation config overrides (``lane_changing``, ``debug_checks``, ...)."""
    record: bool = False
    """Record a replay (to a temporary file) while timing."""
    seed: int = 0
    warmup: float = FULL_WARMUP
    timed: float = FULL_TIMED

    def durations(self, quick: bool) -> tuple[float, float]:
        """``(warmup, timed)`` simulated seconds."""
        if quick:
            return min(self.warmup, QUICK_WARMUP), min(self.timed, QUICK_TIMED)
        return self.warmup, self.timed


def _grid(suite: str, name: str, n: int, demand: float = 1.0, **extra: Any) -> BenchmarkCase:
    rate = INFLOW_PER_INTERSECTION * n * n / (4 * n) * demand
    params = {"rows": n, "cols": n, "lanes": 3, "spacing": 300.0, "entry_rate": rate}
    return BenchmarkCase(suite, name, generator="grid", params=params, **extra)


SUITES: Final[dict[str, tuple[BenchmarkCase, ...]]] = {
    "grid_scaling": tuple(_grid("grid_scaling", f"g{n}", n) for n in (1, 2, 4, 8, 12)),
    "vehicle_sweep": tuple(
        _grid("vehicle_sweep", f"d{f:g}", 8, f) for f in (0.25, 0.5, 1.0, 2.0, 4.0)
    ),
    "overhead": (
        _grid("overhead", "baseline", 4),
        _grid("overhead", "no_lane_change", 4, config={"lane_changing": False}),
        _grid("overhead", "recording", 4, record=True),
        _grid("overhead", "debug_checks", 4, config={"debug_checks": True}),
        _grid("overhead", "empty", 4, demand=1e-6),
    ),
}


def scenario_case(path: str | Path) -> BenchmarkCase:
    """A one-case suite for a scenario file: no warm-up, then 1500 s timed (quick: 300 s)."""
    return BenchmarkCase("scenario", Path(path).stem, path=str(path), warmup=0.0)
