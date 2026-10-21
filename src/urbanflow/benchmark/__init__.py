"""Reproducible performance benchmarks (plan S.2): suites, harness, report."""

from urbanflow.benchmark.harness import run_case, run_suites
from urbanflow.benchmark.report import (
    BenchmarkReport,
    CaseResult,
    Regression,
    compare,
    load_report,
    machine_key,
    save_report,
)
from urbanflow.benchmark.suites import SUITES, BenchmarkCase, scenario_case

__all__ = [
    "SUITES",
    "BenchmarkCase",
    "BenchmarkReport",
    "CaseResult",
    "Regression",
    "compare",
    "load_report",
    "machine_key",
    "run_case",
    "run_suites",
    "save_report",
    "scenario_case",
]
