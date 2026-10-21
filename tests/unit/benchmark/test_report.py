"""Benchmark report: statistics, regression check, files, suites (plan S.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from urbanflow.benchmark import (
    SUITES,
    BenchmarkReport,
    CaseResult,
    compare,
    load_report,
    machine_key,
    save_report,
)
from urbanflow.benchmark.report import stat
from urbanflow.benchmark.suites import QUICK_TIMED, QUICK_WARMUP, scenario_case
from urbanflow.core.errors import ConfigError, NotFoundError


def _case(name: str, veh_steps: list[float], steps: list[float] | None = None) -> CaseResult:
    steps = steps or [100.0] * len(veh_steps)
    one = stat([1.0] * len(veh_steps))
    return CaseResult(
        suite="s", case=name, params={}, config={}, n_intersections=4, n_lanes=48,
        warmup_s=60, timed_steps=300, dt=1.0, mean_active=10.0, max_active=12,
        compile_s=one, steps_per_s=stat(steps), vehicle_steps_per_s=stat(veh_steps),
        rtf=stat(steps), wall_s=one, peak_rss_mb=one, state_digest="d", nondeterministic=False,
    )  # fmt: skip


def _report(*cases: CaseResult) -> BenchmarkReport:
    return BenchmarkReport(
        created_utc="2026-10-22T00:00:00Z",
        machine={"key": machine_key()},
        settings={},
        cases=list(cases),
    )


def test_stat_is_median_and_quartiles() -> None:
    s = stat([1.0, 2.0, 3.0, 4.0, 100.0])
    assert (s.q1, s.median, s.q3) == (2.0, 3.0, 4.0) and s.values[-1] == 100.0


def test_compare_flags_real_slowdowns_only() -> None:
    base = _report(
        _case("slow", [1000, 1010, 1020]),
        _case("noisy", [1000, 1010, 1020]),
        _case("small", [1000, 1010, 1020]),
        _case("empty", [0, 0, 0], steps=[500, 505, 510]),
    )
    new = _report(
        _case("slow", [700, 705, 710]),  # -30 %, quartiles apart
        _case("noisy", [600, 1000, 1400]),  # median -1 %... and wide
        _case("small", [960, 965, 970]),  # -4.5 %: within 10 %
        _case("empty", [0, 0, 0], steps=[300, 301, 302]),  # empty network: steps/s
        _case("new-case", [1, 1, 1]),
    )
    found = compare(base, new, threshold=0.10)
    assert [(r.case, r.metric) for r in found] == [
        ("slow", "vehicle_steps_per_s"),
        ("empty", "steps_per_s"),
    ]
    assert found[0].change_pct == pytest.approx(100 * (705 / 1010 - 1))
    with pytest.raises(ConfigError):
        compare(base, new, threshold=1.5)


def test_report_files_round_trip(tmp_path: Path) -> None:
    report = _report(_case("a", [1, 2, 3]))
    path = save_report(report, tmp_path / "sub" / "r.json")
    assert load_report(path) == report and report.machine_key == machine_key()
    (tmp_path / "bad.json").write_text('{"format": "something else"}', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_report(tmp_path / "bad.json")
    with pytest.raises(NotFoundError):
        load_report(tmp_path / "missing.json")


def test_suites_build_valid_scenarios() -> None:
    from urbanflow.scenario.generators import generate

    names = {s: [c.name for c in cases] for s, cases in SUITES.items()}
    assert names["grid_scaling"] == ["g1", "g2", "g4", "g8", "g12"]
    for cases in SUITES.values():
        for case in cases:
            assert case.durations(quick=True) == (QUICK_WARMUP, QUICK_TIMED)
            if case.params["rows"] <= 8:  # g12 takes seconds; same generator, same params
                assert generate("grid", **case.params).name
    g8 = next(c for c in SUITES["grid_scaling"] if c.name == "g8")
    assert g8.params["entry_rate"] == 360 * 64 / 32  # 360 veh/h per junction over 32 entries
    case = scenario_case("maps/town.json")
    assert (case.suite, case.name, case.path) == ("scenario", "town", "maps/town.json")
