"""``urbanflow benchmark`` end to end: fresh process per repeat, report, regression exit
code (plan S.2, AT-57)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urbanflow.benchmark import BenchmarkCase, load_report, run_case
from urbanflow.cli import main as cli

pytestmark = pytest.mark.integration


def test_run_case_measures_and_checks_determinism() -> None:
    case = BenchmarkCase(
        "t", "tiny", generator="grid", params={"rows": 1, "cols": 1}, warmup=10, timed=60
    )
    r = run_case(case, repeats=2)
    assert r.timed_steps == 60 and r.n_intersections == 1 and r.max_active > 0
    assert not r.nondeterministic and len(r.steps_per_s.values) == 2
    assert r.rtf.median == pytest.approx(r.steps_per_s.median * r.dt)
    assert r.peak_rss_mb.median > 10 and r.compile_s.median > 0


def test_benchmark_command_and_regression_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "now.json"
    with pytest.raises(SystemExit) as info:
        cli.main(
            [
                "benchmark",
                "grid_scaling",
                "--case",
                "g1",
                "--quick",
                "--repeat",
                "2",
                "-o",
                str(out),
            ]
        )
    assert info.value.code == 0
    report = load_report(out)
    assert [c.case for c in report.cases] == ["g1"] and report.settings == {
        "repeats": 2,
        "quick": True,
    }
    assert report.machine["numpy"] and "grid_scaling/g1" in capsys.readouterr().out

    fast = json.loads(out.read_text(encoding="utf-8"))  # a baseline ten times faster
    for key in ("steps_per_s", "vehicle_steps_per_s"):
        s = fast["cases"][0][key]
        s.update({k: s[k] * 10 for k in ("median", "q1", "q3")})
    base = tmp_path / "base.json"
    base.write_text(json.dumps(fast), encoding="utf-8")
    with pytest.raises(SystemExit) as info:
        cli.main(
            [
                "benchmark",
                "grid_scaling",
                "--case",
                "g1",
                "--quick",
                "--repeat",
                "1",
                "--compare",
                str(base),
            ]
        )
    assert info.value.code == 10
    assert "REGRESSION grid_scaling/g1" in capsys.readouterr().out

    with pytest.raises(SystemExit) as info:
        cli.main(["benchmark", "grid_scaling", "--case", "g99"])
    assert info.value.code == 4
