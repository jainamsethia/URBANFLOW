"""Speed against this machine's committed baseline (plan S.2): deselected by default; run
with ``pytest -m performance``. Skipped when ``benchmarks/baselines/<machine key>.json`` does
not exist; regenerate that file with the command in ``benchmarks/README.md``."""

from __future__ import annotations

from pathlib import Path

import pytest

from urbanflow.benchmark import SUITES, compare, load_report, machine_key, run_suites

pytestmark = pytest.mark.performance

BASELINES = Path(__file__).resolve().parents[2] / "benchmarks" / "baselines"
THRESHOLD = 0.30
"""Shared and laptop machines are noisy: only a slowdown of more than 30 % fails."""


def test_quick_suites_are_not_slower_than_the_baseline() -> None:
    path = BASELINES / f"{machine_key()}.json"
    if not path.is_file():
        pytest.skip(f"no baseline for {machine_key()} in {BASELINES}")
    base = load_report(path)
    wanted = {(c.suite, c.case) for c in base.cases}
    cases = [c for cs in SUITES.values() for c in cs if (c.suite, c.name) in wanted]
    report = run_suites(cases, repeats=int(base.settings.get("repeats", 3)), quick=True)
    found = compare(base, report, THRESHOLD)
    assert not found, "\n".join(
        f"{r.suite}/{r.case}: {r.metric} {r.baseline:,.0f} -> {r.current:,.0f} "
        f"({r.change_pct:+.1f} %)"
        for r in found
    )
    assert not any(c.nondeterministic for c in report.cases)
