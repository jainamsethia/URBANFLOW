"""AT-02: generate a scenario, then validate it (plan section V)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance


def _cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "urbanflow", *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=cwd,
    )


def test_generate_then_validate(tmp_path: Path) -> None:
    generated = _cli(tmp_path, "generate", "single_intersection", "-o", "s.json")
    assert generated.returncode == 0, generated.stderr
    assert generated.stdout.startswith("Wrote s.json (5 intersections, 8 roads, 12 flows; hash ")
    validated = _cli(tmp_path, "validate", "s.json")
    assert validated.returncode == 0, validated.stderr
    line = validated.stdout.strip()
    assert re.fullmatch(
        r"OK: single_intersection \(hash [0-9a-f]{12}; 5 intersections, 8 roads, 16 lanes, "
        r"12 movements, 12 flows\)",
        line,
    ), line
    short_hash = generated.stdout.rsplit("hash ", 1)[1].strip().rstrip(")")
    assert f"hash {short_hash};" in line
