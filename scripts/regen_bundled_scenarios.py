"""Regenerate the bundled scenarios (package data) from their generators (plan B.2 #21).

Usage::

    uv run python scripts/regen_bundled_scenarios.py          # rewrite every bundled file
    uv run python scripts/regen_bundled_scenarios.py --check  # exit 1 if any file is stale

Every entry of ``urbanflow.scenario.generators.BUNDLED_SCENARIOS`` is generated with its
fixed parameters, so the output is deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from urbanflow.scenario.generators import BUNDLED_SCENARIOS, generate

BUNDLED_DIR = Path(__file__).resolve().parents[1] / "src" / "urbanflow" / "scenario" / "bundled"


def render(name: str) -> str:
    """The exact file content of bundled scenario ``name``."""
    generator, params = BUNDLED_SCENARIOS[name]
    return generate(generator, dict(params)).to_json() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report stale files")
    args = parser.parse_args(argv)
    stale = []
    for name in sorted(BUNDLED_SCENARIOS):
        path = BUNDLED_DIR / f"{name}.json"
        text = render(name)
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == text:
            continue
        stale.append(path)
        if not args.check:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path}")
    if args.check and stale:
        print("stale bundled scenarios: " + ", ".join(p.name for p in stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
