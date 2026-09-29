"""Every ```python block in docs/ runs (plan AE.1); ``# doc-test: skip`` opts a block out."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance

DOCS = Path(__file__).resolve().parents[2] / "docs"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _snippets() -> list[tuple[str, str]]:
    found = []
    for page in sorted(DOCS.glob("*.md")):
        for i, match in enumerate(BLOCK.finditer(page.read_text(encoding="utf-8"))):
            if "# doc-test: skip" not in match.group(1):
                found.append((f"{page.name}[{i}]", match.group(1)))
    return found


@pytest.mark.parametrize(("name", "code"), _snippets(), ids=[n for n, _ in _snippets()])
def test_snippet_runs(name: str, code: str, workspace: Path) -> None:
    exec(compile(code, name, "exec"), {"__name__": "__doc_snippet__"})  # noqa: S102
