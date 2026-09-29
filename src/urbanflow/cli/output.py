"""Console helpers: results go to stdout, logs/progress/errors to stderr."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from rich.console import Console
from rich.table import Table

from urbanflow.core.errors import ValidationIssue, format_issues

__all__ = [
    "console",
    "emit_json",
    "err_console",
    "print_issues",
    "print_summary",
    "print_table",
]

console = Console(highlight=False, soft_wrap=False)
err_console = Console(stderr=True, highlight=False)


def emit_json(obj: Any) -> None:
    """Print ``obj`` as JSON on stdout (nothing else is printed in ``--json`` mode)."""
    console.print_json(json.dumps(obj, ensure_ascii=False, default=str))


def print_issues(
    issues: Sequence[ValidationIssue], header: str = "Scenario validation failed:"
) -> None:
    """Print issues in the friendly ``  - path: message`` format on stderr."""
    text = format_issues(issues, header=header)
    err_console.print(text, markup=False, highlight=False, soft_wrap=True)


def print_table(
    title: str | None,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    justify: Sequence[str] | None = None,
) -> None:
    """Render a Rich table on stdout."""
    table = Table(title=title, title_justify="center", min_width=len(title) if title else None)
    for i, name in enumerate(columns):
        table.add_column(
            name,
            justify=(justify[i] if justify else "left"),  # type: ignore[arg-type]
            overflow="fold",  # never an ellipsis: legacy Windows consoles cannot render it
        )
    for row in rows:
        table.add_row(*(str(cell) for cell in row))
    console.print(table)


def print_summary(title: str, values: Mapping[str, Any]) -> None:
    """Render a two-column ``Metric | Value`` table."""
    print_table(title, ("Metric", "Value"), values.items(), justify=("left", "right"))
