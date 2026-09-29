"""``KEY=VALUE`` parsing shared by ``-p``, ``--set`` and ``-i`` (plan AC 7.1)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, NoReturn

from urbanflow.core.errors import ConfigError

__all__ = ["parse_assignments"]


def _reject(item: str, reason: str) -> NoReturn:
    raise ConfigError(f'invalid assignment "{item}": {reason}')


def _literal(raw: str) -> Any:
    """A JSON literal (number, true/false/null, array, object, "string") or the raw text."""

    def no_constants(_token: str) -> Any:
        raise ValueError

    try:
        return json.loads(raw, parse_constant=no_constants)
    except ValueError:
        return raw


def parse_assignments(items: Iterable[str]) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` items into a nested dict.

    Dotted keys nest (``turn_ratios.far=0.2`` -> ``{"turn_ratios": {"far": 0.2}}``) and
    values are JSON literals with a fallback to the raw string (``kind=priority``). Later
    items override earlier ones. Malformed items and conflicts between a value and a nested
    key raise :class:`ConfigError` naming the offending item.
    """
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            _reject(item, "expected KEY=VALUE")
        parts = key.split(".")
        if not all(parts):
            _reject(item, f'empty segment in key "{key}"')
        node = result
        for depth, part in enumerate(parts[:-1]):
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                _reject(item, f'"{".".join(parts[: depth + 1])}" already has a value')
            node = child
        if isinstance(node.get(parts[-1]), dict):
            _reject(item, f'"{key}" already has nested keys')
        node[parts[-1]] = _literal(raw)
    return result
