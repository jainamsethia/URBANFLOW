"""Name -> component registries with lazy plugin discovery (plan B.2 #10).

Built-in components register when their module is imported. Third-party packages expose
modules in the ``urbanflow.plugins`` entry-point group; importing such a module registers
its components as a side effect. A registry loads the entry points the first time a
lookup misses, so plugins cost nothing until they are needed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib import metadata
from typing import overload

from urbanflow.core.errors import ConfigError, NotFoundError, suggest

__all__ = ["PLUGIN_GROUP", "Registry", "load_entry_points"]

PLUGIN_GROUP = "urbanflow.plugins"
_log = logging.getLogger(__name__)


def load_entry_points() -> list[str]:
    """Import every ``urbanflow.plugins`` entry point; return the names that loaded.

    A plugin that fails to import is logged at WARNING and skipped, so one broken
    package cannot break unrelated lookups. Already imported modules are not re-run.
    """
    loaded: list[str] = []
    for ep in sorted(metadata.entry_points(group=PLUGIN_GROUP), key=lambda e: e.name):
        try:
            ep.load()
        except Exception as exc:
            _log.warning("failed to load plugin %r (%s): %s", ep.name, ep.value, exc)
        else:
            loaded.append(ep.name)
    return loaded


class Registry[T]:
    """A mapping from names to components of one ``kind`` (e.g. ``"controllers"``)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}
        self._plugins_loaded = False

    @overload
    def register(self, name: str, obj: None = None) -> Callable[[T], T]: ...
    @overload
    def register(self, name: str, obj: T) -> T: ...
    def register(self, name: str, obj: T | None = None) -> T | Callable[[T], T]:
        """Register ``obj`` under ``name``; without ``obj``, return a decorator that does.

        Registering a different object under a taken name raises :class:`ConfigError`.
        """
        if obj is None:

            def decorator(o: T) -> T:
                return self.register(name, o)

            return decorator
        existing = self._items.get(name)
        if existing is not None and existing is not obj:
            raise ConfigError(f'{self.kind} "{name}" is already registered ({existing!r})')
        self._items[name] = obj
        return obj

    def get(self, name: str) -> T:
        """The component registered as ``name``; plugins are loaded on the first miss.

        Raises :class:`NotFoundError` with a did-you-mean hint and the available names.
        """
        if name not in self._items:
            self._load_plugins()
        try:
            return self._items[name]
        except KeyError:
            hint = suggest(name, self._items)
            available = ", ".join(sorted(self._items)) or "none"
            raise NotFoundError(
                f'unknown {self.kind} "{name}"{hint} (available: {available})'
            ) from None

    def names(self) -> list[str]:
        """Sorted names, including plugins."""
        self._load_plugins()
        return sorted(self._items)

    def items(self) -> list[tuple[str, T]]:
        """Sorted ``(name, component)`` pairs, including plugins."""
        self._load_plugins()
        return sorted(self._items.items(), key=lambda kv: kv[0])

    def __contains__(self, name: object) -> bool:
        if name not in self._items:
            self._load_plugins()
        return name in self._items

    def _load_plugins(self) -> None:
        if not self._plugins_loaded:
            self._plugins_loaded = True
            load_entry_points()
