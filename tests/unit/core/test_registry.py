"""Registries: registration, lookup hints and lazy entry-point plugins (plan B.2 #10)."""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path

import pytest

from urbanflow.core.errors import ConfigError, NotFoundError
from urbanflow.core.registry import PLUGIN_GROUP, Registry, load_entry_points


@pytest.fixture
def no_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "entry_points", lambda **_: [])


def test_register_and_get(no_plugins: None) -> None:
    reg = Registry[object]("controllers")

    @reg.register("fixed_time")
    class FixedTime:
        pass

    assert reg.register("external", int) is int
    assert reg.get("fixed_time") is FixedTime
    assert reg.names() == ["external", "fixed_time"]
    assert reg.items() == [("external", int), ("fixed_time", FixedTime)]
    assert "external" in reg and "nope" not in reg


def test_duplicate_name_is_an_error_but_reregistering_same_object_is_not(no_plugins: None) -> None:
    reg = Registry[object]("routers")
    reg.register("shortest", str)
    reg.register("shortest", str)
    with pytest.raises(ConfigError, match='routers "shortest" is already registered'):
        reg.register("shortest", int)


def test_unknown_name_has_hint_and_list(no_plugins: None) -> None:
    reg = Registry[int]("car_following")
    reg.register("idm", 1)
    reg.register("krauss", 2)
    with pytest.raises(NotFoundError) as info:
        reg.get("imd")
    assert (
        str(info.value)
        == 'unknown car_following "imd" (did you mean "idm"?) (available: idm, krauss)'
    )
    assert isinstance(info.value, LookupError)
    with pytest.raises(NotFoundError, match=r"\(available: none\)"):
        Registry[int]("empty").get("x")


# ------------------------------------------------------------------ entry points

HOST = "uf_test_registry_host"
PLUGIN = "uf_test_fake_plugin"
BROKEN = "uf_test_broken_plugin"


@pytest.fixture
def fake_distribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """A fake installed distribution exposing two ``urbanflow.plugins`` entry points."""
    host = types.ModuleType(HOST)
    host.reg = Registry[int]("widgets")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, HOST, host)
    (tmp_path / f"{PLUGIN}.py").write_text(
        f"from {HOST} import reg\nreg.register('gizmo', 42)\n", encoding="utf-8"
    )
    (tmp_path / f"{BROKEN}.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls: list[str] = []

    def entry_points(*, group: str) -> list[metadata.EntryPoint]:
        calls.append(group)
        return [
            metadata.EntryPoint(name="good", value=PLUGIN, group=PLUGIN_GROUP),
            metadata.EntryPoint(name="bad", value=BROKEN, group=PLUGIN_GROUP),
        ]

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    yield calls
    for name in (PLUGIN, BROKEN):
        sys.modules.pop(name, None)


def test_miss_loads_entry_points_once(
    fake_distribution: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    reg: Registry[int] = sys.modules[HOST].reg  # type: ignore[attr-defined]
    with caplog.at_level(logging.WARNING, logger="urbanflow.core.registry"):
        assert reg.get("gizmo") == 42
    assert fake_distribution == [PLUGIN_GROUP]
    assert "failed to load plugin 'bad'" in caplog.text and "boom" in caplog.text
    with pytest.raises(NotFoundError, match="available: gizmo"):
        reg.get("gadget")
    assert "gadget" not in reg
    assert reg.names() == ["gizmo"]
    assert fake_distribution == [PLUGIN_GROUP]  # loaded once per registry


def test_hit_does_not_load_plugins(fake_distribution: list[str]) -> None:
    reg = Registry[int]("local")
    reg.register("a", 1)
    assert reg.get("a") == 1 and "a" in reg
    assert fake_distribution == []


def test_load_entry_points_reports_loaded_names(fake_distribution: list[str]) -> None:
    assert load_entry_points() == ["good"]
    assert load_entry_points() == ["good"]  # already imported modules are not re-run
    assert sys.modules[HOST].reg.names() == ["gizmo"]  # type: ignore[attr-defined]
