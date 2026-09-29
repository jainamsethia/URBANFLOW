"""Architecture test: imports only point downward through the layers (plan section B).

L0 core | L1 geometry, scenario | L2 network, routing | L3 vehicles, signals, demand |
L4 engine | L5 metrics, replay, visualization, facade | L6 rl, experiments, benchmark,
plugins | L7 server, cli.  Same-layer edges are forbidden unless listed below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import urbanflow

PKG_ROOT = Path(urbanflow.__file__).parent

LAYER: dict[str, int] = {
    "_version": 0,
    "core": 0,
    "geometry": 1,
    "scenario": 1,
    "network": 2,
    "routing": 2,
    "vehicles": 3,
    "signals": 3,
    "demand": 3,
    "engine": 4,
    "metrics": 5,
    "replay": 5,
    "visualization": 5,
    "simulation": 5,
    "views": 5,
    "control": 5,
    "results": 5,
    "snapshot": 5,
    "checks": 5,
    "rl": 6,
    "experiments": 6,
    "benchmark": 6,
    "plugins": 6,
    "server": 7,
    "cli": 7,
    "__main__": 7,
}

# Allowed same-layer edges (importer component -> imported component).
ALLOWED_SAME_LAYER: set[tuple[str, str]] = {
    ("scenario", "geometry"),
    ("routing", "network"),
    ("demand", "vehicles"),
    ("replay", "visualization"),
    ("replay", "metrics"),
    ("visualization", "metrics"),
    ("views", "visualization"),
    ("views", "control"),
    ("results", "metrics"),
    ("snapshot", "metrics"),
    ("control", "views"),
    ("simulation", "metrics"),
    ("simulation", "replay"),
    ("simulation", "visualization"),
    ("simulation", "views"),
    ("simulation", "control"),
    ("simulation", "results"),
    ("simulation", "snapshot"),
    ("simulation", "checks"),
    ("benchmark", "experiments"),
    ("benchmark", "rl"),
    ("experiments", "rl"),
    ("plugins", "rl"),
    ("plugins", "experiments"),
    ("cli", "server"),
    ("__main__", "cli"),
}

# Module-specific exceptions (importer module, imported component).
ALLOWED_MODULE_EDGES: set[tuple[str, str]] = {
    ("urbanflow.replay.verify", "simulation"),  # leaf; never imported by replay/__init__
}

# Third-party packages that must never be imported below layer 6 (engine stays pure).
FRAMEWORKS = {
    "fastapi",
    "starlette",
    "uvicorn",
    "typer",
    "click",
    "gymnasium",
    "pettingzoo",
    "sqlalchemy",
    "stable_baselines3",
    "torch",
}
FRAMEWORK_ALLOWED: dict[str, set[str]] = {
    "sqlalchemy": {"experiments", "server", "cli"},
    "gymnasium": {"rl", "experiments", "benchmark", "server", "cli", "plugins"},
    "pettingzoo": {"rl", "experiments", "benchmark", "server", "cli", "plugins"},
    "stable_baselines3": {"rl", "experiments"},
    "torch": {"rl", "experiments"},
    "fastapi": {"server"},
    "starlette": {"server"},
    "uvicorn": {"server", "cli"},
    "typer": {"cli"},
    "click": {"cli"},
}


def _lazy_targets() -> dict[str, str]:
    """name -> module for the lazy public API in urbanflow/__init__.py."""
    tree = ast.parse((PKG_ROOT / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_LAZY":
            value = ast.literal_eval(node.value)  # type: ignore[arg-type]
            return {name: mod for name, (mod, _attr) in value.items()}
    return {}


LAZY = _lazy_targets()


def _module_name(path: Path) -> str:
    rel = path.relative_to(PKG_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _component(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "urbanflow" or len(parts) < 2:
        return None
    return parts[1]


def _imports(path: Path, module: str) -> list[tuple[str, int]]:
    """Absolute module names imported by a file (including lazy function-level imports)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target == "urbanflow":
                for alias in node.names:
                    if alias.name in LAYER:
                        found.append((f"urbanflow.{alias.name}", node.lineno))
                    elif alias.name in LAZY:
                        found.append((LAZY[alias.name], node.lineno))
            else:
                found.append((target, node.lineno))
                found += [
                    (f"{target}.{alias.name}", node.lineno)
                    for alias in node.names
                    if target.startswith("urbanflow")
                ]
    return found


def _source_files() -> list[Path]:
    return sorted(p for p in PKG_ROOT.rglob("*.py") if "_frontend" not in p.parts)


def test_every_component_has_a_layer() -> None:
    unknown = set()
    for path in _source_files():
        comp = _component(_module_name(path))
        if comp is not None and comp not in LAYER:
            unknown.add(comp)
    assert not unknown, f"add these components to LAYER: {sorted(unknown)}"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(PKG_ROOT)))
def test_imports_respect_layers(path: Path) -> None:
    module = _module_name(path)
    comp = _component(module)
    if comp is None:  # the package root only maps names lazily
        return
    layer = LAYER[comp]
    problems: list[str] = []
    for target, lineno in _imports(path, module):
        top = target.split(".")[0]
        if top in FRAMEWORKS and comp not in FRAMEWORK_ALLOWED.get(top, set()):
            problems.append(f"line {lineno}: {comp} (L{layer}) must not import {top}")
            continue
        tcomp = _component(target)
        if tcomp is None or tcomp == comp or tcomp not in LAYER:
            continue
        tlayer = LAYER[tcomp]
        if tlayer > layer:
            problems.append(f"line {lineno}: {comp} (L{layer}) imports {target} (L{tlayer})")
        elif (
            tlayer == layer
            and (comp, tcomp) not in ALLOWED_SAME_LAYER
            and (module, tcomp) not in ALLOWED_MODULE_EDGES
        ):
            problems.append(f"line {lineno}: same-layer import {comp} -> {tcomp} not allowed")
    assert not problems, "\n".join(problems)


# No magic numbers in the simulation core (plan U.1, core/constants.py row): float literals
# with |x| > 1 in L2-L4 modules (network .. engine) must come from core/constants.py.
def _big_float_literals(source: str) -> list[tuple[int, float]]:
    return [
        (node.lineno, node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and type(node.value) is float and abs(node.value) > 1
    ]


def test_engine_float_literals_come_from_constants() -> None:
    assert _big_float_literals("x = 2 * v + 0.5 - 2.0 * b + 1e-9 - 1.0") == [(1, 2.0)]
    problems = [
        f"{path.relative_to(PKG_ROOT)}:{line}: float literal {value!r}; use core/constants.py"
        for path in _source_files()
        if 2 <= LAYER.get(_component(_module_name(path)) or "", -1) <= 4
        for line, value in _big_float_literals(path.read_text(encoding="utf-8"))
    ]
    assert not problems, "\n".join(problems)
