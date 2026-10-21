"""Experiment specs: validation, seed-major expansion, content keys (plan Q.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from urbanflow.core.errors import ConfigError
from urbanflow.experiments.spec import expand, parse_experiment


def _doc(**extra: Any) -> dict[str, Any]:
    return {
        "name": "ctrl",
        "scenario": {"bundled": "single_intersection"},
        "base_config": {"duration": 60},
        "variants": [
            {"name": "fixed"},
            {"name": "mp", "policy": {"kind": "controller", "controller": "max_pressure"}},
        ],
        **extra,
    }


def test_expansion_is_seed_major_with_stable_keys(tmp_path: Path) -> None:
    spec = parse_experiment(_doc(seeds={"start": 5, "count": 2}, record_replay="first_seed"))
    runs = expand(spec, tmp_path, "e1", tmp_path / "runs")
    assert [(r.seed, r.variant) for r in runs] == [(5, "fixed"), (5, "mp"), (6, "fixed"), (6, "mp")]
    assert [r.record_replay for r in runs] == [True, True, False, False]
    assert runs[0].run_id == "e1-fixed-s5" and runs[0].config == {"duration": 60, "seed": 5}
    assert runs[1].policy.controllers() == {"*": {"type": "max_pressure", "params": {}}}
    assert len({r.content_key for r in runs}) == 4
    again = expand(spec, tmp_path, "e2", tmp_path / "runs")
    assert [r.content_key for r in again] == [r.content_key for r in runs]
    assert spec.baseline_name == "fixed" and spec.seed_list() == [5, 6]


def test_demand_scale_changes_the_scenario(tmp_path: Path) -> None:
    doc = _doc()
    doc["variants"][1]["demand_scale"] = 1.5
    runs = expand(parse_experiment(doc), tmp_path, "e", tmp_path / "runs")
    assert runs[0].scenario_hash != runs[1].scenario_hash


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"variants": [{"name": "a"}, {"name": "a"}]}, "variant names must be unique"),
        ({"baseline": "nope"}, 'baseline "nope" is not a variant'),
        ({"scenario": {"bundled": "x", "generator": "grid"}}, "exactly one of"),
        ({"variants": [{"name": "Bad Name"}]}, "variants[0].name"),
        ({"variants": [{"name": "a", "policy": {"kind": "controller"}}]}, "needs"),
        ({"seeds": [1, 1]}, "seeds must be distinct"),
    ],
)
def test_invalid_specs_name_the_problem(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse_experiment(_doc(**change))
    text = str(info.value) + " ".join(f"{i.path}: {i.message}" for i in info.value.issues)
    assert message in text


def test_config_errors_carry_the_variant_path(tmp_path: Path) -> None:
    doc = _doc()
    doc["variants"][1]["config"] = {"dt": 5}
    with pytest.raises(ConfigError, match=r"variants\[1\]\.config\.dt"):
        expand(parse_experiment(doc), tmp_path, "e", tmp_path / "runs")


def test_scenario_paths_stay_in_the_workspace(tmp_path: Path) -> None:
    spec = parse_experiment(_doc(scenario={"path": "../outside.json"}))
    with pytest.raises(ConfigError, match="leaves the workspace"):
        expand(spec, tmp_path / "ws", "e", tmp_path / "runs")
