"""Deep validation, ``urbanflow.check`` (plan E.8 stage X): E903, E904 and compile issues."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import urbanflow
from urbanflow import check
from urbanflow.cli import main as cli
from urbanflow.core.config import SimulationConfig
from urbanflow.core.errors import NotFoundError, Severity
from urbanflow.routing import router_registry
from urbanflow.scenario import Scenario
from urbanflow.vehicles import car_following_registry

DEMO = Path(__file__).resolve().parents[1] / "fixtures" / "scenarios" / "demo.json"
_DEMO: dict[str, Any] = json.loads(DEMO.read_text(encoding="utf-8"))


def _demo(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(_DEMO)
    data.update(changes)
    return data


def _codes(report: urbanflow.ValidationReport) -> list[tuple[str, str]]:
    return [(i.code, i.path) for i in report.issues]


def test_a_clean_scenario_passes() -> None:
    scenario = urbanflow.generate("single_intersection", kind="uncontrolled")
    report = check(scenario)
    assert report.ok and report.issues == ()
    assert report.spec is scenario.spec and report.resolved is scenario.resolved
    assert urbanflow.check is check


def test_e904_unknown_router_and_car_following_model() -> None:
    config = SimulationConfig(router="shortst", car_following="idn")
    report = check(Scenario.load(DEMO), config)
    assert _codes(report) == [
        ("E904", "simulation.router"),
        ("E904", "simulation.car_following"),
    ]
    router, model = report.errors
    routers = ", ".join(router_registry.names())  # other tests may register more
    models = ", ".join(car_following_registry.names())
    assert "shortest" in routers and "idm" in models
    assert router.message == (
        f'unknown router "shortst" (did you mean "shortest"?) (available: {routers})'
    )
    assert model.message == (
        f'unknown car-following model "idn" (did you mean "idm"?) (available: {models})'
    )


def test_e904_from_the_scenario_simulation_block() -> None:
    scenario = Scenario.from_dict(_demo(simulation={"router": "fastest"}))
    assert _codes(check(scenario)) == [("E904", "simulation.router")]
    # a config layer fixes it: config fields override the scenario block
    assert check(scenario, SimulationConfig(router="shortest")).ok


def test_e903_model_params() -> None:
    data = _demo()
    n = len(data["vehicle_types"])
    data["vehicle_types"] += [
        {"id": "fast", "model_params": {"deltta": 4.0}},
        {"id": "odd", "model_params": {"delta": -1.0}},
    ]
    report = check(Scenario.from_dict(data))
    assert _codes(report) == [
        ("E903", f"vehicle_types[{n}].model_params.deltta"),
        ("E004", f"vehicle_types[{n + 1}].model_params.delta"),
    ]
    assert report.errors[0].message == (
        'unknown parameter "deltta" for car-following model "idm" (known: delta)'
    )
    assert report.errors[1].message == "must be > 0 (got -1)"


def test_compile_warning_w302_and_strict() -> None:
    data = _demo()
    data["network"]["intersections"][0]["signal"]["phases"][0]["green"]["N_in->E_out"] = "G"
    scenario = Scenario.from_dict(data)
    report = check(scenario)
    assert report.ok
    assert _codes(report) == [("W302", "network.intersections[0].signal.phases[0]")]
    strict = check(scenario, strict=True)
    assert not strict.ok and strict.errors[0].severity is Severity.error


def test_compile_error_e806() -> None:
    data = {
        "format": "urbanflow.scenario",
        "version": "1.0",
        "meta": {"name": "chain"},
        "network": {
            "intersections": [
                {"id": "A", "point": [-200, 0]},
                {"id": "K1", "point": [0, 0]},
                {"id": "K2", "point": [24, 0]},
                {"id": "B", "point": [224, 0]},
            ],
            "roads": [
                {"id": "A_K1", "from": "A", "to": "K1", "lanes": [{}]},
                {"id": "K1_K2", "from": "K1", "to": "K2", "lanes": [{}]},
                {"id": "K2_B", "from": "K2", "to": "B", "lanes": [{}]},
            ],
        },
        "demand": {
            "flows": [
                {"id": "f", "route": ["A_K1", "K1_K2", "K2_B"], "vehicle_type": "bus", "rate": 60}
            ]
        },
    }
    report = check(Scenario.from_dict(data))
    assert _codes(report) == [("E806", "network.roads[1].lanes[0]")]


def test_file_level_warnings_come_first(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_demo(simulation={"dt": 1.5, "router": "x"})), encoding="utf-8")
    report = check(path)
    codes = [i.code for i in report.issues]
    assert "W701" in codes and codes[-1] == "E904"  # file-level issues, then deep ones
    assert report.spec is not None


def test_files_that_do_not_load_give_a_report(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{oops", encoding="utf-8")
    report = check(broken)
    assert [i.code for i in report.errors] == ["E000"] and report.spec is None
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(_demo(format="cityflow")), encoding="utf-8")
    assert [i.code for i in check(invalid).errors] == ["E010"]
    with pytest.raises(NotFoundError, match="not found"):
        check(tmp_path / "missing.json")


def test_cli_validate_runs_the_deep_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_demo(simulation={"car_following": "krauss"})), encoding="utf-8")

    def run(*argv: str) -> int:
        with pytest.raises(SystemExit) as info:
            cli.main(list(argv))
        return int(info.value.code or 0)

    assert run("validate", str(path)) == 3
    err = capsys.readouterr().err
    assert '  - simulation.car_following: unknown car-following model "krauss"' in err
    assert run("validate", "--no-deep", str(path)) == 0
