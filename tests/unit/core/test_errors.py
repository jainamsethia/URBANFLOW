from __future__ import annotations

import pytest

from urbanflow.core.errors import (
    CommandError,
    ConfigError,
    InvariantViolation,
    MissingDependencyError,
    NotFoundError,
    ReplayFormatError,
    ScenarioValidationError,
    Severity,
    SimulationError,
    UrbanFlowError,
    ValidationIssue,
    format_issues,
    format_path,
    require,
    suggest,
)


@pytest.mark.parametrize(
    ("loc", "expected"),
    [
        (
            ("network", "intersections", 3, "signal", "phases", 2),
            "network.intersections[3].signal.phases[2]",
        ),
        (("demand", "flows", 0, "type_mix", "city_bus"), "demand.flows[0].type_mix.city_bus"),
        (("demand", "flows", 0, "type_mix", "city bus"), 'demand.flows[0].type_mix["city bus"]'),
        (("roads", 3, "lanes", 0, "width"), "roads[3].lanes[0].width"),
        (("network", "roads", 0, "from"), "network.roads[0].from"),
        (("flows", 1, "function-after[check(), FlowSpec]", "rate"), "flows[1].rate"),
        (("flows", 1, "int", "count"), "flows[1].count"),
        ((), "$"),
        ((0,), "[0]"),
        (("a", 'q"x'), 'a["q\\"x"]'),
    ],
)
def test_format_path(loc: tuple[str | int, ...], expected: str) -> None:
    assert format_path(loc) == expected


def test_format_issues_matches_the_required_report_shape() -> None:
    issues = [
        ValidationIssue(
            "roadnet.intersections[3].phases[2]",
            'phase contains unknown lane connection "lane_17 -> lane_99"',
            "E302",
        )
    ]
    assert format_issues(issues) == (
        "Scenario validation failed:\n"
        "  - roadnet.intersections[3].phases[2]: "
        'phase contains unknown lane connection "lane_17 -> lane_99"'
    )


def test_scenario_validation_error_lists_errors_not_warnings() -> None:
    err = ScenarioValidationError(
        [
            ValidationIssue("a", "bad", "E001"),
            ValidationIssue("b", "meh", "W701", Severity.warning),
        ]
    )
    assert len(err.errors) == 1 and len(err.warnings) == 1
    assert "a: bad" in str(err) and "meh" not in str(err)
    assert err.to_dict()["warnings"][0]["code"] == "W701"
    assert err.exit_code == 3


def test_warnings_only_error_still_renders_them() -> None:
    err = ScenarioValidationError([ValidationIssue("x", "promoted", "W1", Severity.warning)])
    assert "x: promoted" in str(err)


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (UrbanFlowError("x"), 1),
        (ConfigError("x"), 4),
        (NotFoundError("x"), 5),
        (SimulationError("x"), 6),
        (CommandError("x"), 6),
        (InvariantViolation("I4", 3), 7),
        (ReplayFormatError("x"), 8),
        (MissingDependencyError("polars", "data"), 9),
    ],
)
def test_exit_codes(exc: UrbanFlowError, code: int) -> None:
    assert exc.exit_code == code


def test_error_types_are_also_builtin_types() -> None:
    assert isinstance(NotFoundError("x"), LookupError)
    assert isinstance(CommandError("x"), ValueError)
    assert isinstance(MissingDependencyError("p", "e"), ImportError)


def test_invariant_violation_message() -> None:
    exc = InvariantViolation("I4", 12, uids=range(15), details="overlap -0.2 m")
    text = str(exc)
    assert "I4" in text and "step 12" in text and "+5 more" in text and "overlap" in text
    assert exc.uids[:3] == (0, 1, 2)


def test_missing_dependency_hint() -> None:
    assert "urbanflow[data]" in str(MissingDependencyError("polars", "data"))


def test_config_error_with_issues() -> None:
    err = ConfigError("bad config", [ValidationIssue("simulation.dt", "must be <= 2", "E004")])
    assert "simulation.dt: must be <= 2" in str(err)


def test_suggest() -> None:
    assert suggest("cr", ["car", "bus"]) == ' (did you mean "car"?)'
    assert suggest("zzz", ["car", "bus"]) == ""


def test_require() -> None:
    assert require("json", "core").__name__ == "json"
    with pytest.raises(MissingDependencyError) as info:
        require("definitely_not_a_module_xyz", "data")
    assert info.value.extra == "data"
