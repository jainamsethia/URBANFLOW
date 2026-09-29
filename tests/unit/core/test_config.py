"""SimulationConfig: defaults, precedence, set-fields-only merging, errors (plan AB 6.1)."""

from __future__ import annotations

import typing
from pathlib import Path

import pydantic
import pytest

from urbanflow.core.config import (
    ConfigOverrides,
    MetricsConfig,
    RecordConfig,
    SimulationConfig,
    resolve_config,
)
from urbanflow.core.errors import ConfigError, NotFoundError

LAYERS = ["scenario", "urbanflow.toml", "--set", "flags", "config", "overrides"]


def test_defaults() -> None:
    c = SimulationConfig()
    assert (c.dt, c.duration, c.seed) == (1.0, 3600.0, 0)
    assert (c.car_following, c.lane_change_model, c.router) == ("idm", "mobil", "shortest")
    assert (c.routing_weight, c.reroute_period, c.halting_speed) == ("freeflow_time", 60.0, 0.1)
    assert (c.deadlock_timeout, c.max_vehicles, c.accel) == (300.0, None, "numpy")
    assert c.lane_changing and not c.debug_checks
    m = c.metrics
    assert (m.interval, m.warmup, m.collectors, m.tables, m.energy) == (
        10.0,
        0.0,
        ("default",),
        None,
        True,
    )
    r = c.record
    assert (r.enabled, r.path, r.every, r.chunk_rows, r.max_chunk_steps, r.compresslevel) == (
        False,
        None,
        1,
        1_000_000,
        300,
        1,
    )
    assert r.events == (
        "vehicle_inserted",
        "vehicle_arrived",
        "vehicle_teleported",
        "phase_changed",
    )
    assert c.model_fields_set == set()


def test_frozen_and_closed() -> None:
    c = SimulationConfig()
    with pytest.raises(pydantic.ValidationError):
        c.dt = 0.5  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        SimulationConfig(unknown=1)  # type: ignore[call-arg]


@pytest.mark.parametrize("top", range(len(LAYERS)))
def test_precedence_matrix(top: int) -> None:
    # Layers 0..top all set dt; the highest one wins. Every layer also sets its own field,
    # which no other layer touches, so it must survive the merge.
    fields = ["seed", "halting_speed", "deadlock_timeout", "reroute_period", "max_vehicles"]
    layers = []
    for i, name in enumerate(LAYERS):
        data: dict[str, object] = {"dt": 0.1 * (i + 1)} if i <= top else {}
        if i < len(fields):
            data[fields[i]] = i + 1
        layers.append((name, data))
    c = resolve_config(*layers)
    assert c.dt == pytest.approx(0.1 * (top + 1))
    assert (c.seed, c.halting_speed, c.deadlock_timeout, c.reroute_period, c.max_vehicles) == (
        1,
        2,
        3,
        4,
        5,
    )


def test_default_instance_never_clobbers_lower_layers() -> None:
    c = resolve_config(("scenario", {"dt": 0.5, "seed": 7}), ("config", SimulationConfig()))
    assert (c.dt, c.seed) == (0.5, 7)


def test_only_set_fields_count_even_when_equal_to_default() -> None:
    c = resolve_config(("scenario", {"dt": 0.5}), ("config", SimulationConfig(dt=1.0)))
    assert c.dt == 1.0
    assert c.model_fields_set == {"dt"}


def test_nested_tables_merge_key_by_key() -> None:
    c = resolve_config(
        ("scenario", {"metrics": {"interval": 5.0}, "record": {"every": 2}}),
        ("config", SimulationConfig(metrics=MetricsConfig(warmup=100.0))),
        ("overrides", {"record": RecordConfig(enabled=True)}),
        ("none", None),
    )
    assert (c.metrics.interval, c.metrics.warmup) == (5.0, 100.0)
    assert (c.record.every, c.record.enabled) == (2, True)


def test_config_overrides_mirror_the_fields() -> None:
    hints = typing.get_type_hints(ConfigOverrides)
    assert set(hints) == set(SimulationConfig.model_fields)
    for name, field in SimulationConfig.model_fields.items():
        if name not in ("metrics", "record"):
            assert hints[name] == field.annotation, name


def test_json_round_trip() -> None:
    c = SimulationConfig(dt=0.5, duration=None, record=RecordConfig(path=Path("run.ufr")))
    again = SimulationConfig.model_validate(c.model_dump(mode="json"))
    assert again == c


def test_environment_never_feeds_simulation_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("URBANFLOW_DT", "0.5")
    monkeypatch.setenv("URBANFLOW_SEED", "9")
    assert SimulationConfig().dt == 1.0
    assert resolve_config().seed == 0


# ------------------------------------------------------------------ errors


def _issues(*layers: tuple[str, object]) -> list[tuple[str, str, str]]:
    with pytest.raises(ConfigError) as info:
        resolve_config(*layers)  # type: ignore[arg-type]
    return [(i.path, i.message, i.code) for i in info.value.issues]


def test_error_is_layer_tagged_like_the_plan_example() -> None:
    with pytest.raises(ConfigError) as info:
        resolve_config(("scenario", {"dt": 0.5}), ("urbanflow.toml", {"dt": 5}))
    assert "urbanflow.toml: simulation.dt: must be <= 2 (got 5)" in str(info.value)
    assert info.value.exit_code == 4


def test_error_blames_the_layer_that_set_the_value() -> None:
    issues = _issues(("scenario", {"seed": -1}), ("overrides", {"dt": 0.5}))
    assert issues == [("scenario: simulation.seed", "must be >= 0 (got -1)", "E004")]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"dtt": 1}, ("x: simulation.dtt", 'unknown field "dtt" (did you mean "dt"?)', "E002")),
        (
            {"metrics": {"intervall": 3}},
            (
                "x: simulation.metrics.intervall",
                'unknown field "intervall" (did you mean "interval"?)',
                "E002",
            ),
        ),
        (
            {"accel": "cuda"},
            (
                "x: simulation.accel",
                "invalid value 'cuda'; expected one of: 'numpy', 'numba'",
                "E005",
            ),
        ),
        (
            {"dt": 2.0, "metrics": {"interval": 1.0}},
            ("x: simulation.metrics", "metrics.interval (1) must be >= dt (2)", "E007"),
        ),
        ({"duration": 0}, ("x: simulation.duration", "must be > 0 (got 0)", "E004")),
        (
            {"record": {"compresslevel": 12}},
            ("x: simulation.record.compresslevel", "must be <= 9 (got 12)", "E004"),
        ),
        ({"dt": float("nan")}, ("x: simulation.dt", "Input should be a finite number", "E009")),
    ],
)
def test_error_messages(data: dict[str, object], expected: tuple[str, str, str]) -> None:
    assert _issues(("x", data)) == [expected]


def test_non_mapping_layer_is_rejected() -> None:
    with pytest.raises(ConfigError, match="flags: simulation config must be a table"):
        resolve_config(("flags", 5))  # type: ignore[arg-type]


# ------------------------------------------------------------------ TOML file layer


def test_from_toml_reads_the_simulation_table(tmp_path: Path) -> None:
    path = tmp_path / "urbanflow.toml"
    path.write_text(
        '[server]\nport = 9000\n\n[simulation]\ndt = 0.5\nrouter = "dynamic"\n\n'
        "[simulation.metrics]\ninterval = 20.0\n",
        encoding="utf-8",
    )
    c = SimulationConfig.from_toml(path)
    assert (c.dt, c.router, c.metrics.interval) == (0.5, "dynamic", 20.0)
    assert c.model_fields_set == {"dt", "router", "metrics"}
    # As a layer it only contributes what the file set.
    merged = resolve_config(("scenario", {"seed": 4, "dt": 0.2}), ("file", c))
    assert (merged.seed, merged.dt) == (4, 0.5)


def test_from_toml_without_table_gives_unset_defaults(tmp_path: Path) -> None:
    path = tmp_path / "urbanflow.toml"
    path.write_text("[server]\nport = 9000\n", encoding="utf-8")
    c = SimulationConfig.from_toml(path)
    assert c == SimulationConfig() and c.model_fields_set == set()


def test_from_toml_errors(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError, match="config file not found"):
        SimulationConfig.from_toml(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[simulation\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        SimulationConfig.from_toml(bad)
    bad.write_text("simulation = 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[simulation\] must be a table"):
        SimulationConfig.from_toml(bad)
    bad.write_text("[simulation]\ndt = 5\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        SimulationConfig.from_toml(bad)
    assert f"{bad}: simulation.dt: must be <= 2 (got 5)" in str(info.value)
