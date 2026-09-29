"""Typed simulation configuration and its layered resolution (plan AB 6.1).

Precedence, lowest first: defaults < ``scenario.simulation`` < config file ``[simulation]``
< CLI ``--set`` < CLI flags < Python ``config=`` < ``**overrides``. Each layer contributes
only the fields it explicitly *set* (pydantic ``model_fields_set``), so a default-constructed
``SimulationConfig()`` never clobbers values from lower layers. Environment variables never
feed this config (B.2 #8): runs have no hidden inputs.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator
from pydantic_core import ErrorDetails

from urbanflow.core import constants as C
from urbanflow.core.errors import ConfigError, NotFoundError, ValidationIssue, format_path, suggest

__all__ = [
    "Accel",
    "ConfigOverrides",
    "MetricsConfig",
    "RecordConfig",
    "RecordEvent",
    "RoutingWeight",
    "SimulationConfig",
    "resolve_config",
]

RoutingWeight = Literal["length", "freeflow_time"]
Accel = Literal["numpy", "numba"]
# The canonical EventType values (B.2 #12); core/events.py owns the enum.
RecordEvent = Literal[
    "vehicle_departed",
    "vehicle_inserted",
    "vehicle_entered_link",
    "vehicle_exited_link",
    "vehicle_stopped",
    "vehicle_resumed",
    "vehicle_changed_lane",
    "vehicle_arrived",
    "vehicle_removed",
    "vehicle_teleported",
    "phase_changed",
    "simulation_started",
    "simulation_ended",
]


class _Config(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, use_attribute_docstrings=True
    )


class MetricsConfig(_Config):
    """What the metrics manager collects (plan J, B.2 #8)."""

    interval: float = Field(default=C.METRICS_INTERVAL_S, gt=0)
    """Timeseries sample interval, s (must be >= dt)."""
    warmup: float = Field(default=0.0, ge=0)
    """Trips departing and windows ending before this time are excluded from the summary, s."""
    collectors: tuple[str, ...] = ("default",)
    """Registered collector names."""
    tables: tuple[str, ...] | None = None
    """Tables to keep; None keeps every default table."""
    energy: bool = True
    """Compute the (uncalibrated) energy and CO2 proxy."""


class RecordConfig(_Config):
    """Replay recording options (plan K, B.2 #8)."""

    enabled: bool = False
    path: Path | None = None
    """Replay file; None records to a temp file kept by ``save_replay()``."""
    every: int = Field(default=1, ge=1)
    """Record every n-th step."""
    chunk_rows: int = Field(default=C.RECORD_CHUNK_ROWS, ge=1)
    """A chunk closes once it holds this many vehicle rows..."""
    max_chunk_steps: int = Field(default=C.RECORD_MAX_CHUNK_STEPS, ge=1)
    """...or this many frames."""
    events: tuple[RecordEvent, ...] = (
        "vehicle_inserted",
        "vehicle_arrived",
        "vehicle_teleported",
        "phase_changed",
    )
    """Event types stored in chunks."""
    compresslevel: int = Field(default=C.RECORD_COMPRESSLEVEL, ge=0, le=C.RECORD_COMPRESSLEVEL_MAX)
    """Deflate level of chunk members."""


class SimulationConfig(_Config):
    """Run parameters of one simulation (frozen; plan AB 6.1)."""

    dt: float = Field(default=C.DT, ge=C.DT_MIN, le=C.DT_MAX)
    """Step length, s."""
    duration: float | None = Field(default=C.DURATION, gt=0)
    """Simulated time, s; None runs until demand is exhausted and the network is empty."""
    seed: int = Field(default=0, ge=0, le=C.SEED_MAX)
    """Root seed of every random stream."""
    car_following: str = "idm"
    """Registered car-following model."""
    lane_change_model: str = "mobil"
    """Registered lane-change model."""
    lane_changing: bool = True
    router: str = "shortest"
    """Registered router (``shortest``, ``dynamic`` or a plugin)."""
    routing_weight: RoutingWeight = "freeflow_time"
    reroute_period: float = Field(default=C.REROUTE_PERIOD, gt=0)
    """Dynamic router travel-time refresh period, s."""
    halting_speed: float = Field(default=C.HALTING_SPEED, ge=0)
    """Vehicles slower than this are halting, m/s."""
    deadlock_timeout: float = Field(default=C.DEADLOCK_TIMEOUT, ge=0)
    """Teleport vehicles stuck this long, s; 0 turns the watchdog off."""
    max_vehicles: int | None = Field(default=None, ge=1)
    """Cap on running vehicles; extra demand queues."""
    accel: Accel = "numpy"
    """Kernel backend; numba must be requested explicitly and is recorded with results."""
    debug_checks: bool = False
    """Check every invariant each step (slow)."""
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    record: RecordConfig = Field(default_factory=RecordConfig)

    @field_validator("metrics")
    @classmethod
    def _interval_not_below_dt(cls, value: MetricsConfig, info: ValidationInfo) -> MetricsConfig:
        dt = info.data.get("dt")
        if dt is not None and value.interval < dt:
            raise ValueError(f"metrics.interval ({value.interval:g}) must be >= dt ({dt:g})")
        return value

    @classmethod
    def from_toml(cls, path: str | Path) -> SimulationConfig:
        """Read the ``[simulation]`` table of a TOML file (the config-file layer).

        Only the keys present in the file count as set, so the result can be passed to
        :func:`resolve_config` without masking lower layers. A file without the table
        yields the defaults.
        """
        file = Path(path)
        try:
            text = file.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise NotFoundError(f"config file not found: {file}") from None
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{file}: invalid TOML: {exc}") from exc
        table = data.get("simulation", {})
        if not isinstance(table, dict):
            raise ConfigError(f"{file}: [simulation] must be a table")
        return resolve_config((str(path), table))


class ConfigOverrides(TypedDict, total=False):
    """Keyword overrides of ``Simulation(**overrides)``; mirrors :class:`SimulationConfig`."""

    dt: float
    duration: float | None
    seed: int
    car_following: str
    lane_change_model: str
    lane_changing: bool
    router: str
    routing_weight: RoutingWeight
    reroute_period: float
    halting_speed: float
    deadlock_timeout: float
    max_vehicles: int | None
    accel: Accel
    debug_checks: bool
    metrics: MetricsConfig | Mapping[str, Any]
    record: RecordConfig | Mapping[str, Any]


Layer = tuple[str, BaseModel | Mapping[str, Any] | None]
_NESTED: dict[str, type[BaseModel]] = {"metrics": MetricsConfig, "record": RecordConfig}
_BOUNDS = {
    "less_than_equal": ("<=", "le"),
    "less_than": ("<", "lt"),
    "greater_than_equal": (">=", "ge"),
    "greater_than": (">", "gt"),
}


def _plain(value: Any) -> Any:
    """Models become dicts of their *set* fields; mappings become plain dicts."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=True)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _merge(
    into: dict[str, Any],
    data: Mapping[str, Any],
    layer: str,
    origin: dict[tuple[str, ...], str],
    prefix: tuple[str, ...] = (),
) -> None:
    for key, value in data.items():
        path = (*prefix, key)
        current = into.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            _merge(current, value, layer, origin, path)
        else:
            into[key] = value
            origin[path] = layer


def _num(value: object) -> object:
    return f"{value:g}" if isinstance(value, float) else value


def _issue(err: ErrorDetails) -> tuple[str, str]:
    """``(code, message)`` of a pydantic error, using the scenario structural codes (E.8)."""
    kind, ctx, got = err["type"], err.get("ctx", {}), err.get("input")
    if kind in _BOUNDS:
        op, key = _BOUNDS[kind]
        return "E004", f"must be {op} {_num(ctx[key])} (got {_num(got)})"
    if kind == "extra_forbidden":
        name = str(err["loc"][-1])
        model = _NESTED.get(str(err["loc"][0])) if len(err["loc"]) > 1 else SimulationConfig
        return "E002", f'unknown field "{name}"{suggest(name, model.model_fields if model else ())}'
    if kind == "literal_error":
        choices = str(ctx["expected"]).replace(" or ", ", ")  # pydantic: "'a', 'b' or 'c'"
        return "E005", f"invalid value {got!r}; expected one of: {choices}"
    if kind == "value_error":
        return "E007", str(ctx.get("error", err["msg"]))
    return "E009", err["msg"]


def _layer_of(loc: tuple[str, ...], origin: Mapping[tuple[str, ...], str]) -> str:
    """The layer that supplied the value at ``loc`` (or, for a sub-tree, its last writer)."""
    for n in range(len(loc), 0, -1):
        if loc[:n] in origin:
            return origin[loc[:n]]
    writers = [layer for path, layer in origin.items() if path[: len(loc)] == loc]
    return writers[-1] if writers else "config"


def resolve_config(*layers: Layer) -> SimulationConfig:
    """Merge ``(name, layer)`` pairs, lowest precedence first, and validate once.

    A layer is a ``SimulationConfig`` (or another pydantic model such as the scenario's
    ``simulation`` block), a mapping, or None (skipped). Only explicitly set fields are
    merged; nested ``metrics``/``record`` tables merge key by key. Invalid values raise
    :class:`ConfigError` whose issues name the layer, e.g.
    ``urbanflow.toml: simulation.dt: must be <= 2 (got 5)``.
    """
    merged: dict[str, Any] = {}
    origin: dict[tuple[str, ...], str] = {}
    for name, layer in layers:
        if layer is None:
            continue
        data = _plain(layer)
        if not isinstance(data, dict):
            raise ConfigError(f"{name}: simulation config must be a table")
        _merge(merged, data, name, origin)
    try:
        return SimulationConfig.model_validate(merged)
    except ValidationError as exc:
        issues = []
        for err in exc.errors():
            tag = _layer_of(tuple(str(p) for p in err["loc"]), origin)
            code, message = _issue(err)
            path = format_path(("simulation", *err["loc"]))
            issues.append(ValidationIssue(f"{tag}: {path}", message, code))
        raise ConfigError("invalid simulation config", issues) from None
