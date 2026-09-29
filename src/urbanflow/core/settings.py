"""Application settings (server, CLI, logging) loaded from defaults, TOML, env and overrides.

Precedence (lowest → highest): defaults < ``urbanflow.toml`` (``[server]``, ``[logging]``)
< ``URBANFLOW_*`` environment variables < explicit overrides (CLI flags / kwargs).

Environment variables only ever feed :class:`AppSettings`; they never change a
``SimulationConfig`` (plan B.2 #8), so simulation results have no hidden inputs.
"""

from __future__ import annotations

import os
import tomllib
from contextvars import ContextVar
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from urbanflow.core.errors import ConfigError

__all__ = ["AppSettings", "load_settings"]

ENV_PREFIX = "URBANFLOW_"
CONFIG_FILENAME = "urbanflow.toml"

# The TOML file layer is resolved per load_settings() call (workspace-dependent).
_file_layer: ContextVar[dict[str, Any] | None] = ContextVar(
    "urbanflow_settings_file_layer", default=None
)

# `[logging]` keys map onto differently named fields.
_LOGGING_KEYS = {"level": "log_level", "format": "log_format"}


def _default_max_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 61))  # Windows caps process pools at 61


class _TomlLayer(PydanticBaseSettingsSource):
    """Settings source that serves the TOML values computed by :func:`load_settings`."""

    def get_field_value(
        self,
        field: FieldInfo,  # noqa: ARG002 - signature fixed by pydantic-settings
        field_name: str,
    ) -> tuple[Any, str, bool]:
        values = _file_layer.get() or {}
        return values.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(_file_layer.get() or {})


class AppSettings(BaseSettings):
    """Settings for the CLI, the web server and worker processes."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        extra="ignore",
        validate_default=True,
        validate_by_name=True,  # kwargs use field names; aliases name env variables
    )

    workspace: Path = Field(default_factory=Path.cwd)
    """Workspace directory holding scenarios, runs, replays, models and state."""
    config_file: Path | None = Field(default=None, validation_alias=f"{ENV_PREFIX}CONFIG")
    """Settings file (env ``URBANFLOW_CONFIG``); defaults to ``<workspace>/urbanflow.toml``."""
    database_url: str | None = None
    """SQLAlchemy URL; defaults to SQLite at ``<workspace>/.urbanflow/urbanflow.db``."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_token: SecretStr | None = None
    """Bearer token; required when ``host`` is not a loopback address."""
    max_sessions: int = Field(default=4, ge=1, le=64)
    max_subscribers: int = Field(default=8, ge=1, le=64)
    session_idle_timeout_s: float = Field(default=900.0, gt=0)
    max_workers: int = Field(default_factory=_default_max_workers, ge=1, le=61)
    max_body_mb: float = Field(default=20.0, gt=0)
    max_upload_mb: float = Field(default=1024.0, gt=0)
    trusted_hosts: list[str] = Field(default_factory=list)
    cors_origins: list[str] = Field(default_factory=list)
    frontend_dir: Path | None = None
    allow_model_loading: bool = False
    """Allow loading SB3 model zips (which can contain pickle) in server experiments."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - required signature
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first: explicit kwargs, then env, then urbanflow.toml.
        return (init_settings, env_settings, _TomlLayer(settings_cls))

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("workspace", mode="after")
    @classmethod
    def _resolve_workspace(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    # ------------------------------------------------------------------ derived
    @property
    def project_name(self) -> str:
        """Human-readable project name shown in the web UI (the workspace directory name)."""
        return self.workspace.name or str(self.workspace)

    @property
    def state_dir(self) -> Path:
        return self.workspace / ".urbanflow"

    @property
    def scenarios_dir(self) -> Path:
        return self.workspace / "scenarios"

    @property
    def runs_dir(self) -> Path:
        return self.workspace / "runs"

    @property
    def replays_dir(self) -> Path:
        return self.workspace / "replays"

    @property
    def models_dir(self) -> Path:
        return self.workspace / "models"

    @property
    def resolved_config_file(self) -> Path:
        return self.config_file or (self.workspace / CONFIG_FILENAME)

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.state_dir / 'urbanflow.db').as_posix()}"

    @property
    def is_loopback(self) -> bool:
        if self.host in {"localhost", "localhost.localdomain"}:
            return True
        try:
            return ip_address(self.host).is_loopback
        except ValueError:
            return False

    def ensure_dirs(self) -> None:
        """Create the workspace sub-directories the server and CLI write to."""
        for directory in (
            self.state_dir,
            self.scenarios_dir,
            self.runs_dir,
            self.replays_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _read_file_layer(path: Path) -> dict[str, Any]:
    """Flatten ``[server]`` and ``[logging]`` tables of a settings TOML file."""
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    values: dict[str, Any] = {}
    server = data.get("server", {})
    if not isinstance(server, dict):
        raise ConfigError(f"{path}: [server] must be a table")
    values.update(server)
    logging_table = data.get("logging", {})
    if not isinstance(logging_table, dict):
        raise ConfigError(f"{path}: [logging] must be a table")
    for key, value in logging_table.items():
        values[_LOGGING_KEYS.get(str(key), str(key))] = value
    return values


def load_settings(**overrides: Any) -> AppSettings:
    """Build :class:`AppSettings` from defaults, ``urbanflow.toml``, env and ``overrides``.

    ``None`` overrides are ignored so CLI options that were not given do not mask lower
    layers. Invalid values raise :class:`~urbanflow.core.errors.ConfigError`.
    """
    from pydantic import ValidationError

    given = {k: v for k, v in overrides.items() if v is not None}
    workspace = Path(
        given.get("workspace") or os.environ.get(f"{ENV_PREFIX}WORKSPACE") or Path.cwd()
    )
    config_file = given.get("config_file") or os.environ.get(f"{ENV_PREFIX}CONFIG")
    path = Path(config_file) if config_file else workspace.expanduser() / CONFIG_FILENAME
    token = _file_layer.set(_read_file_layer(path))
    try:
        return AppSettings(**given)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'settings'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"invalid settings: {details}") from exc
    finally:
        _file_layer.reset(token)
