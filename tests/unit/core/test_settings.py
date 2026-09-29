from __future__ import annotations

from pathlib import Path

import pytest

from urbanflow.core.errors import ConfigError
from urbanflow.core.settings import AppSettings, load_settings


def test_defaults(workspace: Path) -> None:
    s = load_settings()
    assert s.workspace == workspace.resolve()
    assert s.host == "127.0.0.1" and s.port == 8000
    assert s.max_sessions == 4 and s.max_subscribers == 8
    assert s.session_idle_timeout_s == 900 and s.max_body_mb == 20 and s.max_upload_mb == 1024
    assert 1 <= s.max_workers <= 61
    assert s.resolved_database_url.startswith("sqlite:///")
    assert s.resolved_database_url.endswith(".urbanflow/urbanflow.db")
    assert s.is_loopback
    assert s.project_name == "ws"


def test_precedence_file_env_kwargs(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = (
        "[server]\n"
        "port = 9001\n"
        'host = "0.0.0.0"\n'
        "max_sessions = 2\n"
        "[logging]\n"
        'level = "debug"\n'
        'format = "json"\n'
    )
    (workspace / "urbanflow.toml").write_text(toml, encoding="utf-8")
    s = load_settings()
    assert (s.port, s.host, s.max_sessions, s.log_level, s.log_format) == (
        9001,
        "0.0.0.0",
        2,
        "DEBUG",
        "json",
    )
    assert not s.is_loopback

    monkeypatch.setenv("URBANFLOW_PORT", "9002")
    assert load_settings().port == 9002  # env beats file
    assert load_settings(port=9003).port == 9003  # kwargs beat env
    assert load_settings(port=None).port == 9002  # None overrides are ignored


def test_explicit_config_file(workspace: Path) -> None:
    cfg = workspace / "custom.toml"
    cfg.write_text("[server]\nport = 7777\n", encoding="utf-8")
    assert load_settings(config_file=cfg).port == 7777


def test_config_file_from_the_environment(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AB 6.2: URBANFLOW_CONFIG names the settings file."""
    cfg = workspace / "env.toml"
    cfg.write_text("[server]\nport = 7788\n", encoding="utf-8")
    monkeypatch.setenv("URBANFLOW_CONFIG", str(cfg))
    settings = load_settings()
    assert settings.port == 7788
    assert settings.config_file == cfg and settings.resolved_config_file == cfg
    assert AppSettings(config_file=workspace / "x.toml").config_file == workspace / "x.toml"


def test_invalid_values_raise_config_error(workspace: Path) -> None:
    with pytest.raises(ConfigError, match="port"):
        load_settings(port=70000)
    (workspace / "urbanflow.toml").write_text("[server\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_settings()


def test_secret_token_not_in_repr(workspace: Path) -> None:
    s = load_settings(api_token="topsecret")
    assert "topsecret" not in repr(s)
    assert s.api_token is not None and s.api_token.get_secret_value() == "topsecret"


def test_ensure_dirs(workspace: Path) -> None:
    s = load_settings()
    s.ensure_dirs()
    for sub in (".urbanflow", "scenarios", "runs", "replays", "models"):
        assert (workspace / sub).is_dir()


def test_env_does_not_leak_between_calls(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("URBANFLOW_LOG_LEVEL", "error")
    assert load_settings().log_level == "ERROR"
    monkeypatch.delenv("URBANFLOW_LOG_LEVEL")
    assert AppSettings().log_level == "INFO"
