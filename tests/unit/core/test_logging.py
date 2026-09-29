from __future__ import annotations

import io
import json
import logging

import pytest
from pydantic import SecretStr

from urbanflow.core.logging import JsonFormatter, configure_logging, log_duration


def test_library_is_silent_without_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    logging.getLogger("urbanflow.test").warning("should not appear")
    captured = capsys.readouterr()
    assert "should not appear" not in captured.out + captured.err


def test_json_formatter_emits_valid_json_with_extras() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", "json", stream=stream)
    log = logging.getLogger("urbanflow.simulation")
    log.info(
        "run finished", extra={"sim_time": 1800.0, "arrived": 2180, "token": SecretStr("s3cret")}
    )
    record = json.loads(stream.getvalue().strip())
    assert record["level"] == "INFO"
    assert record["logger"] == "urbanflow.simulation"
    assert record["msg"] == "run finished"
    assert record["sim_time"] == 1800.0 and record["arrived"] == 2180
    assert record["token"] == "**********"
    assert record["ts"].endswith("Z")


def test_json_formatter_includes_exceptions() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = logging.LogRecord(
            "urbanflow", logging.ERROR, __file__, 1, "failed", None, sys.exc_info()
        )
    payload = json.loads(formatter.format(rec))
    assert "ValueError: boom" in payload["exc"]


def test_configure_logging_is_idempotent() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "text", stream=stream)
    configure_logging("INFO", "text", stream=stream)
    logger = logging.getLogger("urbanflow")
    tagged = [h for h in logger.handlers if getattr(h, "_urbanflow_handler", False)]
    assert len(tagged) == 1
    logging.getLogger("urbanflow.x").info("once")
    assert stream.getvalue().count("once") == 1


def test_configure_logging_rejects_bad_format() -> None:
    with pytest.raises(ValueError, match="text"):
        configure_logging("INFO", "yaml")  # type: ignore[arg-type]


def test_log_duration_adds_duration_ms() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream=stream)
    with log_duration(logging.getLogger("urbanflow.perf"), "compiled", lanes=12) as fields:
        fields["connectors"] = 30
    record = json.loads(stream.getvalue().strip())
    assert record["lanes"] == 12 and record["connectors"] == 30
    assert record["duration_ms"] >= 0


def test_log_duration_is_silent_when_the_block_fails() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream=stream)
    with pytest.raises(RuntimeError), log_duration(logging.getLogger("urbanflow.x"), "done"):
        raise RuntimeError("boom")
    assert stream.getvalue() == ""  # a failed operation is not reported as done
