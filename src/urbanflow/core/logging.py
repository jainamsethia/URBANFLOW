"""Structured logging for UrbanFlow.

The library never configures handlers on import (it only installs a ``NullHandler``);
applications - the CLI, the server and worker processes - call :func:`configure_logging`.
Modules log with ``logging.getLogger(__name__)`` and pass structured fields via ``extra=``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import IO, Any, Literal

__all__ = ["LOGGER_NAME", "JsonFormatter", "configure_logging", "log_duration"]

LOGGER_NAME = "urbanflow"
LogFormat = Literal["text", "json"]
_HANDLER_TAG = "_urbanflow_handler"
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Attributes present on every LogRecord; anything else came from ``extra=``.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None)).keys()
    | {"message", "asctime", "taskName"}
)


def _is_secret(value: object) -> bool:
    return type(value).__name__ in {"SecretStr", "SecretBytes"}


def _jsonable(value: object) -> object:
    if _is_secret(value):
        return "**********"
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(v) for v in value]
    return str(value)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ``ts, level, logger, msg``, extras, and ``exc`` if any."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, UTC)
        payload: dict[str, Any] = {
            "ts": ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = _jsonable(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)


def _text_handler(stream: IO[str] | None) -> logging.Handler:
    target = stream if stream is not None else sys.stderr
    if stream is None and target.isatty():
        from rich.logging import RichHandler  # lazy: keep plain imports light

        return RichHandler(show_path=False, rich_tracebacks=False, markup=False)
    handler = logging.StreamHandler(target)
    handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    return handler


def configure_logging(
    level: str | int = "INFO",
    fmt: LogFormat = "text",
    *,
    stream: IO[str] | None = None,
) -> None:
    """Attach exactly one handler to the ``urbanflow`` logger (idempotent).

    Calling it again replaces the previous UrbanFlow handler, so reconfiguration under
    reloaders or repeated CLI invocations in one process never duplicates output.
    """
    if fmt not in ("text", "json"):
        raise ValueError(f"log format must be 'text' or 'json', got {fmt!r}")
    logger = logging.getLogger(LOGGER_NAME)
    for existing in list(logger.handlers):
        if getattr(existing, _HANDLER_TAG, False):
            logger.removeHandler(existing)
    if fmt == "json":
        handler: logging.Handler = logging.StreamHandler(
            stream if stream is not None else sys.stderr
        )
        handler.setFormatter(JsonFormatter())
    else:
        handler = _text_handler(stream)
    setattr(handler, _HANDLER_TAG, True)
    logger.addHandler(handler)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    logger.propagate = False


@contextmanager
def log_duration(
    logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any
) -> Iterator[dict[str, Any]]:
    """Log ``msg`` with ``duration_ms`` (and ``fields``) when the block completes.

    The yielded dict can be updated inside the block to add more fields. If the block
    raises, nothing is logged: a failed operation is not reported as done.
    """
    extra: dict[str, Any] = dict(fields)
    start = time.perf_counter()
    yield extra
    extra["duration_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
    logger.log(level, msg, extra=extra)
