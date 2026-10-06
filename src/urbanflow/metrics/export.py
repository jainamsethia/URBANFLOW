"""Table export and import: CSV (stdlib), JSON (stdlib), Parquet (polars, extra ``data``).

CSV writes floats with ``repr`` (exact round trip) and NaN as an empty field; JSON tables
are ``{"format": "urbanflow.metrics.table", "version": "1.0", "columns": [...], "data":
{...}}`` with NaN as null (J.8).
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

import numpy as np

from urbanflow.core.errors import ConfigError, require

__all__ = ["TABLES", "export_tables", "read_table", "write_table"]

TABLES: Final = ("timeseries", "intersections", "trips")
Format = Literal["csv", "json", "parquet"]
_TABLE_FORMAT: Final = "urbanflow.metrics.table"


def _cell(value: object) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    return str(value)


def _json_value(value: object) -> object:
    return None if isinstance(value, float) and math.isnan(value) else value


def write_table(table: Mapping[str, np.ndarray], path: str | os.PathLike[str]) -> Path:
    """Write one table; the format comes from the suffix (.csv, .json, .parquet)."""
    target = Path(path)
    fmt = target.suffix.lower().lstrip(".")
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = list(table)
    if fmt == "csv":
        with target.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for row in zip(*(table[c].tolist() for c in columns), strict=True):
                writer.writerow([_cell(v) for v in row])
    elif fmt == "json":
        doc = {
            "format": _TABLE_FORMAT,
            "version": "1.0",
            "columns": [{"name": c, "dtype": table[c].dtype.str} for c in columns],
            "data": {c: [_json_value(v) for v in table[c].tolist()] for c in columns},
        }
        target.write_text(json.dumps(doc, allow_nan=False), encoding="utf-8")
    elif fmt == "parquet":
        pl = require("polars", "data")
        pl.DataFrame({c: table[c] for c in columns}).write_parquet(target)
    else:
        raise ConfigError(
            f"unsupported table format {target.suffix!r} (use .csv, .json or .parquet)"
        )
    return target


def export_tables(
    tables: Mapping[str, Mapping[str, np.ndarray]],
    directory: str | os.PathLike[str],
    fmt: Format = "csv",
) -> list[Path]:
    """Write every table as ``<directory>/<name>.<fmt>``; returns the paths."""
    return [write_table(t, Path(directory) / f"{name}.{fmt}") for name, t in tables.items()]


def read_table(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Read a table written by :func:`write_table` (CSV floats parse as float64)."""
    source = Path(path)
    fmt = source.suffix.lower().lstrip(".")
    if fmt == "json":
        doc = json.loads(source.read_text(encoding="utf-8"))
        if doc.get("format") != _TABLE_FORMAT:
            raise ConfigError(f"{source}: not an UrbanFlow metrics table")
        out: dict[str, np.ndarray] = {}
        for col in doc["columns"]:
            values = [math.nan if v is None else v for v in doc["data"][col["name"]]]
            out[col["name"]] = np.array(values, dtype=np.dtype(col["dtype"]))
        return out
    if fmt == "parquet":
        pl = require("polars", "data")
        frame = pl.read_parquet(source)
        return {c: frame[c].to_numpy() for c in frame.columns}
    if fmt == "csv":
        with source.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        out = {}
        for i, name in enumerate(header):
            raw = [r[i] for r in body]
            try:
                out[name] = np.array([float(v) if v else math.nan for v in raw])
            except ValueError:
                out[name] = np.array(raw)
        return out
    raise ConfigError(f"unsupported table format {source.suffix!r}")
