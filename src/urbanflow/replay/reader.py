"""Reading replays safely (plan K.5): the same path for trusted and untrusted files.

:class:`ReplayReader` opens a packed ``.ufr`` ZIP or an unpacked ``.ufr.d`` directory (a
crashed recording), validates member names and sizes before reading anything, and gives
random access to frames by step with a small chunk cache.
"""

from __future__ import annotations

import bisect
import json
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from urbanflow.core.errors import NotFoundError, ReplayFormatError
from urbanflow.replay.recorder import FORMAT
from urbanflow.visualization import Frame, decode_frame

__all__ = ["ReplayReader"]

MAX_MEMBERS: Final = 100_000
MAX_JSON_BYTES: Final = 256 * 1024 * 1024
MAX_CHUNK_BYTES: Final = 512 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 2 * 1024**3
_SUPPORTED_MAJOR: Final = 1


def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return not (p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name)


class ReplayReader:
    """Validated, random-access view of a replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise NotFoundError(f"replay not found: {self.path}")
        self._zip: zipfile.ZipFile | None = None
        if self.path.is_dir():
            self._sizes = {
                f.relative_to(self.path).as_posix(): f.stat().st_size
                for f in self.path.rglob("*")
                if f.is_file()
            }
        else:
            try:
                self._zip = zipfile.ZipFile(self.path)
            except (zipfile.BadZipFile, OSError) as exc:
                raise ReplayFormatError(
                    f"{self.path.name} is not a replay archive: {exc}"
                ) from None
            infos = self._zip.infolist()
            if len(infos) > MAX_MEMBERS:
                raise ReplayFormatError("replay archive has too many members")
            self._sizes = {i.filename: i.file_size for i in infos}
        bad = [n for n in self._sizes if not _safe_name(n)]
        if bad:
            raise ReplayFormatError(f"unsafe member name in replay: {bad[0]!r}")
        if sum(self._sizes.values()) > MAX_TOTAL_BYTES:
            raise ReplayFormatError("replay archive is too large once decompressed")
        self.manifest: dict[str, Any] = self._json("manifest.json")
        if self.manifest.get("format") != FORMAT:
            raise ReplayFormatError(
                f"not an UrbanFlow replay (format {self.manifest.get('format')!r})"
            )
        major = str(self.manifest.get("version", "0")).split(".")[0]
        if major != str(_SUPPORTED_MAJOR):
            raise ReplayFormatError(
                f"replay format {self.manifest.get('version')} is not supported (this reader: 1.x)"
            )
        self._chunks: list[dict[str, Any]] = list(self.manifest.get("chunks", []))
        self.steps: list[int] = [int(s) for c in self._chunks for s in c["steps"]]
        if self.steps != sorted(self.steps):
            raise ReplayFormatError("replay frames are not in step order")
        self._chunk_of = [i for i, c in enumerate(self._chunks) for _ in c["steps"]]
        self._pos_in = [k for c in self._chunks for k in range(len(c["steps"]))]
        for c in self._chunks:
            name, offsets = c["name"], c["offsets"]
            if name not in self._sizes or not _safe_name(name):
                raise ReplayFormatError(f"replay chunk {name!r} is missing")
            if len(offsets) != len(c["steps"]) + 1 or offsets[-1] != self._sizes[name]:
                raise ReplayFormatError(f"replay chunk {name!r} has inconsistent offsets")
            if self._sizes[name] > MAX_CHUNK_BYTES:
                raise ReplayFormatError(f"replay chunk {name!r} is too large")
        self._load_chunk = lru_cache(maxsize=4)(self._read_chunk)

    # ------------------------------------------------------------------ members
    def _read(self, name: str, limit: int) -> bytes:
        size = self._sizes.get(name)
        if size is None:
            raise ReplayFormatError(f"replay member {name!r} is missing")
        if size > limit:
            raise ReplayFormatError(f"replay member {name!r} is too large")
        if self._zip is not None:
            try:
                data = self._zip.read(name)
            except (zipfile.BadZipFile, OSError, RuntimeError, ValueError) as exc:
                raise ReplayFormatError(f"cannot read {name!r}: {exc}") from None
        else:
            data = (self.path / name).read_bytes()
        if len(data) != size:
            raise ReplayFormatError(f"replay member {name!r} is truncated")
        return data

    def _json(self, name: str) -> Any:
        try:
            return json.loads(self._read(name, MAX_JSON_BYTES))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ReplayFormatError(f"replay member {name!r} is not valid JSON: {exc}") from None

    def _read_chunk(self, index: int) -> bytes:
        return self._read(self._chunks[index]["name"], MAX_CHUNK_BYTES)

    def close(self) -> None:
        """Close the archive."""
        if self._zip is not None:
            self._zip.close()

    # ------------------------------------------------------------------ content
    @property
    def n_frames(self) -> int:
        return len(self.steps)

    def index_at(self, step: int) -> int:
        """Index of the frame shown at ``step`` (nearest recorded step <= ``step``)."""
        i = bisect.bisect_right(self.steps, step) - 1
        return min(max(i, 0), len(self.steps) - 1)

    def frame_bytes(self, index: int) -> bytes:
        """The raw UFB1 message of frame ``index``."""
        if not 0 <= index < len(self.steps):
            raise IndexError(index)
        chunk = self._chunks[self._chunk_of[index]]
        k = self._pos_in[index]
        data = self._load_chunk(self._chunk_of[index])
        return data[chunk["offsets"][k] : chunk["offsets"][k + 1]]

    def frame(self, step: int) -> Frame:
        """The decoded frame shown at ``step``."""
        return decode_frame(self.frame_bytes(self.index_at(step)))[0]

    def vehicles(self) -> dict[int, tuple[str, int, int]]:
        """uid -> (vehicle id, type index, destination road index)."""
        v = self._json("vehicles.json")
        return {
            int(u): (str(i), int(t), int(d))
            for u, i, t, d in zip(v["uids"], v["ids"], v["types"], v["dests"], strict=True)
        }

    def geometry_json(self) -> bytes:
        """The recorded render geometry (JSON bytes)."""
        return self._read("geometry.json", MAX_JSON_BYTES)

    def summary(self) -> dict[str, Any] | None:
        """The run result written at close (None for a crashed recording)."""
        return self._json("summary.json") if "summary.json" in self._sizes else None

    def timeseries(self) -> dict[str, np.ndarray] | None:
        """The global metrics timeseries written at close (None if absent)."""
        if "timeseries.csv" not in self._sizes:
            return None
        import csv
        import io

        text = self._read("timeseries.csv", MAX_JSON_BYTES).decode("utf-8")
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) < 2:
            return None
        cols = list(zip(*rows[1:], strict=True))
        return {
            name: np.array([float(v) if v else np.nan for v in col], dtype=np.float64)
            for name, col in zip(rows[0], cols, strict=True)
        }

    def scenario_json(self) -> str:
        return self._read("scenario.json", MAX_JSON_BYTES).decode("utf-8")
