"""Replay recording (plan K, simplified container).

A recording is first a directory ``<name>.ufr.d/`` (crash-safe: the manifest is rewritten
atomically after every chunk flush) and is packed into a ZIP ``<name>.ufr`` by
:meth:`ReplayRecorder.close`::

    manifest.json        format, versions, scenario identity, config, chunk table
    scenario.json        the exact scenario
    geometry.json        render geometry (frames' xy are in its local coordinates)
    vehicles.json        uid -> id / type index / destination road of every vehicle seen
    chunks/NNNNNN.bin    concatenated UFB1 frames (<= max_chunk_steps frames each)
    timeseries.csv ...   metrics tables and summary.json (written at close)

ponytail: frames are raw UFB1 messages (deflated in the ZIP), not the K.3 columnar npz
chunks, and they carry no lane values; quantised columns and lane samples are the upgrade.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Final

from urbanflow._version import __version__
from urbanflow.core.constants import SHORT_HASH_LENGTH
from urbanflow.metrics.export import export_tables
from urbanflow.scenario.io import write_json
from urbanflow.visualization import encode_frame, render_geometry

__all__ = ["FORMAT", "MAX_CHUNK_STEPS", "VERSION", "ReplayRecorder"]

FORMAT: Final = "urbanflow.replay"
VERSION: Final = "1.0"
MAX_CHUNK_STEPS: Final = 300
_COMPRESS_LEVEL: Final = 1


class ReplayRecorder:
    """Records every ``every``-th step of a :class:`~urbanflow.Simulation` to ``path``."""

    def __init__(self, sim: Any, path: str | os.PathLike[str], *, every: int = 1) -> None:
        # sim: an urbanflow.Simulation (typed loosely: replay must not import the facade)
        target = Path(path)
        if target.suffix != ".ufr":
            target = target.with_suffix(".ufr")
        self.path = target
        self.dir = target.with_name(target.name + ".d")
        if self.dir.exists():
            shutil.rmtree(self.dir)
        (self.dir / "chunks").mkdir(parents=True)
        self._sim = sim
        self._every = max(1, int(every))
        self._buf: list[bytes] = []
        self._steps: list[int] = []
        self._chunks: list[dict[str, Any]] = []
        self._seq = 0
        self._vehicles: dict[int, tuple[str, int, int]] = {}
        self._closed = False
        scenario = sim.scenario
        net = sim.network.compiled
        self._crc = int(net.geometry_crc)
        (self.dir / "scenario.json").write_text(scenario.to_json(), encoding="utf-8")
        geometry = render_geometry(net).model_dump_json()
        (self.dir / "geometry.json").write_text(geometry, encoding="utf-8")
        self._manifest: dict[str, Any] = {
            "format": FORMAT,
            "version": VERSION,
            "complete": False,
            "urbanflow_version": __version__,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scenario": {"name": scenario.name, "hash": scenario.content_hash},
            "short_hash": scenario.content_hash[:SHORT_HASH_LENGTH],
            "config": sim.config.model_dump(mode="json"),
            "seed": sim.seed,
            "dt": sim.dt,
            "record_every": self._every,
            "geometry_crc": self._crc,
            "n_links": net.n_links,
            "n_movements": len(sim.network.movement_ids),
            "controllers": {j: str(sim.signals[j].controller) for j in sim.signals.ids},
            "chunks": self._chunks,
        }
        self.record()  # the initial state

    @property
    def n_frames(self) -> int:
        """Frames recorded so far."""
        return sum(len(c["steps"]) for c in self._chunks) + len(self._steps)

    def record(self) -> None:
        """Record the current state if it falls on the recording cadence."""
        if self._closed:
            return
        sim = self._sim
        step = sim.step_count
        if step % self._every:
            return
        frame = sim.frame()
        new = [int(u) for u in frame.uid.tolist() if int(u) not in self._vehicles]
        if new:
            ids, types, dests = sim.vehicle_meta(new)
            for u, i, t, d in zip(new, ids, types, dests, strict=True):
                self._vehicles[u] = (i, t, d)
        self._seq += 1
        self._buf.append(encode_frame(frame, self._seq, self._crc, replay=True))
        self._steps.append(step)
        if len(self._steps) >= MAX_CHUNK_STEPS:
            self._flush()

    def _flush(self) -> None:
        if not self._steps:
            return
        name = f"chunks/{len(self._chunks):06d}.bin"
        offsets = [0]
        for b in self._buf:
            offsets.append(offsets[-1] + len(b))
        tmp = self.dir / (name + ".tmp")
        tmp.write_bytes(b"".join(self._buf))
        tmp.replace(self.dir / name)
        self._chunks.append({"name": name, "steps": self._steps, "offsets": offsets})
        self._buf, self._steps = [], []
        self._write_meta()

    def _write_meta(self) -> None:
        uids = sorted(self._vehicles)
        write_json(
            self.dir / "vehicles.json",
            {
                "uids": uids,
                "ids": [self._vehicles[u][0] for u in uids],
                "types": [self._vehicles[u][1] for u in uids],
                "dests": [self._vehicles[u][2] for u in uids],
            },
        )
        steps = [s for c in self._chunks for s in c["steps"]]
        self._manifest.update(
            n_frames=len(steps),
            first_step=steps[0] if steps else None,
            last_step=steps[-1] if steps else None,
            n_vehicles=len(uids),
        )
        write_json(self.dir / "manifest.json", self._manifest)

    def close(self) -> Path:
        """Flush, add metrics tables and summary, pack the ZIP; returns its path."""
        if self._closed:
            return self.path
        self._flush()
        result = self._sim.get_results()
        export_tables(result.tables, self.dir, "csv")
        write_json(self.dir / "summary.json", result.to_dict())
        self._manifest["complete"] = True
        self._write_meta()
        self._closed = True
        tmp = self.path.with_name(self.path.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=_COMPRESS_LEVEL) as zf:
            for f in sorted(self.dir.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(self.dir).as_posix())
        tmp.replace(self.path)
        shutil.rmtree(self.dir, ignore_errors=True)
        return self.path

    def discard(self) -> None:
        """Stop and delete the unfinished recording."""
        self._closed = True
        shutil.rmtree(self.dir, ignore_errors=True)

    def manifest(self) -> dict[str, Any]:
        """A copy of the current manifest."""
        copied: dict[str, Any] = json.loads(json.dumps(self._manifest))
        return copied
