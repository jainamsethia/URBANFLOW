"""UFB1, the binary frame format (plan K.10.2, B.2 #2).

Little-endian; a 64-byte header (``struct`` format ``<4sHHBBHIqdIIIII12x``) followed by
4-byte-aligned columns so a browser can view them with typed arrays without copying::

    uid u32[n] | xy f32[2n] interleaved | heading f32[n] | speed f32[n] | link u32[n]
    | flags u8[n] | pad | signals u8[m] (flag bit 0) | pad | lane_values f32[L] (bit 1)
"""

from __future__ import annotations

import struct
from typing import Final

import numpy as np

from urbanflow.core.errors import ReplayFormatError
from urbanflow.visualization.frames import Frame, LaneMetric

__all__ = ["HEADER", "MAGIC", "VERSION", "decode_frame", "encode_frame"]

MAGIC: Final = b"UFB1"
VERSION: Final = 1
HEADER: Final = struct.Struct("<4sHHBBHIqdIIIII12x")
KIND_VEHICLES: Final = 1
HAS_SIGNALS: Final = 1 << 0
HAS_LANE_VALUES: Final = 1 << 1
SOURCE_REPLAY: Final = 1 << 2


def _align4(offset: int) -> int:
    return (offset + 3) & ~3


def encode_frame(
    frame: Frame, seq: int = 0, geometry_crc: int = 0, *, replay: bool = False
) -> bytes:
    """Serialise ``frame`` as one UFB1 message."""
    n = frame.n
    signals = frame.signals
    lanes = frame.lane_values
    m = 0 if signals is None else int(signals.size)
    nl = 0 if lanes is None else int(lanes.size)
    flags = (
        (HAS_SIGNALS if signals is not None else 0)
        | (HAS_LANE_VALUES if lanes is not None else 0)
        | (SOURCE_REPLAY if replay else 0)
    )
    body = [
        frame.uid.astype("<u4").tobytes(),
        frame.xy.astype("<f4").reshape(-1).tobytes(),
        frame.heading.astype("<f4").tobytes(),
        frame.speed.astype("<f4").tobytes(),
        frame.link.astype("<u4").tobytes(),
        frame.flags.astype("u1").tobytes(),
    ]
    size = HEADER.size + 25 * n
    pad = _align4(size) - size
    body.append(b"\0" * pad)
    size += pad
    if signals is not None:
        body.append(signals.astype("u1").tobytes())
        size += m
        pad = _align4(size) - size
        body.append(b"\0" * pad)
        size += pad
    if lanes is not None:
        body.append(lanes.astype("<f4").tobytes())
        size += 4 * nl
    header = HEADER.pack(
        MAGIC, VERSION, HEADER.size, KIND_VEHICLES, flags, int(frame.lane_metric),
        seq & 0xFFFFFFFF, frame.step, frame.time, n, m, nl, geometry_crc & 0xFFFFFFFF, size,
    )  # fmt: skip
    return b"".join([header, *body])


def decode_frame(buf: bytes) -> tuple[Frame, int, int]:
    """Parse a UFB1 message: ``(frame, seq, geometry_crc)``; ``ReplayFormatError`` if bad."""
    if len(buf) < HEADER.size:
        raise ReplayFormatError("UFB1 frame shorter than its header")
    (magic, version, hsize, _kind, flags, metric, seq, step, time, n, m, nl, crc, total) = (
        HEADER.unpack_from(buf)
    )
    if magic != MAGIC or version != VERSION:
        raise ReplayFormatError(f"not a UFB1 v{VERSION} frame")
    if total != len(buf) or hsize < HEADER.size:
        raise ReplayFormatError("UFB1 frame length does not match its header")
    o = hsize

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal o
        arr = np.frombuffer(buf, dtype=dtype, count=count, offset=o).copy()
        o += arr.nbytes
        return arr

    uid = take("<u4", n)
    xy = take("<f4", 2 * n).reshape(n, 2)
    heading, speed = take("<f4", n), take("<f4", n)
    link, fl = take("<u4", n), take("u1", n)
    o = _align4(o)
    signals = None
    if flags & HAS_SIGNALS:
        signals = take("u1", m)
        o = _align4(o)
    lanes = take("<f4", nl) if flags & HAS_LANE_VALUES else None
    frame = Frame(
        step=step, time=time, uid=uid, xy=xy, heading=heading, speed=speed, link=link,
        flags=fl, signals=signals, lane_values=lanes, lane_metric=LaneMetric(metric),
    )  # fmt: skip
    return frame, seq, crc
