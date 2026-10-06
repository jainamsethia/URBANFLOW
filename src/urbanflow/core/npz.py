"""Safe ``.npz`` writing and reading without pickle (plan K.3, K.5, U.4).

:func:`write_npz` writes a standard npz (one ``<name>.npy`` member per array, deflated),
readable with any numpy. :func:`read_npz` is the only reader: ``np.load`` is never used,
so there is no pickle path at all. It parses every npy header itself (``read_magic`` and
``read_array_header_{1_0,2_0}`` with a bounded header size), accepts exactly the expected
members and dtypes, bounds the element count and the total declared bytes before reading
(a decompression bomb is rejected from the zip directory alone, and zipfile never inflates
a member past its declared size), checks that each member's declared size is exactly its
header plus the data the header describes, and reads the data in chunks straight into the
result. Every malformed input raises ``ValueError``. The same path serves trusted files
(snapshots) and untrusted ones (replays uploaded to the server).
"""

from __future__ import annotations

import io
import math
import zipfile
import zlib
from collections.abc import Mapping
from typing import IO, Any, Final

import numpy as np
from numpy.typing import DTypeLike, NDArray

from urbanflow.core.constants import NPZ_MAX_BYTES

__all__ = ["MAX_HEADER_SIZE", "ZIP_ERRORS", "read_npz", "write_npz"]

MAX_HEADER_SIZE: Final = 10_000
"""Largest npy header accepted, bytes (numpy's own default limit)."""

ZIP_ERRORS: Final = (zipfile.BadZipFile, OSError, EOFError, zlib.error, RuntimeError)
"""What zipfile raises on corrupt input: bad structure or CRC, truncation, corrupt deflate
data, encrypted members (RuntimeError) and unsupported methods (NotImplementedError, a
RuntimeError)."""

_SUFFIX: Final = ".npy"
_CHUNK: Final = 1 << 20  # bytes per read of member data


def write_npz(fileobj: IO[bytes], arrays: Mapping[str, NDArray[Any]], level: int) -> None:
    """Write ``arrays`` as an npz to ``fileobj`` (``ZIP_DEFLATED`` at ``level``, 0-9).

    Members are ``<name>.npy`` in mapping order. Object arrays raise ``ValueError`` (no
    pickle).
    """
    with zipfile.ZipFile(fileobj, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        for name, arr in arrays.items():
            with zf.open(name + _SUFFIX, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(arr), allow_pickle=False)


def read_npz(
    buf: bytes | IO[bytes],
    expected: Mapping[str, DTypeLike],
    max_elems: int,
    *,
    max_bytes: int = NPZ_MAX_BYTES,
) -> dict[str, NDArray[Any]]:
    """The arrays of the npz in ``buf``, exactly the members named in ``expected``.

    Every member must be ``<name>.npy`` for a name in ``expected`` (no others, no
    duplicates, none missing) with exactly the expected dtype; object and void dtypes are
    rejected, as are headers over :data:`MAX_HEADER_SIZE`, arrays of more than
    ``max_elems`` elements, archives whose members declare more than ``max_bytes`` bytes
    in total, and members whose declared size is not their header plus the data the header
    describes. Any violation, and any corrupt zip or deflate data, raises ``ValueError``
    naming the member.
    """
    source = io.BytesIO(buf) if isinstance(buf, bytes | bytearray) else buf
    try:
        zf = zipfile.ZipFile(source)
    except (*ZIP_ERRORS, ValueError) as exc:
        raise ValueError(f"not an npz archive: {exc}") from None
    with zf:
        names = zf.namelist()
        if len(set(names)) != len(names):
            raise ValueError("npz archive has duplicate members")
        wanted = {name + _SUFFIX: name for name in expected}
        unexpected = sorted(set(names) - set(wanted))
        if unexpected:
            raise ValueError(f"unexpected npz members: {', '.join(unexpected)}")
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise ValueError(f"missing npz members: {', '.join(missing)}")
        declared = sum(zf.getinfo(member).file_size for member in wanted)
        if declared > max_bytes:
            raise ValueError(
                f"npz members declare {declared} bytes, more than the limit {max_bytes}"
            )
        out: dict[str, NDArray[Any]] = {}
        for member, name in wanted.items():
            size = zf.getinfo(member).file_size
            try:
                with zf.open(member) as fp:
                    out[name] = _read_npy(fp, size, np.dtype(expected[name]), max_elems)
            except ValueError as exc:
                raise ValueError(f"npz member {member}: {exc}") from None
            except ZIP_ERRORS as exc:
                why = f"{type(exc).__name__}: {exc}"
                raise ValueError(f"npz member {member}: unreadable ({why})") from None
        return out


def _read_npy(fp: IO[bytes], size: int, dtype: np.dtype[Any], max_elems: int) -> NDArray[Any]:
    """One npy member of declared ``size`` bytes."""
    version = np.lib.format.read_magic(fp)  # ValueError on a bad magic string
    if version == (1, 0):
        shape, fortran, found = np.lib.format.read_array_header_1_0(
            fp, max_header_size=MAX_HEADER_SIZE
        )
    elif version == (2, 0):
        shape, fortran, found = np.lib.format.read_array_header_2_0(
            fp, max_header_size=MAX_HEADER_SIZE
        )
    else:
        raise ValueError(f"unsupported npy format version {version}")
    if found.hasobject or found.kind == "V":
        raise ValueError(f"dtype {found} is not allowed (object or void)")
    if found != dtype:
        raise ValueError(f"dtype {found} where {dtype} was expected")
    if any(d < 0 for d in shape):
        raise ValueError(f"negative dimension in shape {shape}")
    n = math.prod(shape)
    if n > max_elems:
        raise ValueError(f"shape {shape} has {n} elements, more than the limit {max_elems}")
    nbytes = n * found.itemsize
    left = size - fp.tell()
    if left != nbytes:
        raise ValueError(f"{left} data bytes where the header declares {nbytes}")
    data = np.empty(nbytes, dtype=np.uint8)  # filled in place: no second copy
    view, got = memoryview(data), 0
    while got < nbytes:
        chunk = fp.read(min(_CHUNK, nbytes - got))
        if not chunk:
            raise ValueError(f"truncated data: {got} of {nbytes} bytes")
        view[got : got + len(chunk)] = chunk
        got += len(chunk)
    return data.view(found).reshape(shape, order="F" if fortran else "C")
