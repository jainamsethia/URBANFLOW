"""Safe npz writing and reading (plan K.5, U.4): no pickle, strict members, dtypes and sizes."""

from __future__ import annotations

import io
import struct
import zipfile
from typing import Any

import numpy as np
import pytest

from urbanflow.core import constants as C
from urbanflow.core.npz import MAX_HEADER_SIZE, read_npz, write_npz


def _npz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _npy(header: dict[str, Any] | str, data: bytes = b"", *, major: int = 1) -> bytes:
    """A hand-made npy member: magic, version, header (padded to 64 bytes), data."""
    text = (header if isinstance(header, str) else repr(header)).encode("latin1")
    size = 4 if major == 2 else 2
    pad = -(6 + 2 + size + len(text) + 1) % 64
    text += b" " * pad + b"\n"
    length = struct.pack("<I" if major == 2 else "<H", len(text))
    return b"\x93NUMPY" + bytes([major, 0]) + length + text + data


def _saved(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    write_npz(buf, arrays, 1)
    return buf.getvalue()


ARRAYS = {
    "f8": np.array([1.5, np.nan, -np.inf, 0.1 + 0.2]),
    "f4": np.arange(6, dtype=np.float32).reshape(2, 3),
    "fortran": np.asfortranarray(np.arange(12, dtype=np.int64).reshape(3, 4)),
    "u1": np.array([0, 255], dtype=np.uint8),
    "b": np.array([True, False, True]),
    "empty": np.zeros(0, dtype=np.int32),
    "scalar": np.array(7, dtype=np.int16),
}


def test_round_trip_and_standard_format() -> None:
    data = _saved(ARRAYS)
    out = read_npz(data, {k: v.dtype for k, v in ARRAYS.items()}, max_elems=100)
    assert list(out) == list(ARRAYS)
    for name, arr in ARRAYS.items():
        assert out[name].dtype == arr.dtype and out[name].shape == arr.shape
        np.testing.assert_array_equal(out[name], arr)
        assert out[name].flags.writeable
    with np.load(io.BytesIO(data), allow_pickle=False) as std:  # any numpy can read it
        np.testing.assert_array_equal(std["fortran"], ARRAYS["fortran"])
    stream = io.BytesIO(data)  # a file object works too
    assert read_npz(stream, {k: v.dtype for k, v in ARRAYS.items()}, 100)["u1"].tolist() == [0, 255]


def test_writer_refuses_object_arrays() -> None:
    with pytest.raises(ValueError, match="pickle"):
        write_npz(io.BytesIO(), {"x": np.array([{"a": 1}], dtype=object)}, 1)


def test_rejects_object_and_void_dtypes() -> None:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.array([1, "a"], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match=r"x\.npy: dtype object is not allowed"):
        read_npz(_npz({"x.npy": buf.getvalue()}), {"x": object}, 10)
    void = _npy({"descr": "|V8", "fortran_order": False, "shape": (1,)}, b"\0" * 8)
    with pytest.raises(ValueError, match="is not allowed"):
        read_npz(_npz({"x.npy": void}), {"x": np.float64}, 10)


def test_rejects_unexpected_dtypes() -> None:
    data = _saved({"x": np.zeros(3, dtype=np.float32)})
    with pytest.raises(ValueError, match=r"dtype float32 where float64 was expected"):
        read_npz(data, {"x": np.float64}, 10)
    swapped = _saved({"x": np.zeros(3, dtype=">f8")})
    with pytest.raises(ValueError, match="where float64 was expected"):
        read_npz(swapped, {"x": np.float64}, 10)


def test_rejects_oversize_and_negative_shapes_before_reading() -> None:
    huge = _npy({"descr": "<f8", "fortran_order": False, "shape": (10**12,)})
    with pytest.raises(ValueError, match=r"has 1000000000000 elements, more than the limit 64"):
        read_npz(_npz({"x.npy": huge}), {"x": np.float64}, 64)
    with pytest.raises(ValueError, match="more than the limit 5"):
        read_npz(_saved({"x": np.zeros((2, 3))}), {"x": np.float64}, 5)
    negative = _npy({"descr": "<f8", "fortran_order": False, "shape": (-1,)})
    with pytest.raises(ValueError, match="negative dimension"):
        read_npz(_npz({"x.npy": negative}), {"x": np.float64}, 64)


def test_rejects_truncated_and_trailing_data() -> None:
    header = {"descr": "<f8", "fortran_order": False, "shape": (4,)}
    with pytest.raises(ValueError, match=r"x\.npy: 24 data bytes where the header declares 32"):
        read_npz(_npz({"x.npy": _npy(header, b"\0" * 24)}), {"x": np.float64}, 10)
    with pytest.raises(ValueError, match="33 data bytes where the header declares 32"):
        read_npz(_npz({"x.npy": _npy(header, b"\0" * 33)}), {"x": np.float64}, 10)
    ok = read_npz(_npz({"x.npy": _npy(header, b"\0" * 32)}), {"x": np.float64}, 10)
    assert ok["x"].tolist() == [0.0] * 4
    v2 = _npy(header, np.arange(4.0).tobytes(), major=2)  # format 2.0 headers are read too
    assert read_npz(_npz({"x.npy": v2}), {"x": np.float64}, 10)["x"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("member", "message"),
    [
        (b"NOTNUMPY" + b"\0" * 20, "magic"),
        (_npy("not a dict"), "Cannot parse header"),
        (_npy("{'descr': '<f8'}"), "correct keys"),
        (_npy({"descr": "<f8", "fortran_order": False, "shape": (1.5,)}), "shape is not valid"),
        (_npy({"descr": "<f8", "fortran_order": False, "shape": (1,)}, major=3), "version"),
        (
            _npy(
                {"descr": "<f8", "fortran_order": False, "shape": (0,), "x": " " * MAX_HEADER_SIZE}
            ),
            "max_header_size",
        ),
        (b"\x93NUMPY", "EOF"),
    ],
)
def test_rejects_bad_headers(member: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_npz(_npz({"x.npy": member}), {"x": np.float64}, 10)


def test_rejects_unexpected_missing_and_duplicate_members() -> None:
    data = _saved({"a": np.zeros(1), "evil": np.zeros(1)})
    with pytest.raises(ValueError, match=r"unexpected npz members: evil.npy"):
        read_npz(data, {"a": np.float64}, 10)
    with pytest.raises(ValueError, match=r"missing npz members: b.npy"):
        read_npz(_saved({"a": np.zeros(1)}), {"a": np.float64, "b": np.float64}, 10)
    with pytest.raises(ValueError, match=r"unexpected npz members: ../a.npy"):
        read_npz(_npz({"../a.npy": _npy({})}), {"a": np.float64}, 10)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf, pytest.warns(UserWarning, match="Duplicate"):
        zf.writestr("a.npy", b"1")
        zf.writestr("a.npy", b"2")
    with pytest.raises(ValueError, match="duplicate members"):
        read_npz(buf.getvalue(), {"a": np.float64}, 10)


def test_rejects_non_zip_input() -> None:
    with pytest.raises(ValueError, match="not an npz archive"):
        read_npz(b"plain bytes", {"a": np.float64}, 10)


def test_never_unpickles(monkeypatch: pytest.MonkeyPatch) -> None:
    """``np.load`` is never called (the only pickle path of numpy's readers)."""

    def banned(*args: object, **kwargs: object) -> None:
        raise AssertionError("np.load must not be used")

    monkeypatch.setattr(np, "load", banned)
    out = read_npz(_saved({"a": np.arange(3)}), {"a": np.int64}, 10)
    assert out["a"].tolist() == [0, 1, 2]


def _patched(data: bytes, signature: bytes, offset: int, value: bytes) -> bytes:
    """``data`` with ``value`` written ``offset`` bytes into its first ``signature`` record."""
    at = data.index(signature) + offset
    return data[:at] + value + data[at + len(value) :]


def test_decompression_bombs_are_rejected_from_the_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P4 review: 8 MiB of zeros deflate to ~8 kB. The members' declared sizes are summed
    from the zip directory and checked against ``max_bytes`` before anything is inflated
    or allocated (np.empty is never reached)."""
    bomb = _saved({"x": np.zeros(1 << 20, dtype=np.int64)})
    assert len(bomb) < 64 * 1024
    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", lambda *_a, **_k: pytest.fail("allocated"))
        with pytest.raises(ValueError, match=r"declare \d+ bytes, more than the limit 1048576"):
            read_npz(bomb, {"x": np.int64}, 1 << 30, max_bytes=1 << 20)
    assert read_npz(bomb, {"x": np.int64}, 1 << 30)["x"].shape == (1 << 20,)  # default budget
    assert C.NPZ_MAX_BYTES == 1 << 31  # K.5: 2 GiB


def test_corrupt_zip_data_is_a_value_error() -> None:
    """P4 review: zlib.error (corrupt deflate stream), RuntimeError (encrypted member) and
    NotImplementedError (unsupported method) are not OSErrors; all surface as ValueError."""
    good = _saved({"a": np.arange(2000)})
    name_len, extra_len = struct.unpack("<HH", good[26:30])
    start = 30 + name_len + extra_len + 40  # inside the first member's deflate stream
    corrupt = good[:start] + bytes(b ^ 0xFF for b in good[start : start + 16]) + good[start + 16 :]
    with pytest.raises(ValueError, match=r"npz member a\.npy: unreadable \(error: "):
        read_npz(corrupt, {"a": np.int64}, 10_000)
    central = b"PK\x01\x02"
    encrypted = _patched(good, central, 8, b"\x01\x00")  # general purpose flag bit 0
    with pytest.raises(ValueError, match=r"a\.npy: unreadable \(RuntimeError: .*encrypted"):
        read_npz(encrypted, {"a": np.int64}, 10_000)
    for method in (9, 99):  # deflate64, AES
        odd = _patched(good, central, 10, struct.pack("<H", method))
        with pytest.raises(ValueError, match=r"a\.npy: unreadable \(NotImplementedError: "):
            read_npz(odd, {"a": np.int64}, 10_000)
