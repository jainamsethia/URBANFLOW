"""Named RNG streams: deterministic, independent, restorable (plan B.2 #11, F.6)."""

from __future__ import annotations

import ast
import inspect
import json

import numpy as np
import pytest

from urbanflow.core import rng as rng_module
from urbanflow.core.rng import RngStreams


def test_stream_is_the_documented_construction() -> None:
    expected = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(42, spawn_key=tuple(b"flow:a")))
    )
    assert np.array_equal(
        RngStreams(42).stream("flow:a").integers(0, 2**32, 16), expected.integers(0, 2**32, 16)
    )


def test_same_name_same_sequence_and_cached() -> None:
    a, b = RngStreams(7), RngStreams(7)
    assert a.stream("trip:x") is a.stream("trip:x")
    assert np.array_equal(a.stream("trip:x").random(8), b.stream("trip:x").random(8))


def test_streams_are_independent_of_each_other() -> None:
    # Common random numbers: drawing from one stream never shifts another.
    a, b = RngStreams(1), RngStreams(1)
    a.stream("flow:noise").random(1000)
    assert np.array_equal(a.stream("flow:main").random(8), b.stream("flow:main").random(8))


@pytest.mark.parametrize(
    ("n1", "n2", "s1", "s2"),
    [("flow:a", "flow:b", 0, 0), ("flow:a", "flow:a", 0, 1), ("ab", "ba", 3, 3), ("é", "e", 0, 0)],
)
def test_different_names_or_seeds_differ(n1: str, n2: str, s1: int, s2: int) -> None:
    x = RngStreams(s1).stream(n1).random(8)
    y = RngStreams(s2).stream(n2).random(8)
    assert not np.array_equal(x, y)


def test_state_round_trip_through_json() -> None:
    r = RngStreams(99)
    r.stream("flow:a").random(5)
    r.stream("controller:J").integers(0, 10, 3)
    state = json.loads(json.dumps(r.state_dict()))
    expected = {n: r.stream(n).random(4) for n in ("flow:a", "controller:J")}

    other = RngStreams(0)
    other.stream("unrelated").random(3)
    other.load_state_dict(state)
    assert other.seed == 99
    assert other.state_dict()["streams"].keys() == {"flow:a", "controller:J"}
    for name, values in expected.items():
        assert np.array_equal(other.stream(name).random(4), values)
    # A stream that was not in the state starts fresh from the restored seed.
    assert np.array_equal(other.stream("new").random(2), RngStreams(99).stream("new").random(2))


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RngStreams(-1)
    with pytest.raises(ValueError, match="empty"):
        RngStreams(0).stream("")


def test_module_never_uses_hash() -> None:
    tree = ast.parse(inspect.getsource(rng_module))
    calls = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "hash"
    ]
    assert not calls
