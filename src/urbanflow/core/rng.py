"""Named, independent, reproducible random streams (plan B.2 #11, F.6).

Every stream is ``PCG64(SeedSequence(seed, spawn_key=tuple(name.encode("utf-8"))))``:
keyed by the UTF-8 bytes of its name, so distinct names never collide and results do not
depend on ``hash()`` randomisation, draw order in other streams, or dict order.
Stream names: ``flow:{id}``, ``trip:{id}``, ``transit:{id}``, ``vehicle_params``,
``controller:{intersection}``, ``generator:{name}``, ``model:{name}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["RngStreams"]


class RngStreams:
    """Lazily created ``numpy.random.Generator`` streams derived from one root seed."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed}")
        self._seed = int(seed)
        self._streams: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        return self._seed

    def stream(self, name: str) -> np.random.Generator:
        """The generator for ``name`` (created on first use, then the same object)."""
        gen = self._streams.get(name)
        if gen is None:
            if not name:
                raise ValueError("stream name must not be empty")
            seq = np.random.SeedSequence(self._seed, spawn_key=tuple(name.encode("utf-8")))
            gen = self._streams[name] = np.random.Generator(np.random.PCG64(seq))
        return gen

    def state_dict(self) -> dict[str, Any]:
        """JSON-safe state: the root seed and every created stream's bit-generator state."""
        return {
            "seed": self._seed,
            "streams": {n: g.bit_generator.state for n, g in sorted(self._streams.items())},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state_dict` output; streams absent from it start fresh when used."""
        fresh = RngStreams(int(state["seed"]))
        for name, bit_state in state["streams"].items():
            fresh.stream(name).bit_generator.state = dict(bit_state)
        self._seed, self._streams = fresh._seed, fresh._streams
