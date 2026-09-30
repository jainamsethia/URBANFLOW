"""Golden content hashes of generated scenarios (plan U.3): derivation or generator changes
must be deliberate. Update a hash only together with a changelog entry.

History (the CHANGELOG arrives in P17): P3 calibrated the built-in car's ``headway`` to
1.1 s (B.2 #25), which is part of every resolved scenario's vehicle types."""

from __future__ import annotations

from typing import Any

import pytest

from urbanflow import generate

GOLDEN = [
    ("single_intersection", {}, "3aefba1f3f7f8e9732b1882ba485c7c0c5f2d632970e0a566ebeb05709fdb958"),
    (
        "single_intersection",
        {"kind": "priority"},
        "cf944ba71bd621e6fd7aabf814e360192d43b9d46ecec6d1ef0e4e6b48974b74",
    ),
    (
        "single_intersection",
        {"kind": "uncontrolled"},
        "3c0ffa9b370bdb62cfc100351e90bbf2551c7640db651350320e4add8c9f04c3",
    ),
    (
        "single_intersection",
        {"arms": 3},
        "fb0ac9593c9ca78df7bcba17d950e7d9fa60553ef1f68846f86c70469b88d23d",
    ),
]


@pytest.mark.parametrize(("name", "params", "digest"), GOLDEN)
def test_generator_content_hash(name: str, params: dict[str, Any], digest: str) -> None:
    assert generate(name, params).content_hash == digest
