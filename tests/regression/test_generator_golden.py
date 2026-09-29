"""Golden content hashes of generated scenarios (plan U.3): derivation or generator changes
must be deliberate. Update a hash only together with a changelog entry."""

from __future__ import annotations

from typing import Any

import pytest

from urbanflow import generate

GOLDEN = [
    ("single_intersection", {}, "33193155c163ff7172a9b01205254366d2593c7f5c8d165ccb05074ccc39df53"),
    (
        "single_intersection",
        {"kind": "priority"},
        "60014dc6431cbd1c3c426e8e951f46f9f1b9b7d36dceb16f210b3be55f789505",
    ),
    (
        "single_intersection",
        {"kind": "uncontrolled"},
        "87f6c97627d0332b909acff9d547819568caed51991e14fd074821cbb2b21003",
    ),
    (
        "single_intersection",
        {"arms": 3},
        "c5322d835743a088cf3c62b8491cbe3b1409502d3ea2999dda07b56a6459e3ce",
    ),
]


@pytest.mark.parametrize(("name", "params", "digest"), GOLDEN)
def test_generator_content_hash(name: str, params: dict[str, Any], digest: str) -> None:
    assert generate(name, params).content_hash == digest
