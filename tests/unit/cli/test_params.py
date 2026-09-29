"""KEY=VALUE parsing shared by -p, --set and -i (plan AC 7.1)."""

from __future__ import annotations

from typing import Any

import pytest

from urbanflow.cli.params import parse_assignments
from urbanflow.core.errors import ConfigError


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], {}),
        (["rows=4"], {"rows": 4}),
        (["dt=0.5", "record=true", "max=null"], {"dt": 0.5, "record": True, "max": None}),
        (["kind=priority"], {"kind": "priority"}),
        (['name="4"'], {"name": "4"}),
        (["xs=[1, 2]", 'mix={"car": 1}'], {"xs": [1, 2], "mix": {"car": 1}}),
        (["a=NaN", "b=Infinity"], {"a": "NaN", "b": "Infinity"}),
        (["path=C:/x=y"], {"path": "C:/x=y"}),
        (["empty="], {"empty": ""}),
        (
            ["turn_ratios.far=0.2", "turn_ratios.straight=0.6", "turn_ratios.near=0.2"],
            {"turn_ratios": {"far": 0.2, "straight": 0.6, "near": 0.2}},
        ),
        (["a.b.c=1", "a.b.d=2", "a.e=3"], {"a": {"b": {"c": 1, "d": 2}, "e": 3}}),
        (["rows=4", "rows=5"], {"rows": 5}),
        ([" padded =1"], {"padded": 1}),
    ],
)
def test_parse(items: list[str], expected: dict[str, Any]) -> None:
    assert parse_assignments(items) == expected


@pytest.mark.parametrize(
    ("items", "fragment"),
    [
        (["rows"], 'invalid assignment "rows": expected KEY=VALUE'),
        (["=4"], 'invalid assignment "=4": expected KEY=VALUE'),
        (["a..b=1"], 'empty segment in key "a..b"'),
        (["a=1", "a.b=2"], '"a" already has a value'),
        (["a.b=1", "a=2"], '"a" already has nested keys'),
    ],
)
def test_errors_name_the_item(items: list[str], fragment: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse_assignments(items)
    assert fragment in str(info.value)
    assert info.value.exit_code == 4
