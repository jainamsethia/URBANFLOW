"""The parameter table of docs/vehicle-model.md matches the code (plan G.7, AE.1)."""

from __future__ import annotations

import re
from pathlib import Path

from urbanflow.core import constants as C
from urbanflow.scenario.schema import VehicleTypeSpec
from urbanflow.vehicles import IDM

DOCS = Path(__file__).resolve().parents[2] / "docs"
ROW = re.compile(r"^\| [^|]+ \| `([A-Za-z_.]+)` \| ([^|]+) \|", re.MULTILINE)


def test_vehicle_model_parameter_table_matches_the_code() -> None:
    rows = ROW.findall((DOCS / "vehicle-model.md").read_text(encoding="utf-8"))
    checked = 0
    for name, cell in rows:
        try:
            shown = float(cell)
        except ValueError:
            assert name == "speed_factor", name  # a distribution, described in words
            continue
        if name.isupper():
            expected = getattr(C, name)
        elif name.startswith("model_params."):
            expected = IDM.Params.model_fields[name.split(".", 1)[1]].default
        else:
            expected = VehicleTypeSpec.model_fields[name].default
        assert shown == expected, name
        checked += 1
    assert checked >= 16
