"""Regenerate the scenario-format references from the code (plan AE.1).

* ``docs/reference/scenario.schema.json``: the JSON Schema of the scenario format.
* ``docs/scenario-format.md``: the blocks between ``<!-- BEGIN GENERATED: x -->`` and
  ``<!-- END GENERATED: x -->`` markers (``example``: the tested example scenario,
  ``schema``: field tables from the JSON Schema, ``issue-codes``: the validation code
  table from ``ISSUE_CODES``).

Usage::

    uv run python scripts/gen_scenario_docs.py          # rewrite the files
    uv run python scripts/gen_scenario_docs.py --check  # exit 1 if anything is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from urbanflow.scenario.schema import scenario_json_schema
from urbanflow.scenario.validate import ISSUE_CODES

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_FILE = ROOT / "docs" / "reference" / "scenario.schema.json"
PAGE = ROOT / "docs" / "scenario-format.md"
EXAMPLE = ROOT / "tests" / "fixtures" / "scenarios" / "demo.json"  # the E.7 §1.8 example
STAGES = {
    "L": "load",
    "S": "structure",
    "P": "pre-derive",
    "Q": "post-derive",
    "X": "deep check",
    "B": "builder / ops",
}
# Order of the schema reference tables (the rest follow alphabetically).
MODEL_ORDER = (
    "ScenarioSpec",
    "MetaSpec",
    "GeneratorInfo",
    "SimulationSpec",
    "NetworkSpec",
    "IntersectionSpec",
    "RoadSpec",
    "LaneSpec",
    "MovementSpec",
    "ConnectionSpec",
    "SignalSpec",
    "ControllerSpec",
    "PhaseSpec",
    "VehicleTypeSpec",
    "SpeedFactorSpec",
    "DemandSpec",
    "FlowSpec",
    "RouteChoiceSpec",
    "TripSpec",
    "TransitLineSpec",
    "StopSpec",
)
_CONSTRAINTS = (
    ("minimum", ">= {}"),
    ("exclusiveMinimum", "> {}"),
    ("maximum", "<= {}"),
    ("exclusiveMaximum", "< {}"),
    ("minItems", "min {} items"),
    ("maxItems", "max {} items"),
    ("minLength", "min length {}"),
    ("maxLength", "max length {}"),
    ("minProperties", "min {} entries"),
    ("pattern", "pattern `{}`"),
)


def schema_text() -> str:
    """Content of ``docs/reference/scenario.schema.json``."""
    return json.dumps(scenario_json_schema(), indent=2, ensure_ascii=False) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _type(prop: dict[str, Any]) -> str:
    if "$ref" in prop:
        name = prop["$ref"].rsplit("/", 1)[-1]
        return f"[{name}](#{name.lower()})"
    if "anyOf" in prop:
        return " or ".join(_type(p) for p in prop["anyOf"])
    if "enum" in prop:
        return " or ".join(json.dumps(v) for v in prop["enum"])
    if "const" in prop:
        return json.dumps(prop["const"])
    kind = prop.get("type")
    if kind == "array":
        if "prefixItems" in prop:
            return "[" + ", ".join(_type(p) for p in prop["prefixItems"]) + "]"
        return f"{_type(prop.get('items', {}))}[]"
    if kind == "object":
        values = prop.get("additionalProperties")
        return f"{{string: {_type(values)}}}" if isinstance(values, dict) else "object"
    return str(kind or "any")


def _constraints(prop: dict[str, Any]) -> str:
    found = [fmt.format(prop[key]) for key, fmt in _CONSTRAINTS if key in prop]
    for option in prop.get("anyOf", ()):
        found += [fmt.format(option[key]) for key, fmt in _CONSTRAINTS if key in option]
    return ", ".join(dict.fromkeys(found))


def schema_tables() -> str:
    """Markdown tables of every model in the JSON Schema."""
    schema = scenario_json_schema()
    models = {"ScenarioSpec": schema, **schema.get("$defs", {})}
    order = [m for m in MODEL_ORDER if m in models] + sorted(set(models) - set(MODEL_ORDER))
    parts = []
    for name in order:
        model = models[name]
        required = set(model.get("required", ()))
        lines = [f"### {name}", "", _cell(model.get("description", "")), ""]
        lines += ["| key | type | default | constraints | description |", "|---|---|---|---|---|"]
        for key, prop in model.get("properties", {}).items():
            default = "required" if key in required else json.dumps(prop.get("default", "-"))
            if key not in required and "default" not in prop:
                default = "derived / optional"
            row = (
                f"`{key}`",
                _type(prop),
                default,
                _constraints(prop),
                prop.get("description", ""),
            )
            lines.append("| " + " | ".join(_cell(str(c)) for c in row) + " |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def issue_table() -> str:
    """Markdown table of every validation code."""
    lines = ["| code | severity | stage | message |", "|---|---|---|---|"]
    for code, tpl in sorted(ISSUE_CODES.items(), key=lambda kv: (kv[0][0] != "E", kv[0])):
        messages = " / ".join(f"`{m}`" for m in (tpl.template, *tpl.alternatives))
        row = (code, str(tpl.severity), f"{tpl.stage} ({STAGES[tpl.stage]})", messages)
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def render_page(text: str) -> str:
    """``text`` with every generated block replaced by fresh content."""
    example = EXAMPLE.read_text(encoding="utf-8").strip()
    blocks = {
        "example": "\n".join(("```json", example, "```")),
        "schema": schema_tables(),
        "issue-codes": issue_table(),
    }
    for name, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            raise SystemExit(f"{PAGE}: missing the markers of generated block {name!r}")
        text = pattern.sub(lambda m, b=body: f"{m.group(1)}{b}\n{m.group(2)}", text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report stale files")
    args = parser.parse_args(argv)
    page = PAGE.read_text(encoding="utf-8")
    wanted = {SCHEMA_FILE: schema_text(), PAGE: render_page(page)}
    stale = []
    for path, text in wanted.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != text:
            stale.append(path)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8", newline="\n")
                print(f"wrote {path}")
    if args.check and stale:
        print("stale: " + ", ".join(str(p) for p in stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
