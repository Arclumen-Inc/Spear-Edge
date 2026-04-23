#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "contracts" / "edge-tripwire" / "EVENTS.v1.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "tripwire_edge_contract"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(payload: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def validate_number(name: str, value: Any, rule: Dict[str, Any]) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if "gt" in rule and not value > rule["gt"]:
        raise ValueError(f"{name} must be > {rule['gt']}")
    if "min" in rule and not value >= rule["min"]:
        raise ValueError(f"{name} must be >= {rule['min']}")
    if "max" in rule and not value <= rule["max"]:
        raise ValueError(f"{name} must be <= {rule['max']}")


def validate_payload(contract: Dict[str, Any], payload: Dict[str, Any], source: str) -> None:
    envelope = contract["envelope"]
    for key in envelope["required_fields"]:
        if key not in payload:
            raise ValueError(f"{source}: missing envelope field '{key}'")
    if payload["schema"] != envelope["schema_literal"]:
        raise ValueError(f"{source}: schema must equal '{envelope['schema_literal']}'")

    entity_type = payload.get("type")
    if entity_type not in contract["entities"]:
        raise ValueError(f"{source}: unknown entity type '{entity_type}'")
    entity = contract["entities"][entity_type]

    for key in entity.get("required", []):
        if key not in payload:
            raise ValueError(f"{source}: {entity_type} missing required field '{key}'")

    for key, rule in entity.get("constraints", {}).items():
        value = get_nested(payload, key)
        if value is None:
            continue
        if "equals" in rule and value != rule["equals"]:
            raise ValueError(f"{source}: {key} must equal {rule['equals']}")
        if rule.get("type") == "number":
            validate_number(key, value, rule)


def main() -> int:
    contract = load_json(CONTRACT_PATH)
    fixtures = sorted(FIXTURE_DIR.glob("*.valid.json"))
    if not fixtures:
        print("No fixtures found.")
        return 1

    try:
        for fixture in fixtures:
            payload = load_json(fixture)
            validate_payload(contract, payload, fixture.name)
    except Exception as exc:
        print(f"Contract validation failed: {exc}")
        return 1

    print(f"Contract validation passed for {len(fixtures)} fixtures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
