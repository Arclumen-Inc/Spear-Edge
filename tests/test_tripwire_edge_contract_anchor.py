from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "contracts" / "edge-tripwire" / "EVENTS.v1.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "tripwire_edge_contract"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(payload: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _validate_number(name: str, value: Any, rule: Dict[str, Any]) -> None:
    assert isinstance(value, (int, float)), f"{name} must be numeric"
    if "gt" in rule:
        assert value > rule["gt"], f"{name} must be > {rule['gt']}"
    if "min" in rule:
        assert value >= rule["min"], f"{name} must be >= {rule['min']}"
    if "max" in rule:
        assert value <= rule["max"], f"{name} must be <= {rule['max']}"


def _validate_payload(contract: Dict[str, Any], payload: Dict[str, Any]) -> None:
    envelope = contract["envelope"]
    for key in envelope["required_fields"]:
        assert key in payload, f"missing envelope field: {key}"
    assert payload["schema"] == envelope["schema_literal"]

    entity_type = payload["type"]
    entity = contract["entities"][entity_type]

    for key in entity["required"]:
        assert key in payload, f"{entity_type} missing required field: {key}"

    constraints = entity.get("constraints", {})
    for key, rule in constraints.items():
        value = _get_nested(payload, key)
        if value is None:
            continue
        if "equals" in rule:
            assert value == rule["equals"], f"{key} must equal {rule['equals']}"
        if rule.get("type") == "number":
            _validate_number(key, value, rule)


def test_contract_fixtures_validate() -> None:
    contract = _load_json(CONTRACT_PATH)
    fixtures = sorted(FIXTURE_DIR.glob("*.valid.json"))
    assert fixtures, "no contract fixtures found"
    for fixture_path in fixtures:
        payload = _load_json(fixture_path)
        _validate_payload(contract, payload)
