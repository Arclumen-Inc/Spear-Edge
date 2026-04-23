# Edge-Tripwire Contract Anchors

This folder is the Edge-side anchor for integration between SPEAR Tripwire and SPEAR Edge.
Tripwire is the reference source-of-truth for payload semantics; this folder mirrors that contract for Edge validation.

## Files
- `ARCHITECTURE.md`: ownership boundaries and dataflow
- `CONSTRAINTS.md`: units, ranges, and semantic invariants
- `INTERFACES.md`: HTTP/WS interface mapping
- `COMPATIBILITY.md`: aliases, drift risks, deprecation policy
- `EVENTS.v1.json`: machine-readable contract
- `CHANGELOG.md`: contract-level changes only

## Validation
- Test fixtures: `tests/fixtures/tripwire_edge_contract/*.valid.json`
- Pytest: `tests/test_tripwire_edge_contract_anchor.py`
- Script: `python3 scripts/validate_tripwire_edge_contract.py`

## Update Rule
Any producer/consumer interface change must update this folder and validation fixtures in the same PR.
Tripwire contract updates must be mirrored here before Edge changes are merged.
