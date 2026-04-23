# Edge-Tripwire Interface Anchor

## HTTP Ingest
- **Endpoint:** `POST /api/tripwire/event`
- **Legacy alias:** `POST /api/tripwire/cue`
- **Body contract:** `EVENTS.v1.json`
- **Contract authority:** Tripwire emitted payload shape is canonical; Edge ingest must remain compatible.

## WebSocket Control Link
- **Endpoint:** `/ws/tripwire` (alias `/ws/tripwire_link`)
- **Required first message:** `hello`
- **Keepalive:** `heartbeat`
- **Status update:** `status`
- **Command ack:** `command_response`

## Event Matrix
- `rf_cue`: advisory cue stream
- `fhss_cluster`: actionable candidate in armed mode policy path
- `aoa_cone`: advisory geometry for fusion/TAI
- `bearing_line`: manual DF geometry for fusion/TAI

## Expected Envelope Example
```json
{
  "schema": "spear.tripwire.event.v1",
  "node_id": "tripwire-01",
  "callsign": "TW-01",
  "gps": { "lat": 37.0, "lon": -122.0, "alt_ft": 120.0 },
  "meta": { "sdr_driver": "bladerf" },
  "type": "rf_cue",
  "timestamp": 1710000000.0,
  "freq_hz": 915000000.0
}
```

## Contract Validation
- Canonical examples live in `tests/fixtures/tripwire_edge_contract/`.
- Validation test: `tests/test_tripwire_edge_contract_anchor.py`.
- Optional script: `scripts/validate_tripwire_edge_contract.py`.
- Edge fixture updates must track Tripwire reference behavior.
