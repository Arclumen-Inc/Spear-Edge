# Edge-Tripwire Constraints Anchor

## Canonical Units
- `timestamp`: Unix epoch seconds (float)
- `freq_hz`, `bandwidth_hz`: Hz
- `signal_freq_mhz`, `freq_span_mhz`, `span_mhz`: MHz
- `bearing_deg`, `cone_width_deg`, `bearing_std_deg`: degrees clockwise from North
- `gps.lat`, `gps.lon`: decimal degrees (WGS84)

## Confidence Scales (Current Reality)
- RF events (`rf_cue`, `fhss_cluster`): expected `0.0 - 1.0`
- AoA and manual DF (`aoa_cone`, `bearing_line`): currently emitted as `0 - 100`
- Until unified, adapters must preserve entity-specific scale and not silently reinterpret.

## Required Envelope
- `schema = "spear.tripwire.event.v1"`
- `node_id`
- `callsign`
- `type`
- `timestamp`

## Required Per Entity
- `rf_cue`: `type`, `freq_hz`, `timestamp`
- `fhss_cluster`: `type`, `freq_hz`, `timestamp`
- `aoa_cone`: `type`, `bearing_deg`, `cone_width_deg`, `timestamp`
- `bearing_line`: `type`, `bearing_deg`, `timestamp`, `gps`

## Compatibility Aliases
- Discriminator fallback: `event_type` (legacy) -> `type` (preferred)
- FHSS span aliases: `freq_span_mhz` <-> `span_mhz`
- LOB node aliases: `tripwire_id` <-> `node_id`

## Guardrails
- Do not mix confidence scales inside one entity type.
- Do not change units without a contract version bump.
- Do not add implicit required fields without updating fixtures/tests in both repos.
