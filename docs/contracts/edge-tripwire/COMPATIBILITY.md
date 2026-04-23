# Edge-Tripwire Compatibility Anchor

## Active Compatibility Modes
- `event_type` accepted as legacy alias for `type`.
- Nested `rf_event` wrappers may be unwrapped by consumer paths.
- `freq_span_mhz` and `span_mhz` are both tolerated for FHSS span.
- `tripwire_id` and `node_id` are both tolerated for bearing-line source identity.
- Legacy endpoint `/api/tripwire/cue` forwards to `/api/tripwire/event`.

## Known Drift Risks
- Confidence scale split (`0..1` vs `0..100`) across entity types.
- GPS mapped under `gps` envelope vs flattened `gps_lat/gps_lon` assumptions in some metadata paths.
- IBW cluster remap (`fhss_cluster` -> `rf_cue`) may alter actionability in policy flows.

## Deprecation Policy
- Keep alias support for at least one full release cycle after producer migration.
- Mark deprecated fields in this file and `CHANGELOG.md`.
- Remove alias support only after both repos pass contract tests without that alias.

## Breaking Change Checklist
1. Bump contract version (e.g., `v1` -> `v2`).
2. Update `EVENTS.v1.json` (or add `EVENTS.v2.json`).
3. Update fixtures + tests in both repos.
4. Add migration notes with rollout window.
