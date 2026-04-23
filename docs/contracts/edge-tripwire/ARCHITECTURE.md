# Edge-Tripwire Architecture Anchor

## Purpose
This document is the shared architecture anchor for SPEAR Edge and SPEAR Tripwire integration.
Tripwire is the canonical reference for outbound interface semantics; Edge implements and validates against that contract.

## System Boundaries
- **Tripwire (producer):** Detects RF/AoA events and publishes normalized event payloads.
- **Edge (consumer + authority):** Ingests events, applies actionability policy, drives capture/UI/TAK outputs.
- **Operator UI:** Shows cue stream, node health, AoA/TAI overlays.

## Primary Dataflow
1. Tripwire produces event payloads (`rf_cue`, `fhss_cluster`, `aoa_cone`, `bearing_line`).
2. Tripwire sends events to Edge HTTP ingest (`/api/tripwire/event`).
3. Edge parses payload and applies policy:
   - advisory-only events are shown/logged
   - actionable events may trigger capture in armed mode
4. Edge emits UI notifications over `/ws/notify`.
5. Edge optionally forwards geometry outputs to TAK.

## Control Plane
- Tripwire and Edge maintain a WS control link (`/ws/tripwire` or `/ws/tripwire_link`).
- Core message types: `hello`, `heartbeat`, `status`, `command_response`.
- Node registry truth comes from the WS link, not from HTTP event side effects.

## Ownership Rules
- Tripwire owns event generation and envelope population.
- Edge owns capture policy and final actionability decisions.
- Interface shape in `EVENTS.v1.json` is defined from Tripwire output and consumed by Edge.

## Change Protocol
- Any payload shape/unit/semantic change must update:
  - `EVENTS.v1.json`
  - `CONSTRAINTS.md`
  - `COMPATIBILITY.md`
  - `CHANGELOG.md`
- Changes originate from Tripwire behavior and must be reflected in Edge fixtures/tests in the same PR window.
- Contract changes require matching fixture/test updates.
