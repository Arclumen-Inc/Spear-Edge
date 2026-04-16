# Tripwire ↔ SPEAR Edge — operator notes (MVP)

## HTTP vs WebSocket ports

Tripwire defaults to **HTTP ingest on 8000** and a **WebSocket control plane on 8080**. SPEAR Edge can run in two common layouts:

1. **Split processes** — Run the small **ingest** app on port 8000 (forwards events to the main UI/API) and the **main Edge** app on **8080** (HTTP API, WebSockets, static UI). Point Tripwire at the ingest URL for `POST /api/tripwire/event` as configured (e.g. `ingest_port`).

2. **Single uvicorn on 8080** — One process serves both HTTP and WebSockets on **8080**. Configure Tripwire so **ingest HTTP** and **Tripwire↔Edge WebSocket** target **8080** (not 8000 for one and 8080 for the other unless that matches your split setup).

If WebSocket connects but HTTP ingest fails (or the reverse), check that Tripwire’s **ingest URL/port** and **Edge WS URL/port** match the process you actually started.

## Node registry and HTTP events

**WebSocket hello / heartbeat** from a Tripwire node is the **authoritative** source for **node registry** (identity, callsign, etc.). **`POST /api/tripwire/event` does not register nodes.**

Operators should treat **WS connected** as the signal that node metadata is fully known. HTTP-only paths may have incomplete `node_id` / callsign resolution until the link is up; AoA fusion and similar features **merge payload fields with the registry** when available.

## RF-only Edge

**`audio_cue`** events are **acknowledged** by Edge but **not processed** (no RF capture, no UI cue). Edge is RF-focused; acoustic cues are ignored by design.
