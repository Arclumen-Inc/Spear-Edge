# spear_edge/api/ws/tripwire_link_ws.py
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

HELLO_TIMEOUT_S = 5.0
STALE_AFTER_S = 6.0  # UI can show amber if no heartbeat > this

print("[TRIPWIRE-WS] handler entered")

async def tripwire_link_ws(websocket: WebSocket, orchestrator) -> None:
    client_ip = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "?"

    try:
        await websocket.accept()
        print(f"[TRIPWIRE-WS] Connection accepted from {client_ip}:{client_port}")
    except Exception as e:
        print(f"[TRIPWIRE-WS] Failed to accept connection from {client_ip}: {e}")
        return

    node_id: Optional[str] = None

    try:
        # --- Require HELLO within timeout ---
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=HELLO_TIMEOUT_S)
        except asyncio.TimeoutError:
            print(f"[TRIPWIRE-WS] No hello received within {HELLO_TIMEOUT_S}s from {client_ip}")
            await websocket.close(code=1008, reason="No hello message")
            return

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[TRIPWIRE-WS] Invalid JSON in hello from {client_ip}")
            await websocket.close(code=1008, reason="Invalid JSON")
            return

        if msg.get("type") != "hello":
            print(f"[TRIPWIRE-WS] First message not 'hello' from {client_ip}")
            await websocket.close(code=1008, reason="Expected hello")
            return

        node_id = str(msg.get("node_id") or "").strip()
        if not node_id:
            print(f"[TRIPWIRE-WS] Missing node_id in hello from {client_ip}")
            await websocket.close(code=1008, reason="Missing node_id")
            return

        callsign = str(msg.get("callsign") or node_id).strip()
        meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
        gps = msg.get("gps") if isinstance(msg.get("gps"), dict) else None
        system_time = msg.get("system_time") if isinstance(msg.get("system_time"), (int, float)) else None

        print(f"[TRIPWIRE-WS] Registered node: {node_id} ({callsign}) from {client_ip}")

        # --- Replace old connection if exists ---
        if not hasattr(orchestrator, "tripwire_links"):
            orchestrator.tripwire_links = {}

        old_ws = orchestrator.tripwire_links.get(node_id)
        if old_ws and old_ws is not websocket:
            print(f"[TRIPWIRE-WS] Closing old connection for {node_id}")
            try:
                await old_ws.close(code=1000, reason="Replaced by new connection")
            except Exception:
                pass

        orchestrator.tripwire_links[node_id] = websocket

        # --- Update registry and notify UI ---
        orchestrator.tripwires.mark_ws_connected(
            node_id=node_id,
            callsign=callsign,
            ip=client_ip,
            meta=meta,
            gps=gps,
            system_time=system_time,
        )
        orchestrator.bus.publish_nowait("tripwire_nodes", {"nodes": orchestrator.tripwires.snapshot()})

        # --- Send current Edge state back ---
        await websocket.send_text(json.dumps({"type": "edge_state", "mode": orchestrator.mode}))

        # --- Main loop: heartbeats and status ---
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue  # ignore malformed

            msg_type = msg.get("type")

            if msg_type == "heartbeat":
                orchestrator.tripwires.mark_ws_heartbeat(node_id=node_id)
                orchestrator.bus.publish_nowait("tripwire_nodes", {"nodes": orchestrator.tripwires.snapshot()})
                continue

            if msg_type == "status":
                orchestrator.tripwires.update_ws_status(node_id=node_id, status=msg)
                orchestrator.bus.publish_nowait("tripwire_nodes", {"nodes": orchestrator.tripwires.snapshot()})
                continue

            # Ignore unknown types silently

    except WebSocketDisconnect:
        print(f"[TRIPWIRE-WS] Client disconnected: {node_id or client_ip}")
    except Exception as e:
        print(f"[TRIPWIRE-WS] Unexpected error for {node_id or client_ip}: {e}")
    finally:
        # --- Cleanup ---
        if node_id:
            if getattr(orchestrator, "tripwire_links", {}).get(node_id) is websocket:
                orchestrator.tripwire_links.pop(node_id, None)

            orchestrator.tripwires.mark_ws_disconnected(node_id=node_id)
            orchestrator.bus.publish_nowait("tripwire_nodes", {"nodes": orchestrator.tripwires.snapshot()})
            print(f"[TRIPWIRE-WS] Cleaned up node: {node_id}")