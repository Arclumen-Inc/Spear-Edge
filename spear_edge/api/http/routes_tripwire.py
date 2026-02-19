from fastapi import APIRouter, Request
from pydantic import BaseModel
import time

class TripwireEvent(BaseModel):
    schema: str
    node_id: str
    freq_hz: float
    confidence: float = 0.0
    scan_plan: str | None = None
    classification: str | None = None
    timestamp: float

def bind():
    router = APIRouter(prefix="/api/tripwire", tags=["tripwire"])

    @router.post("/event")
    async def tripwire_event(payload: dict, request: Request):
        print("[INGEST] Tripwire EVENT received:", payload)
        orch = request.app.state.orchestrator
        client_ip = request.client.host if request.client else "unknown"

        # -------------------------------------------------
        # DO NOT update node registry from HTTP events
        # The WebSocket hello/heartbeat is the authoritative source
        # HTTP events should ONLY trigger cues/captures
        # -------------------------------------------------
        # orch.tripwires.update_from_event(payload, client_ip)
        # orch.bus.publish_nowait("tripwire_nodes", {"nodes": orch.tripwires.snapshot()})

        # -------------------------------------------------
        # Always record cue and publish to UI (advisory only)
        # Per Tripwire v1.1 alignment:
        # - Cues are advisory notifications for operator review
        # - Edge is the authority; Tripwire does not assert intent or threat
        # - Cues never trigger auto-capture, task creation, or ATAK forwarding
        # -------------------------------------------------
        orch.record_tripwire_cue(payload)
        orch.bus.publish_nowait("tripwire_cue", payload)

        # -------------------------------------------------
        # MANUAL MODE → advisory only (cue appears for operator review)
        # Cues are always advisory, regardless of mode
        # -------------------------------------------------
        if orch.mode != "armed":
            return {
                "accepted": True,
                "action": "queued_for_operator"
            }

        # -------------------------------------------------
        # ARMED MODE → policy check
        # -------------------------------------------------
        allowed, reason = orch.can_auto_capture(payload)
        if not allowed:
            orch.bus.publish_nowait(
                "tripwire_auto_reject",
                {"cue": payload, "reason": reason}
            )
            return {
                "accepted": True,
                "action": "rejected",
                "reason": reason
            }

        orch.mark_auto_capture(payload)

        # -------------------------------------------------
        # Queue capture (non-blocking)
        # Convert tripwire payload to CaptureRequest
        # Tripwire payload structure (per v1.1 alignment):
        #   - type: event discriminator (rf_cue, fhss_cluster, etc.)
        #   - stage: detection stage (energy, cue, confirmed)
        #   - freq_hz, bandwidth_hz, delta_db, level_db
        #   - classification, confidence, scan_plan
        #   - node_id, callsign, timestamp
        #   - confidence_source, hypothesis (if present)
        #   - NO sample_rate_sps or duration_s (use defaults)
        # -------------------------------------------------
        from spear_edge.core.bus.models import CaptureRequest
        
        # Normalize event type: Tripwire emits "type", not "event_type"
        event_type = payload.get("type") or payload.get("event_type")  # Support transition
        
        # Extract fields from tripwire payload
        freq_hz = float(payload.get("freq_hz", 0.0))
        if freq_hz <= 0:
            print(f"[TRIPWIRE] Invalid freq_hz in payload: {freq_hz}")
            return {
                "accepted": False,
                "action": "invalid_freq",
                "reason": "freq_hz missing or invalid"
            }
        
        # Use current SDR config for sample_rate if available, else safe default
        sdr_config = getattr(orch, "sdr_config", None)
        sample_rate_sps = int(sdr_config.sample_rate_sps) if sdr_config and sdr_config.sample_rate_sps else 10_000_000
        
        # Default capture duration (5 seconds)
        duration_s = 5.0
        
        # Extract tripwire metadata
        node_id = str(payload.get("node_id", "unknown"))
        callsign = payload.get("callsign")
        scan_plan = payload.get("scan_plan")
        bandwidth_hz = payload.get("bandwidth_hz")  # May be 0.0 or None
        if bandwidth_hz and float(bandwidth_hz) > 0:
            bandwidth_hz = int(float(bandwidth_hz))
        else:
            bandwidth_hz = None  # Will use sample_rate as default in tune()
        
        # Preserve all metadata fields per Tripwire v1.1 alignment requirements
        # These are critical for debugging, multi-node correlation, and analyst review
        capture_req = CaptureRequest(
            ts=time.time(),
            reason="tripwire_armed",
            freq_hz=freq_hz,
            sample_rate_sps=sample_rate_sps,
            duration_s=duration_s,
            rx_channel=0,
            scan_plan=scan_plan,  # Preserved end-to-end
            priority=60,  # Higher priority than manual captures
            source_node=node_id,
            meta={
                # Event identification
                "type": event_type,  # Normalized: use "type" not "event_type"
                "stage": payload.get("stage"),  # Detection stage: energy, cue, confirmed
                # Confidence and classification
                "confidence": payload.get("confidence"),
                "confidence_source": payload.get("confidence_source"),  # Preserved
                "classification": payload.get("classification"),  # Preserved
                "hypothesis": payload.get("hypothesis"),  # Preserved if present
                # Timing
                "timestamp": payload.get("timestamp"),
                # RF metrics
                "delta_db": payload.get("delta_db"),
                "level_db": payload.get("level_db"),
                "bandwidth_hz": bandwidth_hz,
                # Node identification
                "callsign": callsign,
                # Additional context
                "remarks": payload.get("remarks"),
            },
        )
        
        ok = orch.capture_mgr.submit_nowait(capture_req)

        return {
            "accepted": ok,
            "action": "auto_capture_started" if ok else "queue_full"
        }

    # -------------------------------------------------
    # Legacy endpoint (keep for backward compatibility)
    # -------------------------------------------------
    @router.post("/cue")
    async def tripwire_cue_legacy(payload: dict, request: Request):
        return await tripwire_event(payload, request)

    @router.get("/ping")
    async def tripwire_ping():
        return {"ok": True}

    @router.post("/scan-plan")
    async def set_scan_plan(payload: dict, request: Request):
        """
        Send scan plan command to a specific tripwire node.
        Forwards the command via WebSocket if the node is connected.
        """
        orch = request.app.state.orchestrator
        node_id = payload.get("node_id")
        scan_plan = payload.get("scan_plan")
        
        if not node_id or not scan_plan:
            return {
                "ok": False,
                "error": "missing_fields",
                "detail": "node_id and scan_plan are required"
            }
        
        # Get WebSocket connection for this node
        links = getattr(orch, "tripwire_links", None)
        if not isinstance(links, dict):
            return {
                "ok": False,
                "error": "no_connections",
                "detail": "No tripwire connections available"
            }
        
        ws = links.get(node_id)
        if not ws:
            return {
                "ok": False,
                "error": "node_not_connected",
                "detail": f"Node {node_id} is not connected"
            }
        
        # Send scan plan command via WebSocket
        import json
        try:
            msg = {
                "type": "set_scan_plan",
                "scan_plan": scan_plan
            }
            await ws.send_text(json.dumps(msg))
            print(f"[TRIPWIRE] Sent scan plan {scan_plan} to {node_id}")
            return {
                "ok": True,
                "node_id": node_id,
                "scan_plan": scan_plan
            }
        except Exception as e:
            print(f"[TRIPWIRE] Error sending scan plan to {node_id}: {e}")
            return {
                "ok": False,
                "error": "send_failed",
                "detail": str(e)
            }

    return router