from fastapi import APIRouter, Request
from pydantic import BaseModel
import time

from spear_edge.core.integrate.tripwire_events import (
    parse_tripwire_event,
    RfCueEvent,
    AoaConeEvent,
    FhssClusterEvent,
    RfEnergyEvent,
    RfSpikeEvent,
    BearingLineEvent,
)

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
        # Parse event using v2.0 schemas (with backward compatibility)
        # -------------------------------------------------
        try:
            parsed_event = parse_tripwire_event(payload)
            event_type = payload.get("type") or payload.get("event_type", "")
            print(f"[INGEST] Parsed event type: {event_type}")
        except Exception as e:
            print(f"[INGEST] Error parsing event (using raw payload): {e}")
            parsed_event = None
            event_type = payload.get("type") or payload.get("event_type", "")

        # -------------------------------------------------
        # Handle special event types that don't trigger captures
        # -------------------------------------------------
        # Handle nested rf_event format (v2.0 compatibility)
        # {"type": "rf_event", "event_type": "fhss_cluster", ...}
        if event_type == "rf_event":
            nested_type = payload.get("event_type", "")
            if nested_type:
                event_type = nested_type
                print(f"[INGEST] Nested rf_event, actual type: {event_type}")
        
        # AoA cone events are continuous updates - advisory only, but used for TAI
        if event_type == "aoa_cone":
            node_id = payload.get("node_id") or payload.get("tripwire_id") or "unknown"
            bearing_deg = payload.get("bearing_deg")
            
            # Get callsign from registry if available
            nodes = orch.tripwires.snapshot()
            node = next((n for n in nodes if n["node_id"] == node_id), None)
            callsign = node.get("callsign") if node else payload.get("callsign")
            
            # Ensure source_type is set for fusion tracking
            aoa_data = dict(payload)
            aoa_data["source_type"] = "aoa_auto"  # Automated AoA from antenna array
            aoa_data["node_id"] = node_id
            aoa_data["callsign"] = callsign
            
            # Use GPS from payload or node registry
            if not aoa_data.get("gps"):
                if node and node.get("gps"):
                    aoa_data["gps"] = node["gps"]
            
            print(f"[INGEST] AoA cone (Auto) from {node_id}: {bearing_deg}° - storing for TAI")
            orch.record_tripwire_cue(aoa_data)
            orch.bus.publish_nowait("tripwire_cue", aoa_data)
            return {
                "accepted": True,
                "action": "aoa_update",
                "source_type": "aoa_auto",
                "message": "Auto AoA cone stored for triangulation"
            }
        
        # Bearing line events for triangulation - advisory, stored for TAI calculation
        if event_type == "bearing_line":
            print("[INGEST] Bearing line event (Manual DF) - storing for triangulation")
            # Parse into structured event
            try:
                bearing_event = BearingLineEvent(**payload)
                node_id = bearing_event.get_node_id()
                
                # Get callsign from registry if available
                nodes = orch.tripwires.snapshot()
                node = next((n for n in nodes if n["node_id"] == node_id), None)
                callsign = node.get("callsign") if node else None
                
                # Store bearing data with GPS for triangulation
                # Mark as manual_df source for fusion tracking
                bearing_data = {
                    "type": "bearing_line",
                    "source_type": "manual_df",  # Manual DF bearing
                    "node_id": node_id,
                    "callsign": callsign,
                    "bearing_deg": bearing_event.bearing_deg,
                    "confidence": bearing_event.confidence or 0.7,  # Manual DF default confidence
                    "signal_strength_db": bearing_event.signal_strength_db,
                    "null_bearing_deg": bearing_event.null_bearing_deg,
                    "cone_width_deg": bearing_event.cone_width_deg,  # May be None
                    "bearing_std_deg": bearing_event.bearing_std_deg,  # Will be converted if cone_width missing
                    "gps": bearing_event.gps,
                    "timestamp": bearing_event.timestamp,
                    "signal_freq_mhz": bearing_event.signal_freq_mhz,
                }
                # Record as AoA cone for TAI calculation (bearing lines are similar to AoA)
                orch._record_aoa_cone(bearing_data)
                print(f"[INGEST] Manual DF bearing from {node_id}: {bearing_event.bearing_deg}° (conf={bearing_event.confidence})")
            except Exception as e:
                print(f"[INGEST] Error parsing bearing_line: {e}")
            
            orch.record_tripwire_cue(payload)
            orch.bus.publish_nowait("tripwire_cue", payload)
            orch.bus.publish_nowait("bearing_line", payload)
            return {
                "accepted": True,
                "action": "bearing_stored",
                "source_type": "manual_df",
                "message": "Manual DF bearing stored for triangulation"
            }
        
        # RF energy start/end events are tracking events - advisory only
        if event_type in ("rf_energy_start", "rf_energy_end"):
            print(f"[INGEST] RF energy {event_type} - advisory only, no capture")
            orch.record_tripwire_cue(payload)
            orch.bus.publish_nowait("tripwire_cue", payload)
            return {
                "accepted": True,
                "action": "energy_tracking",
                "message": "RF energy events are tracking only"
            }
        
        # RF spike events are low-confidence - advisory only
        if event_type == "rf_spike":
            print("[INGEST] RF spike event - low confidence, advisory only")
            orch.record_tripwire_cue(payload)
            orch.bus.publish_nowait("tripwire_cue", payload)
            return {
                "accepted": True,
                "action": "advisory_only",
                "message": "RF spike events are advisory only"
            }
        
        # IBW calibration and DF events are system events - no capture
        if event_type.startswith("ibw_calibration_") or event_type in ("df_metrics", "df_bearing"):
            print(f"[INGEST] System event {event_type} - no capture")
            return {
                "accepted": True,
                "action": "system_event",
                "message": "System events do not trigger captures"
            }

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
        
        # Extract tripwire metadata (support both v1.1 and v2.0)
        node_id = str(payload.get("node_id", "unknown"))
        callsign = payload.get("callsign")
        scan_plan = payload.get("scan_plan")
        bandwidth_hz = payload.get("bandwidth_hz")  # May be 0.0 or None
        if bandwidth_hz and float(bandwidth_hz) > 0:
            bandwidth_hz = int(float(bandwidth_hz))
        else:
            bandwidth_hz = None  # Will use sample_rate as default in tune()
        
        # Extract v2.0 specific fields if present
        event_id = payload.get("event_id")
        event_group_id = payload.get("event_group_id")
        gps_lat = payload.get("gps_lat")
        gps_lon = payload.get("gps_lon")
        gps_alt = payload.get("gps_alt")
        heading_deg = payload.get("heading_deg")
        
        # For FHSS cluster events, extract additional fields
        hop_count = payload.get("hop_count")
        span_mhz = payload.get("span_mhz")
        unique_buckets = payload.get("unique_buckets")
        
        # Preserve all metadata fields per Tripwire v1.1 and v2.0 alignment requirements
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
                "stage": payload.get("stage"),  # Detection stage: energy, cue, confirmed (v1.1)
                "event_id": event_id,  # v2.0 event ID
                "event_group_id": event_group_id,  # v2.0 event group ID
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
                # GPS and location (v2.0)
                "gps_lat": gps_lat,
                "gps_lon": gps_lon,
                "gps_alt": gps_alt,
                "heading_deg": heading_deg,
                # FHSS cluster fields (v2.0)
                "hop_count": hop_count,
                "span_mhz": span_mhz,
                "unique_buckets": unique_buckets,
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

    @router.get("/aoa-fusion")
    async def get_aoa_fusion(request: Request):
        """
        Get AoA fusion data: active AoA cones with node GPS positions,
        and calculated TAI (Targeted Area of Interest) if available.
        Returns cones from up to 4 tripwire nodes for triangulation.
        """
        from spear_edge.core.integrate.aoa_fusion import (
            cones_from_dicts, fuse_bearing_cones, tai_to_dict
        )
        
        orch = request.app.state.orchestrator
        now = time.time()
        
        # Get active AoA cones (most recent from each node)
        active_cones = []
        node_ids_seen = set()
        
        # Get most recent cone from each node (reverse order, newest first)
        for cone in reversed(orch.aoa_cones):
            # Extract node_id from various possible fields
            node_id = cone.get("node_id") or cone.get("source_node") or cone.get("callsign")
            if not node_id:
                continue
            
            # Only take one cone per node (most recent)
            if node_id in node_ids_seen:
                continue
            
            # Only include recent cones (within last 60 seconds)
            cone_timestamp = cone.get("timestamp", 0)
            if now - cone_timestamp > 60:
                continue
            
            # Ensure node_id is in the cone dict for frontend
            cone_copy = dict(cone)
            cone_copy["node_id"] = node_id
            
            node_ids_seen.add(node_id)
            active_cones.append(cone_copy)
            
            # Limit to 4 nodes
            if len(active_cones) >= 4:
                break
        
        # Get node GPS positions and add to cones
        nodes = orch.tripwires.snapshot()
        node_map = {n["node_id"]: n for n in nodes}
        
        # Combine cones with node GPS data
        fusion_data = []
        cones_for_fusion = []
        
        for cone in active_cones:
            node_id = cone.get("node_id")
            node = node_map.get(node_id)
            
            # Get GPS from cone or node registry
            gps = cone.get("gps")
            if not gps or not gps.get("lat"):
                gps = node.get("gps") if node else None
            
            # Determine source type
            source_type = cone.get("source_type", "unknown")
            event_type_hint = cone.get("type") or cone.get("event_type", "")
            if source_type == "unknown":
                if event_type_hint == "bearing_line":
                    source_type = "manual_df"
                elif event_type_hint == "aoa_cone":
                    source_type = "aoa_auto"
            
            source_labels = {
                "manual_df": "Manual DF",
                "aoa_auto": "Auto AoA",
                "bearing_line": "Bearing Line",
                "unknown": "Unknown",
            }
            
            fusion_item = {
                "node_id": node_id,
                "callsign": cone.get("callsign") or (node.get("callsign", node_id) if node else node_id),
                "bearing_deg": cone.get("bearing_deg"),
                "cone_width_deg": cone.get("cone_width_deg", 30.0),
                "bearing_std_deg": cone.get("bearing_std_deg"),  # Include if available
                "confidence": cone.get("confidence", 0.5),
                "timestamp": cone.get("timestamp"),
                "gps": gps,
                "source_type": source_type,
                "source_label": source_labels.get(source_type, "Unknown"),
            }
            fusion_data.append(fusion_item)
            
            # Build cone dict for fusion calculation
            if gps and gps.get("lat") and cone.get("bearing_deg") is not None:
                cones_for_fusion.append({
                    "node_id": node_id,
                    "callsign": fusion_item["callsign"],
                    "gps": gps,
                    "bearing_deg": cone.get("bearing_deg"),
                    "cone_width_deg": cone.get("cone_width_deg"),  # May be None, fusion will handle
                    "bearing_std_deg": cone.get("bearing_std_deg"),  # Alternative uncertainty
                    "confidence": cone.get("confidence", 0.5),
                    "timestamp": cone.get("timestamp", 0),
                    "source_type": source_type,
                    "type": event_type_hint,  # For source inference
                })
        
        # Calculate TAI if we have enough cones
        tai_result = None
        if len(cones_for_fusion) >= 2:
            try:
                bearing_cones = cones_from_dicts(cones_for_fusion)
                if len(bearing_cones) >= 2:
                    tai = fuse_bearing_cones(bearing_cones)
                    if tai.valid:
                        tai_result = tai_to_dict(tai)
            except Exception as e:
                print(f"[AOA-FUSION] Error calculating TAI: {e}")
        
        # Summarize sources
        sources_summary = {}
        for cone in fusion_data:
            src = cone.get("source_type", "unknown")
            sources_summary[src] = sources_summary.get(src, 0) + 1
        
        return {
            "cones": fusion_data,
            "count": len(fusion_data),
            "sources_summary": sources_summary,
            "timestamp": now,
            "tai": tai_result,
        }
    
    @router.get("/tai")
    async def get_tai(request: Request):
        """
        Get the current TAI (Targeted Area of Interest) result.
        Returns the most recent triangulation result if available.
        """
        orch = request.app.state.orchestrator
        
        # Check if we have a recent TAI
        tai_dict = getattr(orch, "_last_tai_dict", None)
        tai_ts = getattr(orch, "_last_tai_ts", 0)
        
        if tai_dict is None:
            return {
                "valid": False,
                "error": "No TAI calculated yet",
                "timestamp": None,
            }
        
        # Check staleness (TAI older than 60 seconds is stale)
        now = time.time()
        stale = (now - tai_ts) > 60
        
        return {
            "valid": tai_dict.get("valid", False) and not stale,
            "stale": stale,
            "age_s": now - tai_ts,
            "timestamp": tai_ts,
            **tai_dict,
        }

    @router.post("/scan-plan")
    async def set_scan_plan(payload: dict, request: Request):
        """
        Send scan plan command to a specific tripwire node.
        Forwards the command via WebSocket if the node is connected.
        Includes command_id for response correlation per Tripwire v2.0 spec.
        """
        import uuid
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
        
        # Generate command_id for response correlation
        command_id = f"cmd-{uuid.uuid4().hex[:12]}"
        
        # Initialize pending commands tracking if not exists
        if not hasattr(orch, "_pending_commands"):
            orch._pending_commands = {}
        
        # Track pending command
        orch._pending_commands[command_id] = {
            "node_id": node_id,
            "action": "set_scan_plan",
            "scan_plan": scan_plan,
            "sent_at": time.time(),
            "status": "pending",
            "response": None,
            "completed_at": None,
        }
        
        # Send scan plan command via WebSocket (v2.0 format with command_id)
        import json
        try:
            msg = {
                "type": "set_scan_plan",
                "scan_plan": scan_plan,
                "command_id": command_id,
            }
            await ws.send_text(json.dumps(msg))
            print(f"[TRIPWIRE] Sent scan plan {scan_plan} to {node_id} (command_id={command_id})")
            return {
                "ok": True,
                "node_id": node_id,
                "scan_plan": scan_plan,
                "command_id": command_id,
            }
        except Exception as e:
            print(f"[TRIPWIRE] Error sending scan plan to {node_id}: {e}")
            # Mark command as failed
            orch._pending_commands[command_id]["status"] = "send_failed"
            orch._pending_commands[command_id]["completed_at"] = time.time()
            return {
                "ok": False,
                "error": "send_failed",
                "detail": str(e),
                "command_id": command_id,
            }
    
    @router.get("/command/{command_id}")
    async def get_command_status(command_id: str, request: Request):
        """
        Get status of a pending command by command_id.
        """
        orch = request.app.state.orchestrator
        
        if not hasattr(orch, "_pending_commands"):
            return {
                "ok": False,
                "error": "not_found",
                "detail": "No pending commands"
            }
        
        command = orch._pending_commands.get(command_id)
        if not command:
            return {
                "ok": False,
                "error": "not_found",
                "detail": f"Command {command_id} not found"
            }
        
        return {
            "ok": True,
            "command_id": command_id,
            **command
        }

    return router