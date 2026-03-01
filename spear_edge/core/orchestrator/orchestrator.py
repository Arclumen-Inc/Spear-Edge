"""
Central runtime owner of SDR hardware.
Responsibilities:
- Own SDR device and scan lifecycle
- Publish live FFT frames
- Maintain operator mode
- Record Tripwire cues (advisory only)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from spear_edge.core.scan.ring_buffer import IQRingBuffer
from spear_edge.core.scan.rx_task import RxTask
from spear_edge.core.capture.capture_manager import CaptureManager  # ← Added here (top-level import)

import asyncio
import json
import time
from typing import Optional, Dict, List, Any

from spear_edge.core.sdr.base import SdrConfig, GainMode
from spear_edge.core.scan.scan_task import ScanTask
from spear_edge.core.bus.event_bus import EventBus
from spear_edge.core.bus.models import LiveSpectrumFrame
from spear_edge.core.classify.pipeline import ClassifierPipeline
from spear_edge.core.integrate.cot import CoTBroadcaster
from spear_edge.core.integrate.tripwire_registry import TripwireRegistry

import spear_edge.core.integrate.tripwire_registry as _twreg
print("[DEBUG] TripwireRegistry loaded from:", _twreg.__file__)


# Tripwire connection threshold (matches UI logic)
CONNECTED_SECS = 5.0

@dataclass
class AutoCapturePolicy:
    enabled: bool = True  # only used when mode == "armed"
    min_confidence: float = 0.90  # gate low-confidence spam
    global_cooldown_s: float = 3.0  # minimum time between any auto captures
    per_node_cooldown_s: float = 2.0  # minimum time between captures from same node
    per_freq_cooldown_s: float = 8.0  # minimum time between captures in same freq bin
    freq_bin_hz: int = 100_000  # binning to prevent near-same freq spam
    max_captures_per_min: int = 10  # global rate limit


class Orchestrator:
    """
    Single authoritative owner of SDR hardware.
    Modes:
      - manual : operator control only
      - armed  : automatic capture allowed (guarded)
      - tasked : internal, transient (never set directly)
    
    Tripwire Integration (v1.1 alignment):
    - Edge is the authority for capture decisions
    - Tripwire cues are advisory only - they do not assert intent or threat
    - Only confirmed events (stage="confirmed") are eligible for auto-capture
    - Cues (type="rf_cue" or stage="cue") are never actionable
    """

    def __init__(self, sdr):
        # Hardware
        self.sdr = sdr
        self._open = False

        # Scan state
        self._scan: Optional[ScanTask] = None
        self._last_frame_ts: Optional[float] = None

        # Concurrency
        self._lock = asyncio.Lock()

        # Event bus (live FFT, capture results, etc.)
        self.bus = EventBus()

        # Operator mode
        self.mode: str = "manual"  # manual | armed | tasked

        # Task / job placeholders (future)
        self._job = None
        self.task_info = None

        # Tripwire cues (advisory only - never actionable)
        # Per Tripwire v1.1: cues are ephemeral, advisory notifications
        # They appear in UI for operator review but never trigger auto-capture
        self.tripwire_cues: List[Dict[str, Any]] = []
        self.max_tripwire_cues: int = 10
        self.tripwires = TripwireRegistry(max_nodes=3)
        
        # AoA cone tracking (v2.0)
        # Store recent AoA cone events for bearing visualization
        self.aoa_cones: List[Dict[str, Any]] = []
        self.max_aoa_cones: int = 20  # Store more AoA events (they're continuous updates)

        # Auto-capture policy (used only in ARMED mode)
        self.auto_policy = AutoCapturePolicy()

        # Auto-capture guard state
        self._last_auto_ts: float = 0.0
        self._last_auto_by_node: dict[str, float] = {}
        self._last_auto_by_freqbin: dict[int, float] = {}

        # Simple sliding-window rate limiter (timestamps of recent captures)
        self._auto_history: list[float] = []

        # ATAK status message tracking
        self._last_atak_tripwire_count: int = -1  # Track last sent count for updates

        # Storage
        self.capture_log: list[dict] = []

        # SDR Config
        self.sdr_config = None

        # Ring Buffer + RX task
        self._ring = None
        self._rx_task = None

        # ----------------------------------------
        # Capture Manager
        # ----------------------------------------
        self.capture_mgr = CaptureManager(orchestrator=self)
        # Hook up callbacks for classification + ATAK alerting
        self.capture_mgr.on_result = self._on_capture_result
        self.capture_mgr.on_log = self._log_capture_result
        # Start the capture worker if it has a start method
        if hasattr(self.capture_mgr, "start"):
            asyncio.create_task(self.capture_mgr.start())

        # ----------------------------------------
        # ML classifier + ATAK CoT broadcaster
        # ----------------------------------------
        self.classifier = ClassifierPipeline()

        self.cot = CoTBroadcaster(
            uid="SPEAR-EDGE",
            callsign="SPEAR-EDGE",
        )
        self.cot.start()
        
        # Subscribe to tripwire_nodes events to update ATAK status when count changes
        self._setup_tripwire_status_updates()

    # -------------------------------------------------
    # SDR lifecycle
    # -------------------------------------------------
    async def open(self) -> None:
        if self._open:
            return
        self.sdr.open()
        self._open = True

    async def close(self) -> None:
        async with self._lock:
            if self._scan and self._scan.is_running():
                await self._scan.stop()
            if self._open:
                self.sdr.close()
                self._open = False

    # -------------------------------------------------
    # Scan control (RX task + ring buffer)
    # -------------------------------------------------
    async def start_scan(
        self,
        center_freq_hz: int,
        sample_rate_sps: int,
        fft_size: int,
        fps: float,
    ) -> None:
        async with self._lock:
            # Ensure SDR open
            await self.open()

            # Build + store SDR config (for /sdr/info)
            # SANITY PROFILE (lab/near TX): Safe defaults to prevent overdriving
            # - BT200: OFF (external LNA not connected, adds ~16-20 dB gain)
            # - LNA: 0 dB (internal LNA gain off for headroom)
            # - System gain: 10 dB (conservative starting point, can be increased if needed)
            # 
            # If samples mean > 0.15-0.20 or max > 0.8, reduce gain further
            # Target: samples mean ~0.02-0.10, max < 0.8, noise floor -70 to -95 dBFS
            prev_bt200 = getattr(self.sdr_config, 'bt200_enabled', None) if self.sdr_config else None
            # Safety: Ensure BT200 is explicitly False (not None) if not previously configured
            # BT200 should never be enabled by default (hardware not connected, adds too much gain)
            bt200_default = False if prev_bt200 is None else prev_bt200
            # LNA gain is now automatically optimized by bladerf_set_gain() - no manual control needed
            
            self.sdr_config = SdrConfig(
                center_freq_hz=center_freq_hz,
                sample_rate_sps=sample_rate_sps,
                gain_mode=self.sdr_config.gain_mode if self.sdr_config else GainMode.MANUAL,
                gain_db=self.sdr_config.gain_db if self.sdr_config else 0.0,  # Default 0 dB - user can adjust via UI slider
                rx_channel=self.sdr_config.rx_channel if self.sdr_config else 0,
                bandwidth_hz=self.sdr_config.bandwidth_hz if self.sdr_config else None,
                # LNA gain is now automatically optimized by bladerf_set_gain() - no manual control needed
                bt200_enabled=bt200_default,  # Explicitly False by default (BT200 not connected)
                dual_channel=getattr(self.sdr_config, 'dual_channel', False) if self.sdr_config else False,  # Single channel by default
            )

            # Apply SDR config atomically
            try:
                self.sdr.apply_config(self.sdr_config)
            except Exception as e:
                print(f"[ORCH] apply_config failed: {e}")

            # Stop existing scan if running
            if self._scan and self._scan.is_running():
                await self._scan.stop()
            if self._rx_task and self._rx_task.is_running():
                await self._rx_task.stop()
            
            # Clear FFT params when stopping scan
            self._current_fft_size = None
            self._current_fps = None

            # RX drain path (line rate)
            ring_size = int(sample_rate_sps * 0.5)
            self._ring = IQRingBuffer(ring_size)

            # Aim for ~10–25 ms of samples per read (good tradeoff)
            # chunk = sample_rate * 0.02  (20 ms)
            chunk = int(sample_rate_sps * 0.02)

            # clamp to reasonable range
            chunk = max(chunk, 32768)
            chunk = min(chunk, 262144)

            self._rx_task = RxTask(
                sdr=self.sdr,
                ring=self._ring,
                chunk_size=chunk,
            )
            await self._rx_task.start()

            # FFT task (FPS limited)
            # No calibration offset - using raw bladeRF values
            self._scan = ScanTask(
                ring=self._ring,
                center_freq_hz=center_freq_hz,
                sample_rate_sps=sample_rate_sps,
                fft_size=fft_size,
                fps=fps,
                calibration_offset_db=0.0,  # No offset - raw bladeRF values
            )
            
            # Store FFT params for status() to retrieve
            self._current_fft_size = fft_size
            self._current_fps = fps

            def _on_frame(frame: Dict[str, Any]) -> None:
                self._last_frame_ts = frame.get("ts", time.time())
                try:
                    evt = LiveSpectrumFrame(
                        ts=frame["ts"],
                        center_freq_hz=frame["center_freq_hz"],
                        sample_rate_sps=frame["sample_rate_sps"],
                        fft_size=frame["fft_size"],
                        freqs_hz=frame.get("freqs_hz"),  # Optional - client can compute
                        power_dbfs=frame["power_dbfs"],
                        power_inst_dbfs=frame.get("power_inst_dbfs"),
                        noise_floor_dbfs=frame.get("noise_floor_dbfs"),
                        calibration_offset_db=frame.get("calibration_offset_db"),  # Calibration metadata
                        power_units=frame.get("power_units"),  # "dBm" or "dBFS"
                    )
                    self.bus.publish_nowait("live_spectrum", evt)
                except Exception:
                    # Live FFT must never crash runtime
                    pass

            self._scan.subscribe(_on_frame)
            await self._scan.start()

    async def stop_scan(self) -> None:
        async with self._lock:
            print("[ORCH] stop_scan() called")
            
            if self._scan and self._scan.is_running():
                print("[ORCH] Stopping scan task...")
                await self._scan.stop()
                self._scan = None
                print("[ORCH] Scan task stopped")

            if self._rx_task and self._rx_task.is_running():
                print("[ORCH] Stopping RX task...")
                await self._rx_task.stop()
                self._rx_task = None
                print("[ORCH] RX task stopped")

            self._ring = None
            
            # CRITICAL: Deactivate SDR stream to free hardware resources
            # This prevents stream from being in a bad state for captures
            # BladeRFNativeDevice uses _stream_active flag, SoapySDR uses rx_stream object
            if hasattr(self.sdr, "_stream_active") and self.sdr._stream_active:
                print("[ORCH] Deactivating SDR stream (BladeRFNativeDevice)...")
                try:
                    if hasattr(self.sdr, "_deactivate_stream"):
                        self.sdr._deactivate_stream()
                        print("[ORCH] SDR stream deactivated")
                except Exception as e:
                    print(f"[ORCH] Error deactivating stream: {e}")
            elif hasattr(self.sdr, "rx_stream") and self.sdr.rx_stream is not None:
                print("[ORCH] Deactivating SDR stream (SoapySDR)...")
                try:
                    if hasattr(self.sdr, "dev") and self.sdr.dev is not None:
                        self.sdr.dev.deactivateStream(self.sdr.rx_stream)
                        print("[ORCH] SDR stream deactivated")
                except Exception as e:
                    print(f"[ORCH] Error deactivating stream: {e}")
            
            print("[ORCH] stop_scan() completed")

    # -------------------------------------------------
    # Mode control
    # -------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """
        Set operator mode.
        Only manual or armed are allowed externally.
        """
        if mode not in ("manual", "armed"):
            raise ValueError(f"invalid mode: {mode}")
        
        old_mode = self.mode
        self.mode = mode

        # Notify UI via event bus
        try:
            self.bus.publish_nowait("edge_mode", {"mode": self.mode, "ts": time.time()})
        except Exception:
            pass

        # Notify connected tripwires (WS)
        links = getattr(self, "tripwire_links", None)
        if isinstance(links, dict):
            msg = {"type": "edge_state", "mode": self.mode}
            for _node, ws in list(links.items()):
                try:
                    # fire-and-forget
                    asyncio.create_task(ws.send_text(json.dumps(msg)))
                except Exception:
                    pass
        
        # Send ATAK status messages
        if mode == "armed":
            count = self._count_connected_tripwires()
            self._send_atak_status(online=True, tripwire_count=count)
        elif old_mode == "armed" and mode != "armed":
            # Transitioning away from armed
            self._send_atak_status(online=False)

    # -------------------------------------------------
    # Tripwire cues (advisory only - never actionable)
    # Per Tripwire v1.1 alignment: cues are ephemeral, stored in short rolling buffer
    # They appear in UI for operator review but never trigger auto-capture or ATAK forwarding
    # -------------------------------------------------
    def record_tripwire_cue(self, cue: Dict[str, Any]) -> None:
        """
        Record a Tripwire detection for operator awareness.
        
        Per Tripwire v1.1 alignment:
        - Cues are advisory only - they never trigger auto-capture, task creation, or ATAK forwarding
        - Edge is the authority for capture decisions
        - Tripwire does not assert intent or threat
        - Cues are stored in short rolling buffer for UI display only
        """
        # Special handling for AoA cone events (v2.0)
        event_type = cue.get("type") or cue.get("event_type", "")
        if event_type == "aoa_cone":
            self._record_aoa_cone(cue)
            return
        
        self.tripwire_cues.append(cue)
        if len(self.tripwire_cues) > self.max_tripwire_cues:
            self.tripwire_cues.pop(0)
    
    def _record_aoa_cone(self, cone: Dict[str, Any]) -> None:
        """
        Record an AoA cone event for bearing tracking.
        AoA cones are continuous updates, so we maintain a larger buffer.
        """
        # Add timestamp if not present
        if "timestamp" not in cone:
            cone["timestamp"] = time.time()
        
        self.aoa_cones.append(cone)
        if len(self.aoa_cones) > self.max_aoa_cones:
            self.aoa_cones.pop(0)
        
        # Publish AoA update to UI
        self.bus.publish_nowait("aoa_cone", cone)
        
        # Check if we have enough cones for TAI calculation and send to ATAK
        self._update_tai_if_ready()

    def _update_tai_if_ready(self) -> None:
        """
        Calculate TAI from active AoA cones and send to ATAK if we have 2+ cones with GPS.
        """
        import time
        from spear_edge.core.integrate.tripwire_events import AoaConeEvent
        
        now = time.time()
        
        # Get active cones (recent, with GPS and bearing)
        active_cones = []
        node_ids_seen = set()
        
        for cone in reversed(self.aoa_cones):
            node_id = cone.get("node_id") or cone.get("source_node")
            if not node_id:
                continue
            
            # Only take one cone per node (most recent)
            if node_id in node_ids_seen:
                continue
            
            # Only include recent cones (within last 60 seconds)
            cone_timestamp = cone.get("timestamp", 0)
            if now - cone_timestamp > 60:
                continue
            
            # Must have GPS and bearing
            if not cone.get("bearing_deg") is not None:
                continue
            
            # Get node GPS from registry
            nodes = self.tripwires.snapshot()
            node = next((n for n in nodes if n["node_id"] == node_id), None)
            if not node or not node.get("gps") or not node["gps"].get("lat"):
                continue
            
            # Add GPS to cone data
            cone_with_gps = dict(cone)
            cone_with_gps["gps"] = node["gps"]
            
            node_ids_seen.add(node_id)
            active_cones.append(cone_with_gps)
            
            if len(active_cones) >= 3:
                break
        
        # Need at least 2 cones for triangulation
        if len(active_cones) < 2:
            return
        
        # Calculate intersections
        intersections = []
        for i in range(len(active_cones)):
            for j in range(i + 1, len(active_cones)):
                cone1 = active_cones[i]
                cone2 = active_cones[j]
                
                # Calculate intersection (simplified - using bearing lines)
                # This is a simplified calculation - full implementation would use proper triangulation
                lat1 = cone1["gps"]["lat"]
                lon1 = cone1["gps"]["lon"]
                bearing1 = cone1["bearing_deg"]
                lat2 = cone2["gps"]["lat"]
                lon2 = cone2["gps"]["lon"]
                bearing2 = cone2["bearing_deg"]
                
                # Simple intersection calculation (for now - could be improved)
                # This is a placeholder - proper implementation would use spherical geometry
                try:
                    # Use a simplified approach: calculate point along bearing from each node
                    # and find intersection (this is approximate)
                    import math
                    R = 6371000  # Earth radius in meters
                    
                    # Convert to radians
                    lat1_rad = math.radians(lat1)
                    lon1_rad = math.radians(lon1)
                    lat2_rad = math.radians(lat2)
                    lon2_rad = math.radians(lon2)
                    brg1_rad = math.radians(bearing1)
                    brg2_rad = math.radians(bearing2)
                    
                    # Calculate intersection using plane approximation (for small distances)
                    # This is simplified - full implementation would use great circle intersection
                    d_lat = lat2_rad - lat1_rad
                    d_lon = lon2_rad - lon1_rad
                    
                    # Approximate intersection (simplified)
                    # For proper implementation, use Vincenty's formula or similar
                    # For now, use midpoint as approximation when bearings are close
                    if abs(bearing1 - bearing2) > 5:  # Only if bearings differ significantly
                        # Simplified: use average of node positions weighted by confidence
                        conf1 = cone1.get("confidence", 0.5)
                        conf2 = cone2.get("confidence", 0.5)
                        total_conf = conf1 + conf2
                        if total_conf > 0:
                            avg_lat = (lat1 * conf1 + lat2 * conf2) / total_conf
                            avg_lon = (lon1 * conf1 + lon2 * conf2) / total_conf
                            intersections.append({"lat": avg_lat, "lon": avg_lon})
                except Exception as e:
                    print(f"[TAI] Error calculating intersection: {e}")
                    continue
        
        if not intersections:
            return
        
        # Average intersections for TAI center
        avg_lat = sum(p["lat"] for p in intersections) / len(intersections)
        avg_lon = sum(p["lon"] for p in intersections) / len(intersections)
        
        # Calculate TAI radius and quality metrics
        avg_confidence = sum(c.get("confidence", 0.5) for c in active_cones) / len(active_cones)
        avg_cone_width = sum(c.get("cone_width_deg", 45) for c in active_cones) / len(active_cones)
        
        # Estimate radius in meters (simplified - based on cone widths and distances)
        # Average distance between nodes
        import math
        distances = []
        for i in range(len(active_cones)):
            for j in range(i + 1, len(active_cones)):
                lat1 = active_cones[i]["gps"]["lat"]
                lon1 = active_cones[i]["gps"]["lon"]
                lat2 = active_cones[j]["gps"]["lat"]
                lon2 = active_cones[j]["gps"]["lon"]
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance_m = 6371000 * c
                distances.append(distance_m)
        
        avg_distance = sum(distances) / len(distances) if distances else 1000
        
        # Estimate radius based on cone width and confidence
        # Wider cones and lower confidence = larger radius
        base_radius = avg_distance * 0.1  # 10% of average node distance
        width_factor = 1.0 + (avg_cone_width / 90.0)  # 45deg = 1.5x, 90deg = 2x
        confidence_factor = 1.0 + (1.0 - avg_confidence) * 2.0  # Low conf = 3x, high conf = 1x
        radius_m = base_radius * width_factor * confidence_factor
        
        # Clamp radius
        radius_m = max(50, min(5000, radius_m))  # 50m to 5km
        
        # Calculate quality
        quality = avg_confidence * (1.0 - min(avg_cone_width / 90.0, 0.5))  # Simplified quality
        
        # Send to ATAK (throttle to once per 5 seconds)
        if not hasattr(self, "_last_tai_send") or (now - self._last_tai_send) > 5.0:
            self.cot.send_tai(avg_lat, avg_lon, radius_m, avg_confidence, quality)
            self._last_tai_send = now

    def _freq_bin(self, freq_hz: float) -> int:
        b = int(self.auto_policy.freq_bin_hz)
        if b <= 0:
            b = 100_000
        return int(freq_hz) // b

    def can_auto_capture(self, payload: dict) -> tuple[bool, str]:
        """
        Determine if an event is eligible for auto-capture.
        
        Supports both Tripwire v1.1 and v2.0 event formats:
        - v1.1: Uses "stage" field (requires stage="confirmed")
        - v2.0: Uses event types directly (fhss_cluster is actionable)
        - Edge is the authority; Tripwire does not assert intent or threat
        """
        # 1. Hard-block cues - they are advisory only, never actionable
        # Tripwire emits type="rf_cue" for cue-stage events
        event_type = payload.get("type") or payload.get("event_type")  # Support both for transition
        if event_type == "rf_cue":
            return False, "cue_only"
        
        # 2. Never capture heartbeats
        if event_type == "heartbeat":
            return False, "heartbeat"
        
        # 3. Block v2.0 advisory-only event types
        if event_type in ("aoa_cone", "rf_energy_start", "rf_energy_end", "rf_spike"):
            return False, f"advisory_only_{event_type}"
        
        # 4. Block system events
        if event_type.startswith("ibw_calibration_") or event_type in ("df_metrics", "df_bearing"):
            return False, f"system_event_{event_type}"

        # 5. Check for v1.1 stage field OR v2.0 event type eligibility
        # v1.1: requires stage="confirmed"
        # v2.0: fhss_cluster is actionable (equivalent to confirmed)
        stage = payload.get("stage")
        if stage is not None:
            # v1.1 format - require stage="confirmed"
            if stage != "confirmed":
                return False, f"stage_not_confirmed_{stage or 'missing'}"
        else:
            # v2.0 format - only fhss_cluster is actionable (no stage field)
            if event_type != "fhss_cluster":
                # For v2.0, if no stage field and not fhss_cluster, check if it's a legacy confirmed_event
                if event_type != "confirmed_event":
                    return False, f"v2_event_not_actionable_{event_type}"

        # 6. Require confidence (use policy min_confidence)
        # Tripwire confidence = signal confidence; Edge applies its own threshold
        conf = float(payload.get("confidence", 0.0))
        if conf < float(self.auto_policy.min_confidence):
            return False, "low_confidence"

        # 5. Ignore awareness-only scans
        scan_plan = payload.get("scan_plan") or ""
        if scan_plan in ("survey_wide", "wifi_bt_24g"):
            return False, "awareness_only"

        # 6. Cooldown and rate limiting
        now = time.time()

        # Global cooldown
        if (now - float(self._last_auto_ts)) < float(self.auto_policy.global_cooldown_s):
            return False, "global_cooldown"

        # Per-node cooldown
        node_id = str(payload.get("node_id") or "unknown")
        last_node = float(self._last_auto_by_node.get(node_id, 0.0))
        if (now - last_node) < float(self.auto_policy.per_node_cooldown_s):
            return False, "node_cooldown"

        # Per-freqbin cooldown
        f = float(payload.get("freq_hz") or 0.0)
        fb = self._freq_bin(f)
        last_fb = float(self._last_auto_by_freqbin.get(fb, 0.0))
        if (now - last_fb) < float(self.auto_policy.per_freq_cooldown_s):
            return False, "freq_cooldown"

        # Sliding window rate limit (per minute)
        cutoff = now - 60.0
        self._auto_history = [t for t in self._auto_history if t >= cutoff]
        if len(self._auto_history) >= int(self.auto_policy.max_captures_per_min):
            return False, "rate_limited"

        # 7. Rate limit (queue full)
        if self.capture_mgr.queue_full():
            return False, "queue_full"

        # 8. De-duplicate (same node + freq) - placeholder
        # TODO: Implement recent_tripwire_hit if needed
        # node = payload.get("node_id")
        # freq = payload.get("freq_hz")
        # if self.recent_tripwire_hit(node, freq, window_s=10):
        #     return False, "duplicate"

        return True, "ok"

    def mark_auto_capture(self, cue: dict) -> None:
        """Call this immediately when we accept an auto capture."""
        now = time.time()
        node_id = str(cue.get("node_id") or "unknown")
        f = float(cue.get("freq_hz") or 0.0)
        fb = self._freq_bin(f)

        self._last_auto_ts = now
        self._last_auto_by_node[node_id] = now
        self._last_auto_by_freqbin[fb] = now
        self._auto_history.append(now)

    # -------------------------------------------------
    # Status (must never throw)
    # -------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "sdr_open": self._open,
            "scan_running": bool(self._scan and self._scan.is_running()),
            "last_frame_ts": self._last_frame_ts,
            "running_job": self._job,
            "tripwire_cues": list(self.tripwire_cues),
            "aoa_cones": list(self.aoa_cones[-5:]),  # Return last 5 AoA cones
            "queue_depth": None,  # reserved for future job queue
            "task": self.task_info,
            "auto_policy": asdict(self.auto_policy),
            "fft_size": getattr(self, "_current_fft_size", None),  # Current FFT size
            "fps": getattr(self, "_current_fps", None),  # Current FPS
        }
    
    def get_sdr_health(self) -> Dict[str, Any]:
        """
        Get SDR health metrics aggregated from SDR device and RX task.
        Must never throw.
        """
        try:
            # Get health from SDR device
            sdr_health = self.sdr.get_health() if self.sdr else {}
            
            # Get RX task stats if available
            rx_stats = {}
            if self._rx_task:
                rx_stats = {
                    "rx_read_calls": getattr(self._rx_task, "read_calls", 0),
                    "rx_empty_reads": getattr(self._rx_task, "empty_reads", 0),
                    "rx_overflows": getattr(self._rx_task, "overflows", 0),
                }
            
            # Get ring buffer lock contention metrics if available
            ring_metrics = {}
            if self._ring and hasattr(self._ring, "get_lock_metrics"):
                try:
                    ring_metrics = self._ring.get_lock_metrics()
                except Exception:
                    pass  # Don't fail health check if metrics fail
            
            # Merge SDR health with RX task stats and ring buffer metrics
            health = {
                **sdr_health,
                "rx_task": rx_stats,
                "ring_buffer": ring_metrics,
            }
            
            return health
        except Exception as e:
            print(f"[ORCH] Error getting SDR health: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    # -------------------------------------------------
    # Captures Meta Data
    # -------------------------------------------------
    def list_captures(self, limit: int = 50):
        return self.capture_log[-limit:]

    def _log_capture_result(self, res) -> None:
        """
        Record capture metadata for UI / operator visibility.
        Must never throw.
        """
        try:
            entry = {
                "ts": res.ts,
                "freq_hz": res.freq_hz,
                "duration_s": res.duration_s,
                "source_node": getattr(res, "source_node", None),
                "scan_plan": getattr(res, "scan_plan", None),
                "reason": getattr(res, "reason", "unknown"),
            }

            self.capture_log.append(entry)

            if len(self.capture_log) > 500:
                self.capture_log = self.capture_log[-250:]

        except Exception:
            pass

    def _on_capture_result(self, res):
        """
        Capture completed → classify → ATAK alert
        
        Per Tripwire v1.1 alignment:
        - Only forward confirmed events (stage="confirmed") to ATAK
        - Cues are advisory only and never forwarded
        - Edge is the authority; Tripwire does not assert intent or threat
        """
        try:
            # ATAK forwarding eligibility: only confirmed events
            # Per Tripwire v1.1: cues are advisory only, not actionable
            stage = getattr(res, "stage", None)
            if stage != "confirmed":
                # Not eligible for ATAK forwarding - log but don't forward
                print(f"[ATAK] Skipping ATAK forward - stage={stage} (only confirmed events forwarded)")
                return
            
            capture = {
                "iq_path": res.iq_path,
                "meta": {
                    "freq_hz": res.freq_hz,
                    "sample_rate_sps": res.sample_rate_sps,
                    "duration_s": res.duration_s,
                }
            }

            result = self.classifier.classify_capture(capture)

            label = result.get("primary_label", "unknown")
            confidence = float(result.get("confidence", 0.0))
            freq_mhz = res.freq_hz / 1e6

            # Human-readable label only (no raw RF metrics per alignment doc)
            msg = (
                f"{label.upper()} detected @ "
                f"{freq_mhz:.3f} MHz "
                f"(confidence {confidence:.2f})"
            )

            self.cot.send_chat(msg)
            xml = self.cot.build_detection_marker(msg)
            self.cot.send_event(xml)

            self.bus.publish_nowait(
                "classification_result",
                {
                    "label": label,
                    "confidence": confidence,
                    "freq_hz": res.freq_hz,
                    "source_node": res.source_node,
                    "scan_plan": res.scan_plan,
                    "ts": res.ts,
                }
            )

        except Exception as e:
            print("[SPEAR-EDGE][CLASSIFY ERROR]", e)

    # -------------------------------------------------
    # ATAK Status Messages
    # -------------------------------------------------

    def _count_connected_tripwires(self) -> int:
        """Count currently connected tripwire nodes"""
        nodes = self.tripwires.snapshot()
        now = time.time()
        connected = 0
        for node in nodes:
            last_seen = node.get("last_seen", 0)
            if (now - last_seen) < CONNECTED_SECS:
                connected += 1
        return connected

    def _send_atak_status(self, online: bool, tripwire_count: int = 0):
        """Send Edge online/offline status to ATAK via CoT chat"""
        if online and self.mode == "armed":
            msg = f"SPEAR-Edge Online connect to {tripwire_count} tripwires ARMED and awaiting cues"
            self._last_atak_tripwire_count = tripwire_count
        else:
            msg = "SPEAR Edge Offline"
            self._last_atak_tripwire_count = -1
        self.cot.send_chat(msg)
        print(f"[ATAK] Status: {msg}")

    def _on_tripwire_nodes_updated(self, payload: dict):
        """Called when tripwire_nodes event is published - update ATAK status if count changed"""
        if self.mode != "armed":
            return
        
        count = self._count_connected_tripwires()
        if count != self._last_atak_tripwire_count:
            self._send_atak_status(online=True, tripwire_count=count)

    def _on_capture_start(self, payload: dict):
        """Called when capture_start event is published - send ATAK message"""
        freq_hz = payload.get("freq_hz", 0)
        reason = payload.get("reason", "unknown")
        source_node = payload.get("source_node")
        duration_s = payload.get("duration_s", 0)
        
        freq_mhz = freq_hz / 1e6
        
        # Build message based on capture reason
        if reason == "tripwire_armed":
            if source_node:
                msg = f"SPEAR-Edge capturing {freq_mhz:.3f} MHz from {source_node} (Tripwire cue)"
            else:
                msg = f"SPEAR-Edge capturing {freq_mhz:.3f} MHz (Tripwire cue)"
        elif reason == "manual":
            msg = f"SPEAR-Edge capturing {freq_mhz:.3f} MHz (Manual capture)"
        elif reason == "tasked":
            msg = f"SPEAR-Edge capturing {freq_mhz:.3f} MHz (Tasked)"
        else:
            msg = f"SPEAR-Edge capturing {freq_mhz:.3f} MHz ({reason})"
        
        if duration_s > 0:
            msg += f" for {duration_s:.1f}s"
        
        self.cot.send_chat(msg)
        print(f"[ATAK] Capture start: {msg}")

    def _setup_tripwire_status_updates(self):
        """Subscribe to tripwire_nodes and capture_start events for ATAK status updates"""
        async def _watch_tripwire_nodes():
            try:
                queue = await self.bus.subscribe("tripwire_nodes", maxsize=50)
                while True:
                    try:
                        payload = await queue.get()
                        self._on_tripwire_nodes_updated(payload)
                    except Exception as e:
                        print(f"[ATAK] Error processing tripwire_nodes event: {e}")
            except Exception as e:
                print(f"[ATAK] Error subscribing to tripwire_nodes: {e}")
        
        async def _watch_capture_start():
            try:
                queue = await self.bus.subscribe("capture_start", maxsize=50)
                while True:
                    try:
                        payload = await queue.get()
                        self._on_capture_start(payload)
                    except Exception as e:
                        print(f"[ATAK] Error processing capture_start event: {e}")
            except Exception as e:
                print(f"[ATAK] Error subscribing to capture_start: {e}")
        
        # Start background tasks to watch for events
        asyncio.create_task(_watch_tripwire_nodes())
        asyncio.create_task(_watch_capture_start())