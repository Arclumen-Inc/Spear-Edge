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
        # Hook up log callback (classification/ATAK handled via event bus subscription)
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
            # Reduce ring buffer size for high rates to save memory
            # 0.3 seconds for rates > 20 MS/s, 0.5 seconds otherwise
            ring_duration = 0.3 if sample_rate_sps > 20_000_000 else 0.5
            ring_size = int(sample_rate_sps * ring_duration)
            self._ring = IQRingBuffer(ring_size)

            # Aim for ~8–12 ms of samples per read (conservative to match USB buffer sizes)
            # Read sizes must be SMALLER than USB buffer size to prevent timeouts
            # New conservative buffer sizes:
            #   >40 MS/s: 256K samples buffer → use 128K-192K read size
            #   20-40 MS/s: 128K samples buffer → use 64K-96K read size
            #   10-20 MS/s: 64K samples buffer → use 32K-48K read size
            #   <10 MS/s: 32K samples buffer → use 16K-24K read size
            if sample_rate_sps > 40_000_000:
                chunk_duration_ms = 0.008  # 8ms for very high rates (60 MS/s = 480k samples, clamp to 192K)
            elif sample_rate_sps > 20_000_000:
                chunk_duration_ms = 0.010  # 10ms for high rates (30 MS/s = 300k samples, clamp to 96K)
            elif sample_rate_sps > 10_000_000:
                chunk_duration_ms = 0.012  # 12ms for medium rates (15 MS/s = 180k samples, clamp to 48K)
            else:
                chunk_duration_ms = 0.015  # 15ms for normal rates (2.4 MS/s = 36k samples, clamp to 24K)
            
            chunk = int(sample_rate_sps * chunk_duration_ms)

            # Adaptive clamping: ensure read size is smaller than buffer size
            chunk = max(chunk, 16384)  # Minimum 16K samples
            # Clamp to be safely smaller than buffer sizes (use ~75% of buffer size)
            if sample_rate_sps > 40_000_000:
                max_chunk = 196608  # 192K (75% of 256K buffer)
            elif sample_rate_sps > 20_000_000:
                max_chunk = 98304  # 96K (75% of 128K buffer)
            elif sample_rate_sps > 10_000_000:
                max_chunk = 49152  # 48K (75% of 64K buffer)
            else:
                max_chunk = 24576  # 24K (75% of 32K buffer)
            chunk = min(chunk, max_chunk)

            self._rx_task = RxTask(
                sdr=self.sdr,
                ring=self._ring,
                chunk_size=chunk,
            )
            await self._rx_task.start()

            # FFT task (FPS limited)
            # Use calibration offset from settings (defaults to 0.0 for raw bladeRF values)
            from spear_edge.settings import settings
            self._scan = ScanTask(
                ring=self._ring,
                center_freq_hz=center_freq_hz,
                sample_rate_sps=sample_rate_sps,
                fft_size=fft_size,
                fps=fps,
                calibration_offset_db=settings.CALIBRATION_OFFSET_DB,
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

    async def set_fft_smoothing(self, alpha: float) -> None:
        """
        Set FFT smoothing alpha value (0.0 to 1.0).
        Lower values = more smoothing, higher values = less smoothing.
        Default: 0.1 (good for wideband signals like VTX).
        """
        async with self._lock:
            if self._scan and self._scan.is_running():
                await self._scan.set_smoothing(alpha)
                print(f"[ORCH] FFT smoothing set to {alpha:.3f}")

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

        # Notify connected tripwires (WS) with v2.0 format
        links = getattr(self, "tripwire_links", None)
        if isinstance(links, dict):
            # Build active_nodes list
            active_nodes = []
            nodes = self.tripwires.snapshot()
            now = time.time()
            for node in nodes:
                last_seen = node.get("last_seen", 0)
                if (now - last_seen) < CONNECTED_SECS:
                    active_nodes.append(node.get("node_id", "unknown"))
            
            # Count TAIs
            tai_count = len(getattr(self, "aoa_cones", []))
            
            msg = {
                "type": "edge_state",
                "mode": self.mode,
                "active_nodes": active_nodes,
                "tai_count": tai_count,
                "timestamp": now,
            }
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
        Calculate TAI from active AoA cones using proper geometric triangulation.
        Sends TAI to ATAK and publishes to event bus if we have 2+ cones with GPS.
        """
        from spear_edge.core.integrate.aoa_fusion import (
            cones_from_dicts, fuse_bearing_cones, tai_to_dict
        )
        
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
            
            # Must have bearing
            if cone.get("bearing_deg") is None:
                continue
            
            # Get node GPS from registry (or use GPS in cone if present)
            gps = cone.get("gps")
            if not gps or not gps.get("lat"):
                nodes = self.tripwires.snapshot()
                node = next((n for n in nodes if n["node_id"] == node_id), None)
                if not node or not node.get("gps") or not node["gps"].get("lat"):
                    continue
                gps = node["gps"]
            
            # Build cone dict with GPS
            cone_with_gps = dict(cone)
            cone_with_gps["gps"] = gps
            cone_with_gps["node_id"] = node_id
            
            node_ids_seen.add(node_id)
            active_cones.append(cone_with_gps)
            
            # Limit to 4 cones max for performance
            if len(active_cones) >= 4:
                break
        
        # Need at least 2 cones for triangulation
        if len(active_cones) < 2:
            return
        
        # Convert to BearingCone objects and run fusion
        try:
            cones = cones_from_dicts(active_cones)
            if len(cones) < 2:
                print("[TAI] Not enough valid cones for fusion")
                return
            
            tai = fuse_bearing_cones(cones)
            
            if not tai.valid:
                print(f"[TAI] Fusion failed: {tai.error_message}")
                return
            
            # Store last TAI result for API access
            self._last_tai = tai
            self._last_tai_dict = tai_to_dict(tai)
            self._last_tai_ts = now
            
            # Publish TAI to event bus
            self.bus.publish_nowait("tai_update", {
                "tai": self._last_tai_dict,
                "active_cones": len(cones),
                "ts": now,
            })
            
            # Calculate quality for ATAK (based on GDOP and confidence)
            # Lower GDOP = better, so invert it for quality score
            quality = tai.confidence * (1.0 / max(tai.gdop, 1.0))
            quality = min(1.0, max(0.0, quality))
            
            # Send to ATAK (throttle to once per 5 seconds)
            if not hasattr(self, "_last_tai_send") or (now - self._last_tai_send) > 5.0:
                # Build polygon vertices for visualization
                polygon_vertices = None
                if tai.polygon and len(tai.polygon) >= 3:
                    polygon_vertices = [(p.lat, p.lon) for p in tai.polygon]
                
                # Send TAI with circle and polygon
                self.cot.send_tai(
                    tai.centroid.lat,
                    tai.centroid.lon,
                    tai.radius_m,
                    tai.confidence,
                    quality,
                    polygon_vertices=polygon_vertices,
                )
                
                # Send bearing lines from each contributing sensor
                bearing_data = []
                for cone_data in active_cones:
                    gps = cone_data.get("gps", {})
                    if gps.get("lat") and gps.get("lon") and cone_data.get("bearing_deg") is not None:
                        bearing_data.append({
                            "node_id": cone_data.get("node_id", "unknown"),
                            "callsign": cone_data.get("callsign"),
                            "lat": gps["lat"],
                            "lon": gps["lon"],
                            "bearing_deg": cone_data["bearing_deg"],
                            "confidence": cone_data.get("confidence", 0.5),
                        })
                
                if bearing_data:
                    # Calculate range for bearing lines (distance to TAI center)
                    from spear_edge.core.integrate.aoa_fusion import distance_m, GeoPoint
                    max_range = 5000  # Default 5km
                    for b in bearing_data:
                        try:
                            dist = distance_m(
                                GeoPoint(lat=b["lat"], lon=b["lon"]),
                                GeoPoint(lat=tai.centroid.lat, lon=tai.centroid.lon)
                            )
                            max_range = max(max_range, dist * 1.5)  # Extend past TAI
                        except Exception:
                            pass
                    
                    self.cot.send_bearing_lines(bearing_data, range_m=max_range)
                
                self._last_tai_send = now
                
                print(f"[TAI] Triangulation: center=({tai.centroid.lat:.6f}, {tai.centroid.lon:.6f}), "
                      f"radius={tai.radius_m:.0f}m, area={tai.area_m2:.0f}m², "
                      f"GDOP={tai.gdop:.2f}, confidence={tai.confidence:.2f}, "
                      f"polygon_vertices={len(tai.polygon)}, bearings={len(bearing_data)}")
        
        except Exception as e:
            print(f"[TAI] Error during fusion: {e}")
            import traceback
            traceback.print_exc()

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

        # 8. De-duplicate (same node + freq)
        # Note: Per-freqbin cooldown (above) already provides deduplication
        # Additional per-node+freq deduplication can be added here if needed

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
        """Return captures with classification and directory info."""
        from pathlib import Path
        import json
        from datetime import datetime
        
        results = []
        for entry in self.capture_log[-limit:]:
            # Try to find capture directory and classification
            capture_dir = None
            classification = None
            
            # Derive directory name from timestamp and frequency
            ts = entry.get("ts", 0)
            freq_hz = entry.get("freq_hz", 0)
            if ts and freq_hz:
                # Format: YYYYMMDD_HHMMSS_{freq}Hz_{sample_rate}sps_{reason}
                try:
                    dt = datetime.fromtimestamp(ts)
                    dir_pattern = f"{dt.strftime('%Y%m%d_%H%M%S')}_{int(freq_hz)}Hz"
                    
                    # Search for matching directory
                    captures_dir = Path("data/artifacts/captures")
                    if captures_dir.exists():
                        for cap_dir in captures_dir.iterdir():
                            if cap_dir.is_dir() and dir_pattern in cap_dir.name:
                                capture_dir = cap_dir.name
                                # Try to read classification
                                json_path = cap_dir / "capture.json"
                                if json_path.exists():
                                    try:
                                        with open(json_path, 'r') as f:
                                            data = json.load(f)
                                        classification = data.get("classification", {})
                                    except:
                                        pass
                                break
                except Exception:
                    pass
            
            result = entry.copy()
            result["capture_dir"] = capture_dir
            result["classification"] = classification
            results.append(result)
        
        return results

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
        Capture completed → read classification → ATAK alert
        
        Per Tripwire v1.1 and v2.0 alignment:
        - v1.1: Only forward confirmed events (stage="confirmed") to ATAK
        - v2.0: fhss_cluster events are actionable (no stage field)
        - Cues are advisory only and never forwarded
        - Edge is the authority; Tripwire does not assert intent or threat
        - Uses classification from capture_manager (already done, stored in capture.json)
        """
        try:
            # ATAK forwarding eligibility check
            # Read event_type from capture.json to determine v2.0 actionable events
            import json
            from pathlib import Path
            
            meta_path = Path(res.meta_path)
            if not meta_path.exists():
                print(f"[ATAK] Warning: capture.json not found at {meta_path}, skipping ATAK message")
                return
            
            with open(meta_path, 'r') as f:
                capture_data = json.load(f)
            
            # Check ATAK forwarding eligibility
            # v1.1: requires stage="confirmed"
            # v2.0: fhss_cluster and confirmed_event are actionable even without stage
            stage = getattr(res, "stage", None)
            event_type = None
            if capture_data.get("request"):
                meta = capture_data["request"].get("meta", {})
                if meta:
                    event_type = meta.get("type") or meta.get("event_type")
            
            # v2.0 actionable event types (no stage field needed)
            v2_actionable_types = {"fhss_cluster", "confirmed_event"}
            
            is_v1_confirmed = (stage == "confirmed")
            is_v2_actionable = (event_type in v2_actionable_types)
            
            if not is_v1_confirmed and not is_v2_actionable:
                # Not eligible for ATAK forwarding - log but don't forward
                print(f"[ATAK] Skipping ATAK forward - stage={stage}, event_type={event_type} (not actionable)")
                return
            
            # Get classification from capture.json (already loaded above)
            classification = capture_data.get("classification")
            if not classification:
                print(f"[ATAK] No classification in capture.json, skipping ATAK message")
                return
            
            label = classification.get("label", "unknown")
            confidence = float(classification.get("confidence", 0.0))
            freq_mhz = res.freq_hz / 1e6

            # Get human-readable name if available
            device_name = classification.get("device_name", label.upper())
            
            # Human-readable label only (no raw RF metrics per alignment doc)
            msg = (
                f"{device_name} detected @ "
                f"{freq_mhz:.3f} MHz "
                f"(confidence {confidence:.2f})"
            )

            self.cot.send_chat(msg)
            xml = self.cot.build_detection_marker(msg)
            self.cot.send_event(xml)
            
            print(f"[ATAK] Classification sent to ATAK: {msg}")

        except Exception as e:
            print(f"[ATAK] Error sending classification to ATAK: {e}")
            import traceback
            traceback.print_exc()

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
        """Subscribe to tripwire_nodes, capture_start, and capture_result events for ATAK updates"""
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
        
        async def _watch_capture_result():
            """Subscribe to capture_result events to send classifications to ATAK"""
            try:
                queue = await self.bus.subscribe("capture_result", maxsize=50)
                while True:
                    try:
                        res = await queue.get()
                        # res is a CaptureResult dataclass
                        self._on_capture_result(res)
                    except Exception as e:
                        print(f"[ATAK] Error processing capture_result event: {e}")
            except Exception as e:
                print(f"[ATAK] Error subscribing to capture_result: {e}")
        
        # Start background tasks to watch for events
        asyncio.create_task(_watch_tripwire_nodes())
        asyncio.create_task(_watch_capture_start())
        asyncio.create_task(_watch_capture_result())