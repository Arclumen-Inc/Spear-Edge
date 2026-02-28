# SPEAR-Edge Detailed Assessment

**Date:** 2025-01-XX  
**Assessor:** AI Code Review  
**Scope:** RF Engineering & Software Engineering Perspectives  
**Objective:** Evaluate if SPEAR-Edge meets its stated objectives as a standalone SDR software with Tripwire integration and ML classification

---

## Executive Summary

SPEAR-Edge is a **well-architected** SDR monitoring and capture system that **largely meets** its core objectives, with some **critical gaps** in ML classification and **notable strengths** in RF signal processing and Tripwire integration.

### Overall Assessment: **7.5/10**

**Strengths:**
- ✅ Robust SDR hardware abstraction and bladeRF lifecycle management
- ✅ Excellent Tripwire integration with proper policy enforcement
- ✅ High-quality signal processing pipeline (FFT, spectrograms, quality metrics)
- ✅ Memory-efficient capture system with streaming to disk
- ✅ Well-structured async architecture

**Gaps:**
- ⚠️ ML classification is **stub-only** (no trained model, dummy inference)
- ⚠️ Limited error recovery for SDR stream failures
- ⚠️ No automated testing infrastructure
- ⚠️ GPU acceleration not utilized (CPU-only ONNX)

---

## 1. RF Engineering Assessment

### 1.1 SDR Hardware Control ⭐⭐⭐⭐⭐ (5/5)

**Strengths:**
- **Proper bladeRF lifecycle**: Correctly implements the critical order (sample rate → bandwidth → frequency → enable RX → stream setup)
- **Power-of-two read sizes**: Enforces 8192/16384/262144 sample reads (bladeRF requirement)
- **CS16 format optimization**: Uses CS16 (4 bytes/sample) instead of CF32 (8 bytes/sample) for 50% bandwidth reduction at high sample rates
- **Stream buffer management**: Sets 64 buffers with proper timeout handling
- **Health monitoring**: Tracks read success rates, overflows, throughput metrics

**Code Evidence:**
```200:223:spear_edge/core/sdr/soapy.py
    def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None):
        if not self.dev:
            return
    
        ch = int(self.rx_channel)
    
        self.center_freq_hz = int(center_freq_hz)
        self.sample_rate_sps = int(sample_rate_sps)
        self.bandwidth_hz = int(bandwidth_hz) if bandwidth_hz else int(sample_rate_sps)
    
        # bladeRF REQUIRED ORDER
        self.dev.setSampleRate(SOAPY_SDR_RX, ch, float(self.sample_rate_sps))
        self.dev.setBandwidth(SOAPY_SDR_RX, ch, float(self.bandwidth_hz))
        self.dev.setFrequency(SOAPY_SDR_RX, ch, float(self.center_freq_hz))
    
        # bladeRF RX enable kick
        try:
            self.dev.writeSetting("ENABLE_CHANNEL", "RX", "true")
        except Exception:
            pass

        # ✅ STREAM SETUP MUST HAPPEN LAST (after all RF settings)
        # bladeRF requires stream to be created/activated AFTER sample rate/frequency are set
        self._setup_stream()
```

**Assessment:** Excellent. The code demonstrates deep understanding of bladeRF hardware constraints and implements them correctly.

### 1.2 Signal Processing Quality ⭐⭐⭐⭐ (4/5)

**Strengths:**
- **Proper windowing**: Uses Hanning window with energy normalization for stable scaling
- **FFT normalization**: Normalizes by window energy (`win_energy`) for ML-friendly, consistent power levels
- **Noise floor estimation**: Uses 10th percentile for robust noise floor calculation
- **Spectrogram generation**: Chunked processing for memory efficiency (5M sample chunks)
- **Quality metrics**: Computes SNR, occupied bandwidth, duty cycle, burstiness, clipping detection

**Code Evidence:**
```42:56:spear_edge/core/scan/scan_task.py
        # Window & frequency axis are fixed for a given configuration.
        self._win = np.hanning(self.fft_size).astype(np.float32)
        self._freqs = (
            np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / self.sample_rate_sps))
            + self.center_freq_hz
        ).astype(np.float64)
        # Cache the list to avoid rebuilding every frame
        self._freqs_list = self._freqs.tolist()

        # Normalization constants for stable scaling
        # We normalize power by window energy so the scale doesn't drift with fft_size/window.
        self._win_energy = float(np.sum(self._win * self._win))  # sum(win^2)
        self._eps = 1e-12
```

**Gaps:**
- **No DC offset correction**: DC offsets are detected but not corrected in real-time
- **No IQ imbalance correction**: No calibration for I/Q imbalance
- **Fixed display range**: Waterfall uses fixed -90 to -20 dBFS (no auto-leveling)

**Assessment:** Very good signal processing fundamentals. Missing some advanced calibration features but adequate for most use cases.

### 1.3 Capture Quality ⭐⭐⭐⭐ (4/5)

**Strengths:**
- **Memory-efficient streaming**: Writes IQ directly to disk (no full-file buffering)
- **Rate-adaptive chunk sizes**: Adjusts read chunk size based on sample rate (16384 → 262144)
- **Stream priming**: Performs dummy reads to warm up bladeRF stream before capture
- **Quality triage**: Detects signal presence, noise, burstiness, bandwidth occupancy
- **Comprehensive metadata**: SigMF format, capture.json with full provenance

**Code Evidence:**
```258:293:spear_edge/core/capture/capture_manager.py
                # CRITICAL: Flush/prime the stream with dummy reads
                # bladeRF stream needs to be "warmed up" after activation
                # Also need to drain any stale buffers
                print("[CAPTURE] Priming stream with dummy reads...")
                total_primed = 0
                for flush_attempt in range(10):  # More attempts
                    dummy = await asyncio.to_thread(self.orch.sdr.read_samples, 8192)
                    if dummy is not None and dummy.size > 0:
                        total_primed += dummy.size
                        print(f"[CAPTURE] Stream primed: got {dummy.size} samples (total {total_primed}) on attempt {flush_attempt+1}")
                        # Continue priming to drain buffers
                        if flush_attempt >= 3 and dummy.size == 8192:
                            # Got full read, stream is ready
                            break
                    else:
                        # Got 0 samples - wait a bit and try again
                        await asyncio.sleep(0.05)
                
                # Final verification - do a few continuous reads to ensure stream is stable
                print("[CAPTURE] Performing continuous verification reads...")
                verify_count = 0
                for verify_attempt in range(3):
                    test_samples = await asyncio.to_thread(self.orch.sdr.read_samples, 8192)
                    if test_samples is not None and test_samples.size > 0:
                        verify_count += 1
                        print(f"[CAPTURE] Verification read {verify_attempt+1}: got {test_samples.size} samples")
                    else:
                        print(f"[CAPTURE] Verification read {verify_attempt+1}: got 0 samples")
                    # No delay - keep reads continuous to maintain stream state
                
                if verify_count == 0:
                    print("[CAPTURE] WARNING: All verification reads returned 0 samples")
                    print("[CAPTURE] Stream may not be ready, but continuing capture attempt...")
                else:
                    print(f"[CAPTURE] Stream verified: {verify_count}/3 reads successful")
```

**Gaps:**
- **No GPS timestamping**: Capture timestamps are system time only (no GPS sync)
- **No sample clock calibration**: No correction for SDR clock drift
- **Fixed capture duration**: 5 seconds default (not adaptive based on signal characteristics)

**Assessment:** Excellent capture pipeline with proper stream handling. Missing some advanced features but production-ready.

### 1.4 RF Configuration Management ⭐⭐⭐⭐ (4/5)

**Strengths:**
- **Gain control**: Supports manual gain and AGC mode
- **Bandwidth control**: Properly sets bandwidth (defaults to sample rate if not specified)
- **Frequency tuning**: Atomic tuning with proper stream lifecycle
- **Channel selection**: Supports multi-channel SDRs (though only single channel used)

**Gaps:**
- **No gain optimization**: No automatic gain selection based on signal level
- **No bandwidth optimization**: Doesn't automatically adjust bandwidth for signal characteristics
- **No frequency calibration**: No correction for frequency offset/LO drift

**Assessment:** Solid RF configuration management. Functional but could benefit from more automation.

---

## 2. Software Engineering Assessment

### 2.1 Architecture & Design ⭐⭐⭐⭐⭐ (5/5)

**Strengths:**
- **Clear separation of concerns**: Orchestrator (SDR lifecycle), CaptureManager (captures), ScanTask (FFT), RxTask (sampling)
- **Event-driven design**: EventBus for pub/sub (live_spectrum, capture_result, tripwire_cue)
- **Async/await throughout**: Proper use of asyncio for non-blocking operations
- **Thread-safe ring buffer**: Dedicated thread for SDR reads, async for FFT processing
- **Dependency injection**: SDR passed to Orchestrator, CaptureManager takes Orchestrator

**Code Evidence:**
```48:136:spear_edge/core/orchestrator/orchestrator.py
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
```

**Assessment:** Excellent architecture. Clean separation, proper async patterns, event-driven design.

### 2.2 Tripwire Integration ⭐⭐⭐⭐⭐ (5/5)

**Strengths:**
- **Proper policy enforcement**: Only `stage="confirmed"` events trigger auto-capture (not cues)
- **Rate limiting**: Global, per-node, and per-frequency cooldowns prevent spam
- **WebSocket + HTTP**: WebSocket for node registration/heartbeat, HTTP for events
- **Metadata preservation**: All Tripwire fields (confidence, classification, scan_plan) preserved in capture metadata
- **Advisory cues**: Cues are displayed in UI but never trigger captures (per v1.1 alignment)

**Code Evidence:**
```338:411:spear_edge/core/orchestrator/orchestrator.py
    def can_auto_capture(self, payload: dict) -> tuple[bool, str]:
        """
        Determine if an event is eligible for auto-capture.
        
        Per Tripwire v1.1 alignment:
        - Cues (type="rf_cue") are advisory only and never trigger auto-capture
        - Only confirmed events (stage="confirmed") with sufficient confidence are eligible
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

        # 3. Require stage == "confirmed" for action eligibility
        # Per Tripwire v1.1: only confirmed events are eligible for auto-capture
        stage = payload.get("stage")
        if stage != "confirmed":
            return False, f"stage_not_confirmed_{stage or 'missing'}"

        # 4. Require confidence (use policy min_confidence)
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
```

**Assessment:** Excellent Tripwire integration. Proper policy enforcement, rate limiting, and metadata handling.

### 2.3 ML Classification ⭐⭐ (2/5) **CRITICAL GAP**

**Current State:**
- **ONNX Runtime infrastructure**: ✅ Installed and working
- **Classification pipeline**: ✅ Integrated into capture flow
- **Model loading**: ✅ Attempts to load `spear_dummy.onnx`
- **Trained model**: ❌ **NO TRAINED MODEL** (using dummy/stub)
- **GPU acceleration**: ❌ CPU-only (no TensorRT/CUDA providers)

**Code Evidence:**
```59:86:spear_edge/core/capture/capture_manager.py
        # Classifier for ML-based classification
        # Try ONNX first, then orchestrator's classifier, then stub
        self.classifier = None
        
        # Try ONNX classifier first
        try:
            from spear_edge.ml.infer_onnx import ONNXRfClassifier
            model_path = Path("spear_edge/ml/models/spear_dummy.onnx")
            if model_path.exists():
                self.classifier = ONNXRfClassifier(str(model_path))
                print("[CaptureManager] Loaded ONNX classifier")
            else:
                print(f"[CaptureManager] ONNX model not found: {model_path}")
        except ImportError as e:
            print(f"[CaptureManager] ONNX Runtime not available: {e}")
            print("[CaptureManager] Install with: pip3 install onnxruntime")
        except Exception as e:
            print(f"[CaptureManager] Failed to load ONNX classifier: {e}")
        
        # Fallback to orchestrator's classifier if available
        if self.classifier is None:
            if hasattr(orchestrator, "classifier") and hasattr(orchestrator.classifier, "classify"):
                self.classifier = orchestrator.classifier
                print("[CaptureManager] Using orchestrator-provided classifier")
        
        # Final fallback to stub
        if self.classifier is None:
            from spear_edge.ml.infer_stub import StubRFClassifier
            self.classifier = StubRFClassifier()
            print("[CaptureManager] Using stub classifier (no ML model available)")
```

**Impact:**
- **Objective NOT met**: "ML inference for classification" is **not functional** - only stub/dummy classification
- **Data collection ready**: Spectrograms are generated correctly (512x512 float32, noise-floor normalized)
- **Infrastructure ready**: ONNX pipeline exists, just needs trained model

**Assessment:** **CRITICAL GAP**. Infrastructure is excellent, but no trained model means classification is non-functional. This is the **primary blocker** for meeting the stated objective.

### 2.4 Error Handling & Resilience ⭐⭐⭐ (3/5)

**Strengths:**
- **SDR read never throws**: `read_samples()` returns empty array on error (never crashes)
- **Stream recovery**: Attempts to force stream setup if None detected
- **Capture queue protection**: Queue full detection prevents overload
- **Event bus isolation**: Subscriber exceptions don't crash runtime

**Code Evidence:**
```249:310:spear_edge/core/sdr/soapy.py
    def read_samples(self, num_samples: int) -> np.ndarray:
        if self.dev is None or self.rx_stream is None:
            return np.empty(0, dtype=np.complex64)

        n = int(num_samples)
        if n <= 0:
            return np.empty(0, dtype=np.complex64)

        # CS16 format: interleaved int16 (I, Q, I, Q...)
        # Need to read as int16 buffer, then convert to complex64
        cs16_buff = np.empty(n * 2, dtype=np.int16)  # 2 int16 per complex sample

        # Track read time for health metrics
        read_start_ns = time.perf_counter_ns()
        self._health_stats["total_reads"] += 1

        try:
            # Use 100ms timeout for live mode (fast retries prevent overflow)
            # For captures, the capture manager can handle longer waits if needed
            sr = self.dev.readStream(self.rx_stream, [cs16_buff], n, timeoutUs=100_000)
            read_time_ns = time.perf_counter_ns() - read_start_ns
            self._health_stats["total_read_time_ns"] += read_time_ns
            
            # sr.ret: >0 samples, 0 timeout, <0 error
            if sr.ret > 0:
                # Successful read
                self._health_stats["successful_reads"] += 1
                self._health_stats["total_samples"] += sr.ret
                
                # Convert CS16 (interleaved int16) to complex64
                # Scale from int16 range [-32768, 32767] to [-1.0, 1.0]
                iq_int16 = cs16_buff[: sr.ret * 2].reshape(sr.ret, 2)
                buff = np.empty(sr.ret, dtype=np.complex64)
                buff.real = iq_int16[:, 0].astype(np.float32) / 32768.0
                buff.imag = iq_int16[:, 1].astype(np.float32) / 32768.0
                return buff
            elif sr.ret == 0:
                # Timeout
                self._health_stats["timeout_reads"] += 1
                # Timeout - log for debugging but return empty
                # This is normal if stream is not producing data fast enough
                return np.empty(0, dtype=np.complex64)
            else:
                # Error (negative return code)
                self._health_stats["error_reads"] += 1
                # -1 = SOAPY_SDR_TIMEOUT
                # -4 = SOAPY_SDR_OVERFLOW (buffer overflow - reading too slow)
                # -7 = SOAPY_SDR_UNDERFLOW (buffer underflow - reading too fast)
                if sr.ret == -4:
                    self._overflow_count += 1
                    self._health_stats["overflow_errors"] += 1
                    if self._overflow_count % 200 == 0:
                        print(f"[SDR] OVERFLOW x{self._overflow_count} (reading too slowly)")
                elif sr.ret == -7:
                    print(f"[SDR] readStream UNDERFLOW (ret={sr.ret}): Reading too fast, buffer empty")
                else:
                    print(f"[SDR] readStream error: ret={sr.ret}")
                return np.empty(0, dtype=np.complex64)
        except Exception as e:
            self._health_stats["error_reads"] += 1
            print(f"[SDR] readStream exception: {e}")
            return np.empty(0, dtype=np.complex64)
```

**Gaps:**
- **No automatic stream recovery**: If stream fails, requires manual restart
- **No device reconnection**: If SDR disconnects, no automatic reconnection
- **Limited retry logic**: Capture failures don't retry automatically
- **No health-based degradation**: Doesn't reduce sample rate if overflows persist

**Assessment:** Good defensive programming, but lacks automatic recovery mechanisms.

### 2.5 Code Quality ⭐⭐⭐⭐ (4/5)

**Strengths:**
- **Type hints**: Extensive use of type hints (`Optional[Dict[str, Any]]`, etc.)
- **Dataclasses**: Uses `@dataclass` for data models
- **Docstrings**: Good documentation of critical functions
- **Consistent naming**: Clear, descriptive variable names
- **Modular design**: Well-organized into logical modules

**Gaps:**
- **No unit tests**: No test files found in codebase
- **No integration tests**: No automated testing of capture pipeline
- **Limited logging**: Uses `print()` instead of proper logging framework
- **No type checking**: No mypy/pyright configuration

**Assessment:** Good code quality, but lacks testing infrastructure.

### 2.6 Performance Optimization ⭐⭐⭐⭐ (4/5)

**Strengths:**
- **Memory-efficient captures**: Streams IQ to disk (no full-file buffering)
- **Chunked spectrogram processing**: 5M sample chunks prevent memory exhaustion
- **CS16 format**: 50% bandwidth reduction vs CF32
- **Ring buffer**: Thread-safe, lock-free(ish) for RX task
- **Async I/O**: Non-blocking operations throughout

**Code Evidence:**
```550:637:spear_edge/core/capture/capture_manager.py
    async def _capture_iq_to_disk(
        self,
        iq_path: Path,
        sample_rate: int,
        duration_s: float,
    ) -> int:
        """
        Capture IQ samples directly to disk (memory-efficient).
        Returns number of samples actually captured.
        """
        n = int(sample_rate * duration_s)
        if n <= 0:
            raise ValueError("Invalid capture length")

        # Aggressive chunk sizing based on sample rate (reduces syscall/GIL pressure)
        # At high rates, larger chunks are critical for reliable capture
        if sample_rate >= 20_000_000:
            # 30 MS/s: use 262144 or larger
            chunk = 262144
        elif sample_rate >= 10_000_000:
            # 10 MS/s: use 65536 or 131072
            chunk = 131072
        elif sample_rate >= 5_000_000:
            # 5 MS/s: use 32768
            chunk = 32768
        else:
            # Lower rates: use 16384
            chunk = 16384
        
        # Ensure power-of-two (bladeRF requirement)
        chunk = 1 << (chunk.bit_length() - 1) if chunk > 0 else 16384
        
        # Timeout budget to prevent infinite loops
        t0 = time.time()
        max_wait_s = max(2.0, duration_s * 2.0)

        print(f"[CAPTURE] Starting IQ capture to disk: {n} samples ({duration_s}s @ {sample_rate/1e6:.3f} MS/s)")
        print(f"[CAPTURE] Using chunk size: {chunk} samples (rate-adaptive)")

        total_captured = 0
        
        # Open file for writing
        with open(iq_path, 'wb') as f:
            while total_captured < n:
                # Check timeout budget
                if (time.time() - t0) > max_wait_s:
                    print(f"[CAPTURE] ERROR: timeout budget exceeded ({max_wait_s:.1f}s), aborting capture (got {total_captured}/{n} samples)")
                    break

                # Always request constant chunk size (power-of-two)
                raw = await asyncio.to_thread(self.orch.sdr.read_samples, chunk)

                if raw is None or raw.size == 0:
                    # Timeout: cooperative yield only (no wall-time sleep)
                    await asyncio.sleep(0)
                    continue

                # Convert real → complex if needed
                if raw.dtype.kind != "c":
                    samples = np.empty(raw.size, dtype=np.complex64)
                    samples.real = raw
                    samples.imag = 0.0
                else:
                    samples = raw.astype(np.complex64, copy=False)

                # Write what we got (slice to exact remaining length if needed)
                take = min(samples.size, n - total_captured)
                samples[:take].tofile(f)
                total_captured += take

                # Progress update every ~0.2s (smoother progress bar)
                progress_interval = max(sample_rate // 5, chunk * 2)  # At least 2 chunks
                if total_captured % progress_interval < take or total_captured == n:
                    progress_pct = (total_captured / n) * 100.0
                    if total_captured % (sample_rate // 2) == 0:  # Log less frequently
                        print(f"[CAPTURE] Progress: {total_captured}/{n} samples ({progress_pct:.1f}%)")
                    # Publish progress event
                    self.orch.bus.publish_nowait("capture_progress", {
                        "samples_captured": total_captured,
                        "samples_total": n,
                        "progress_pct": progress_pct,
                        "ts": time.time(),
                    })

                # Cooperative yield only (no wall-time sleep at high sample rates)
                await asyncio.sleep(0)

        print(f"[CAPTURE] IQ capture to disk complete: {total_captured} samples")
        return total_captured
```

**Gaps:**
- **No GPU FFT**: FFT is CPU-only (Jetson GPU not utilized)
- **No NumPy vectorization optimization**: Some loops could be vectorized
- **No profiling**: No performance profiling infrastructure

**Assessment:** Good performance optimizations, but could leverage Jetson GPU more.

---

## 3. Objective Assessment

### 3.1 Standalone SDR Software for Observation & Capture ⭐⭐⭐⭐⭐ (5/5)

**Status: ✅ FULLY MET**

- Real-time FFT/waterfall visualization (30 FPS)
- Manual capture with full RF control
- Web-based UI for monitoring
- Proper SDR hardware abstraction
- Memory-efficient capture pipeline

**Verdict:** Excellent standalone SDR software.

### 3.2 Tripwire Integration (Armed Mode Auto-Capture) ⭐⭐⭐⭐⭐ (5/5)

**Status: ✅ FULLY MET**

- WebSocket connection for node registration
- HTTP endpoint for event ingestion
- Proper policy enforcement (only confirmed events)
- Rate limiting and cooldowns
- Metadata preservation
- ATAK integration for alerts

**Verdict:** Excellent Tripwire integration with proper safeguards.

### 3.3 ML Inference for Classification ⭐⭐ (2/5)

**Status: ❌ NOT MET (Infrastructure Ready, No Model)**

- ✅ ONNX Runtime infrastructure installed
- ✅ Classification pipeline integrated
- ✅ Spectrogram generation (512x512, noise-floor normalized)
- ✅ Dataset export system
- ❌ **NO TRAINED MODEL** (using dummy/stub)
- ❌ CPU-only inference (no GPU acceleration)

**Verdict:** **CRITICAL GAP**. Infrastructure is excellent, but classification is non-functional without a trained model.

---

## 4. Critical Findings

### 4.1 Must-Fix (Blockers)

1. **ML Classification Not Functional**
   - **Issue**: No trained model, only stub classifier
   - **Impact**: Objective "ML inference for classification" is not met
   - **Recommendation**: Train and deploy ONNX model, or document that classification is future work

2. **No Automated Testing**
   - **Issue**: No unit tests, integration tests, or CI/CD
   - **Impact**: Risk of regressions, difficult to verify fixes
   - **Recommendation**: Add pytest-based test suite for critical paths

### 4.2 Should-Fix (High Priority)

1. **Error Recovery**
   - **Issue**: No automatic stream recovery or device reconnection
   - **Impact**: Manual intervention required on SDR failures
   - **Recommendation**: Add automatic retry logic and health-based degradation

2. **Logging Infrastructure**
   - **Issue**: Uses `print()` instead of proper logging
   - **Impact**: Difficult to debug production issues, no log levels
   - **Recommendation**: Migrate to Python `logging` module with file/console handlers

3. **GPU Acceleration**
   - **Issue**: FFT and ONNX inference are CPU-only
   - **Impact**: Underutilizes Jetson Orin Nano capabilities
   - **Recommendation**: Add TensorRT provider for ONNX, consider cuFFT for FFT

### 4.3 Nice-to-Have (Low Priority)

1. **DC Offset Correction**: Real-time DC offset removal
2. **IQ Imbalance Calibration**: I/Q imbalance correction
3. **GPS Timestamping**: GPS sync for capture timestamps
4. **Auto-Gain Optimization**: Automatic gain selection
5. **Type Checking**: Add mypy/pyright for static analysis

---

## 5. Recommendations

### 5.1 Immediate Actions

1. **Deploy Trained ML Model**
   - Train ONNX model on collected dataset
   - Place in `spear_edge/ml/models/spear_dummy.onnx` (or rename)
   - Verify classification pipeline end-to-end

2. **Add Basic Testing**
   - Unit tests for signal processing (FFT, spectrogram)
   - Integration test for capture pipeline
   - Mock SDR for testing without hardware

3. **Improve Logging**
   - Replace `print()` with `logging` module
   - Add log levels (DEBUG, INFO, WARNING, ERROR)
   - Add file rotation for production

### 5.2 Short-Term Improvements (1-3 months)

1. **Error Recovery**
   - Automatic stream re-initialization on failure
   - SDR device reconnection logic
   - Health-based sample rate reduction

2. **GPU Acceleration**
   - TensorRT provider for ONNX Runtime
   - Benchmark CPU vs GPU inference

3. **Performance Profiling**
   - Add cProfile for performance analysis
   - Identify bottlenecks in capture pipeline

### 5.3 Long-Term Enhancements (3-6 months)

1. **Advanced RF Features**
   - DC offset correction
   - IQ imbalance calibration
   - Frequency calibration

2. **Testing Infrastructure**
   - CI/CD pipeline
   - Automated hardware-in-the-loop tests
   - Performance regression tests

3. **Documentation**
   - API documentation (OpenAPI/Swagger)
   - Architecture diagrams
   - Deployment guide

---

## 6. Conclusion

SPEAR-Edge is a **well-engineered SDR system** that **largely meets** its objectives:

✅ **Standalone SDR Software**: Excellent - fully functional  
✅ **Tripwire Integration**: Excellent - proper policy enforcement and rate limiting  
❌ **ML Classification**: **Not functional** - infrastructure ready but no trained model

### Overall Verdict

**SPEAR-Edge meets 2 of 3 core objectives** (67%). The ML classification gap is significant but the infrastructure is excellent and ready for model deployment.

**Recommendation:** Deploy a trained ONNX model to complete the classification objective. The codebase is production-ready for standalone SDR use and Tripwire integration.

---

## Appendix: Code Quality Metrics

- **Lines of Code**: ~8,000+ (estimated)
- **Modules**: 20+ (well-organized)
- **Type Hints**: Extensive (good)
- **Docstrings**: Moderate (adequate)
- **Tests**: None (gap)
- **Dependencies**: Minimal (fastapi, numpy, onnxruntime)

**Maintainability**: ⭐⭐⭐⭐ (4/5) - Well-structured, but lacks tests
