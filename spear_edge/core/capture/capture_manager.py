from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

from spear_edge.core.bus.models import CaptureRequest, CaptureResult
from spear_edge.core.capture.spectrogram import (
    compute_spectrogram,
    compute_spectrogram_chunked,
    save_spectrogram_thumbnail,
    extract_basic_stats,
)

class CaptureManager:
    """
    Owns capture queue + artifact writing.
    Called by:
      - Tripwire ingest route (tasking)
      - UI/manual capture route (optional future)
    """

    def __init__(
        self,
        orchestrator,
        out_dir: str = "data/artifacts/captures",
        max_queue: int = 8,
        cooldown_s: float = 1.5,
    ):
        self.orch = orchestrator

        # FORCE ABSOLUTE PATH — CHANGE IF NEEDED
        self.out_dir = Path("/home/spear/spear-edgev1_0/data/artifacts/captures")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CAPTURE MGR] Initialized — writing to: {self.out_dir.resolve()}")

        self._q: asyncio.Queue[CaptureRequest] = asyncio.Queue(maxsize=max_queue)
        self._worker: Optional[asyncio.Task] = None
        self._running = False

        # Serialize SDR ownership
        self._cap_lock = asyncio.Lock()

        # Cooldown protection
        self._last_capture_ts = 0.0
        self._cooldown_s = float(cooldown_s)

        # Optional explicit state (debugging / UI)
        self.state: str = "idle"  # idle | capturing | error
        
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

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = asyncio.create_task(self._loop(), name="capture_manager_loop")
        print("[CAPTURE MGR] === WORKER STARTED === Ready for capture jobs")

    async def stop(self) -> None:
        self._running = False
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        print("[CAPTURE MGR] Worker stopped")

    # --------------------------------------------------
    # Submission API - for manual capture route
    # --------------------------------------------------
    def submit_nowait(self, req: CaptureRequest) -> bool:
        """Best-effort enqueue; returns False if queue is full."""
        print("[CAPTURE MGR] submit_nowait called with CaptureRequest:")
        print(f"  Freq: {req.freq_hz / 1e6:.3f} MHz | Duration: {req.duration_s}s | Reason: {req.reason} | Source: {req.source_node or 'unknown'}")

        if self._q.full():
            print("[CAPTURE MGR] Queue is full — rejecting capture")
            return False

        try:
            self._q.put_nowait(req)
            print("[CAPTURE MGR] *** CAPTURE JOB SUCCESSFULLY QUEUED ***")
            return True
        except Exception as e:
            print("[CAPTURE MGR] Failed to put in queue:", e)
            return False

    def queue_full(self) -> bool:
        """Check if capture queue is full."""
        return self._q.full()

    # --------------------------------------------------
    # Worker loop
    # --------------------------------------------------
    async def _loop(self) -> None:
        print("[CAPTURE MGR] Worker loop started — waiting for jobs")
        while self._running:
            try:
                print("[CAPTURE MGR] Waiting for next capture job...")
                req = await self._q.get()
                print(f"[CAPTURE MGR] *** NEW CAPTURE JOB DEQUEUED ***")
                print(f"[CAPTURE MGR] Freq: {req.freq_hz / 1e6:.3f} MHz | Duration: {req.duration_s}s | Reason: {req.reason}")

                await self._execute(req)
            except Exception as e:
                print("[CAPTURE MGR] Unexpected error in worker loop:", e)
                import traceback
                traceback.print_exc()
            finally:
                self._q.task_done()

    # --------------------------------------------------
    # Core capture execution
    # --------------------------------------------------
    async def _execute(self, req: CaptureRequest) -> None:
        async with self._cap_lock:
            now = time.time()
            print(f"[CAPTURE] Starting execution: {req.reason} @ {req.freq_hz / 1e6:.3f} MHz")

            # Cooldown enforcement
            if now - self._last_capture_ts < self._cooldown_s:
                print("[CAPTURE] Skipped due to cooldown")
                return

            # Snapshot state for guaranteed restore
            prev_mode = getattr(self.orch, "mode", "manual")
            prev_task_info = getattr(self.orch, "task_info", None)

            self.state = "capturing"
            
            # Publish capture start event
            self.orch.bus.publish_nowait("capture_start", {
                "freq_hz": req.freq_hz,
                "sample_rate_sps": req.sample_rate_sps,
                "duration_s": req.duration_s,
                "reason": req.reason,
                "source_node": req.source_node,
                "ts": time.time(),
            })
            self.orch.mode = "tasked"
            self.orch.task_info = {
                "source": req.reason,
                "source_node": req.source_node,
                "freq_hz": req.freq_hz,
                "sample_rate_sps": req.sample_rate_sps,
                "duration_s": req.duration_s,
                "scan_plan": req.scan_plan,
                "ts": req.ts,
            }

            try:
                # Pause live scan
                was_scanning, resume_params = self._snapshot_scan_params()
                if was_scanning:
                    print("[CAPTURE] Pausing live scan")
                    await self.orch.stop_scan()

                # Tune SDR
                await self.orch.open()
                
                # Extract bandwidth and gain from meta if present
                # Manual captures: from SDR controls
                # Armed captures: from Tripwire event metadata
                bandwidth_hz = None
                gain_db = None
                gain_mode = None
                if req.meta:
                    bandwidth_hz = req.meta.get("bandwidth_hz")
                    gain_db = req.meta.get("gain_db")
                    gain_mode = req.meta.get("gain_mode")
                
                print(f"[CAPTURE] Tuning SDR to {req.freq_hz / 1e6:.3f} MHz, {req.sample_rate_sps/1e6:.2f} MS/s")
                if bandwidth_hz:
                    source = "Tripwire event" if req.reason == "tripwire_armed" else "SDR controls"
                    print(f"[CAPTURE] Using bandwidth from {source}: {bandwidth_hz/1e6:.2f} MHz")
                else:
                    print(f"[CAPTURE] Using default bandwidth (sample rate)")
                self.orch.sdr.tune(
                    int(req.freq_hz),
                    int(req.sample_rate_sps),
                    int(bandwidth_hz) if bandwidth_hz else None,
                )
                
                # Apply gain settings if provided (manual captures have this, armed captures use defaults)
                if gain_mode and hasattr(self.orch.sdr, "set_gain_mode"):
                    self.orch.sdr.set_gain_mode(gain_mode)
                    source = "Tripwire event" if req.reason == "tripwire_armed" else "SDR controls"
                    print(f"[CAPTURE] Set gain mode from {source}: {gain_mode}")
                if gain_db is not None and hasattr(self.orch.sdr, "set_gain"):
                    self.orch.sdr.set_gain(gain_db)
                    source = "Tripwire event" if req.reason == "tripwire_armed" else "SDR controls"
                    print(f"[CAPTURE] Set gain from {source}: {gain_db} dB ({gain_mode or 'manual'})")
                else:
                    print(f"[CAPTURE] Using default gain (gain_db not provided in meta)")
                
                # CRITICAL: Verify stream is active and wait for it to be ready
                print("[CAPTURE] Verifying SDR stream state...")
                if not hasattr(self.orch.sdr, "rx_stream"):
                    print("[CAPTURE] ERROR: SDR has no rx_stream attribute!")
                    raise RuntimeError("SDR stream attribute missing")
                
                if self.orch.sdr.rx_stream is None:
                    print("[CAPTURE] ERROR: SDR stream is None after tuning!")
                    print("[CAPTURE] Attempting to force stream setup...")
                    # Force stream setup if it's None
                    if hasattr(self.orch.sdr, "_setup_stream"):
                        self.orch.sdr._setup_stream()
                    if self.orch.sdr.rx_stream is None:
                        raise RuntimeError("SDR stream not initialized after force setup")
                
                print(f"[CAPTURE] SDR stream is active: {self.orch.sdr.rx_stream}")
                
                # Wait for stream to be ready (bladeRF needs time after activation)
                print("[CAPTURE] Waiting for stream to be ready...")
                await asyncio.sleep(0.5)  # Increased from 0.25s
                
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
                
                # NO delay before capture - start immediately to maintain stream continuity

                # Create capture directory structure FIRST (before capture)
                ts = float(req.ts) if req.ts else time.time()
                stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(ts))
                base = (
                    f"{stamp}_"
                    f"{int(req.freq_hz)}Hz_"
                    f"{int(req.sample_rate_sps)}sps_"
                    f"{req.reason}"
                )
                cap_dir = self.out_dir / base
                cap_dir.mkdir(parents=True, exist_ok=True)
                
                # Ensure subfolders exist
                (cap_dir / "iq").mkdir(parents=True, exist_ok=True)
                (cap_dir / "features").mkdir(parents=True, exist_ok=True)
                (cap_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
                
                # IQ file path (will be written during capture)
                iq_path = cap_dir / "iq" / "samples.iq"

                # Capture IQ directly to disk (memory-efficient)
                print(f"[CAPTURE] Recording IQ for {req.duration_s}s...")
                actual_samples = await self._capture_iq_to_disk(
                    iq_path,
                    req.sample_rate_sps,
                    req.duration_s,
                )
                actual_duration_s = actual_samples / req.sample_rate_sps
                print(f"[CAPTURE] Captured {actual_samples} samples ({actual_duration_s:.2f}s)")

                # ----------------------------
                # Features + thumbnails
                # ----------------------------
                features_dir = cap_dir / "features"
                thumb_dir = cap_dir / "thumbnails"

                # ----------------------------
                # MEMORY-EFFICIENT SPECTROGRAM GENERATION
                # ----------------------------
                # Process IQ file in chunks to avoid loading entire file into memory
                # This generates downsampled spectrogram (≤512x512) directly
                # ----------------------------
                
                fft_size = 1024
                hop_size = fft_size // 4
                
                print("[CAPTURE] Computing spectrogram from IQ file (chunked processing)...")
                spec_ml, stats = compute_spectrogram_chunked(
                    iq_path=iq_path,
                    sample_rate_sps=int(req.sample_rate_sps),
                    fft_size=fft_size,
                    hop_size=hop_size,
                    chunk_size_samples=5_000_000,  # 5M samples = ~40 MB per chunk
                )
                print(f"[CAPTURE] Spectrogram computed: shape {spec_ml.shape}")
                
                # Compute triage from ML spectrogram
                triage = self._compute_triage(
                    spec_ml=spec_ml,
                    derived_stats=stats,
                    sample_rate_sps=int(req.sample_rate_sps),
                )
                
                # Classification (if signal present and not noise)
                classification = None
                if triage.get("signal_present") and not triage.get("likely_noise"):
                    if self.classifier is not None:
                        try:
                            classification = self.classifier.classify(spec_ml)
                        except Exception as e:
                            print(f"[CAPTURE] Classification failed: {e}")
                            import traceback
                            traceback.print_exc()
                            classification = None
                    else:
                        print("[CAPTURE] No classifier available, skipping classification")
                
                # Save ML tensor (enforced contract: ≤512x512, float32, ~1 MB)
                np.save(features_dir / "spectrogram.npy", spec_ml)
                
                # Generate PNG thumbnail from downsampled spectrogram
                # (This is acceptable - thumbnail doesn't need full resolution)
                thumb_path = thumb_dir / "spectrogram.png"
                requested_duration_s = req.duration_s
                save_spectrogram_thumbnail(
                    spec_ml,  # Use downsampled spectrogram
                    thumb_path,
                    center_freq_hz=int(req.freq_hz),
                    sample_rate_sps=int(req.sample_rate_sps),
                    duration_s=actual_duration_s,
                    requested_duration_s=requested_duration_s,
                    fft_size=fft_size,
                    hop_size=hop_size,
                )
                
                # Aggressive memory cleanup
                import gc
                gc.collect()
                print("[CAPTURE] Memory cleanup completed")

                # Store spectrogram axis metadata for later use (single source of truth)
                # This metadata should be used when re-rendering or analyzing spectrograms
                freq_span_hz = req.sample_rate_sps
                freq_start_hz = int(req.freq_hz) - freq_span_hz // 2
                freq_end_hz = int(req.freq_hz) + freq_span_hz // 2
                spectrogram_axes = {
                    "time_start_s": 0.0,
                    "time_end_s": actual_duration_s,  # Use actual duration, not requested
                    "freq_start_hz": freq_start_hz,
                    "freq_end_hz": freq_end_hz,
                    "fft_size": fft_size,
                    "hop_size": hop_size,
                }

                # Stats already computed from full-resolution spectrogram above

                with open(features_dir / "stats.json", "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)

                # Write files (IQ already written, just need metadata and other artifacts)
                # Note: iq_path is already set above, and IQ is already on disk
                cap_dir, iq_path, capture_json_path, spec_path_out, stats = self._write_artifacts_from_disk(
                    req, iq_path, cap_dir, str(thumb_path), spectrogram_axes, actual_duration_s, triage, classification
                )
                print(f"[CAPTURE] Files written:")
                print(f"   Directory: {cap_dir}")
                print(f"   IQ: {iq_path}")
                print(f"   Capture JSON: {capture_json_path}")
                if spec_path_out:
                    print(f"   Thumbnail: {spec_path_out}")

                # Dataset export (if classification available)
                dataset_dir = Path("data/dataset_raw")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                if classification is not None:
                    sample_dir = dataset_dir / f"{cap_dir.name}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(features_dir / "spectrogram.npy", sample_dir / "spectrogram.npy")
                    shutil.copy(cap_dir / "capture.json", sample_dir / "capture.json")
                    # Copy thumbnails directory if it exists
                    thumb_dir = cap_dir / "thumbnails"
                    if thumb_dir.exists():
                        sample_thumb_dir = sample_dir / "thumbnails"
                        sample_thumb_dir.mkdir(parents=True, exist_ok=True)
                        for thumb_file in thumb_dir.iterdir():
                            if thumb_file.is_file():
                                shutil.copy(thumb_file, sample_thumb_dir / thumb_file.name)
                    print(f"[CAPTURE] Dataset export: {sample_dir}")

                # Extract stage from request meta (for ATAK forwarding eligibility)
                stage = None
                if req.meta:
                    stage = req.meta.get("stage")
                
                # Publish result (using capture.json as meta_path for backward compatibility)
                res = CaptureResult(
                    ts=time.time(),
                    request_ts=req.ts,
                    freq_hz=req.freq_hz,
                    sample_rate_sps=req.sample_rate_sps,
                    duration_s=req.duration_s,
                    iq_path=str(iq_path),
                    meta_path=str(capture_json_path),  # Use capture.json as meta_path
                    spec_path=spec_path_out,  # Thumbnail PNG path
                    stats=stats,
                    source_node=req.source_node,
                    scan_plan=req.scan_plan,
                    stage=stage,  # Preserve stage for ATAK forwarding check
                )
                self.orch.bus.publish_nowait("capture_result", res)

                # Log for UI
                if hasattr(self.orch, "capture_log"):
                    self.orch.capture_log.append({
                        "ts": res.ts,
                        "freq_hz": res.freq_hz,
                        "duration_s": res.duration_s,
                        "source_node": res.source_node,
                        "scan_plan": req.scan_plan,
                        "reason": req.reason,
                    })

                # Resume live scan
                if was_scanning and resume_params is not None:
                    print("[CAPTURE] Resuming previous live scan")
                    await self.orch.start_scan(**resume_params)

                self._last_capture_ts = time.time()
                print("[CAPTURE] Capture completed successfully")
                
                # Publish capture complete event
                self.orch.bus.publish_nowait("capture_complete", {
                    "success": True,
                    "freq_hz": req.freq_hz,
                    "duration_s": req.duration_s,
                    "reason": req.reason,
                    "ts": time.time(),
                })

            except Exception as e:
                print("[CAPTURE] Capture failed:", e)
                import traceback
                traceback.print_exc()
                
                # Publish capture failure event
                self.orch.bus.publish_nowait("capture_complete", {
                    "success": False,
                    "error": str(e),
                    "freq_hz": req.freq_hz,
                    "reason": req.reason,
                    "ts": time.time(),
                })
            finally:
                self.state = "idle"
                self.orch.mode = prev_mode
                self.orch.task_info = prev_task_info

    # --------------------------------------------------
    # Scan snapshot / resume
    # --------------------------------------------------
    def _snapshot_scan_params(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        scan = getattr(self.orch, "_scan", None)
        if scan is None:
            return False, None

        is_running = True
        try:
            if hasattr(scan, "is_running"):
                is_running = bool(scan.is_running())
        except Exception:
            is_running = True

        if not is_running:
            return False, None

        center = getattr(scan, "center_freq_hz", None)
        rate = getattr(scan, "sample_rate_sps", None)
        fft = getattr(scan, "fft_size", 2048)
        fps = getattr(scan, "fps", 15.0)

        if center is None or rate is None:
            return True, None

        return True, {
            "center_freq_hz": int(center),
            "sample_rate_sps": int(rate),
            "fft_size": int(fft),
            "fps": float(fps),
        }

    # --------------------------------------------------
    # IQ capture (handles real or complex samples)
    # --------------------------------------------------
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

    async def _capture_iq(self, sample_rate: int, duration_s: float) -> np.ndarray:
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
        
        out = np.empty(n, dtype=np.complex64)
        i = 0
        
        # Timeout budget to prevent infinite loops
        t0 = time.time()
        max_wait_s = max(2.0, duration_s * 2.0)

        print(f"[CAPTURE] Starting IQ capture of {n} samples ({duration_s}s @ {sample_rate/1e6:.3f} MS/s)")
        print(f"[CAPTURE] Using chunk size: {chunk} samples (rate-adaptive)")

        while i < n:
            # Check timeout budget
            if (time.time() - t0) > max_wait_s:
                print(f"[CAPTURE] ERROR: timeout budget exceeded ({max_wait_s:.1f}s), aborting capture (got {i}/{n} samples)")
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

            # Copy what we got (slice to exact remaining length if needed)
            take = min(samples.size, n - i)
            out[i:i + take] = samples[:take]
            i += take

            # Progress update every ~0.2s (smoother progress bar)
            progress_interval = max(sample_rate // 5, chunk * 2)  # At least 2 chunks
            if i % progress_interval < take or i == n:
                progress_pct = (i / n) * 100.0
                if i % (sample_rate // 2) == 0:  # Log less frequently
                    print(f"[CAPTURE] Progress: {i}/{n} samples ({progress_pct:.1f}%)")
                # Publish progress event
                self.orch.bus.publish_nowait("capture_progress", {
                    "samples_captured": i,
                    "samples_total": n,
                    "progress_pct": progress_pct,
                    "ts": time.time(),
                })

            # Cooperative yield only (no wall-time sleep at high sample rates)
            await asyncio.sleep(0)

        # Trim if short
        if i < n:
            print(f"[CAPTURE] Capture ended early — got {i} samples")
            out = out[:i]

        print(f"[CAPTURE] IQ capture complete: {out.size} samples")
        return out

    # --------------------------------------------------
    # Stats + artifact writing
    # --------------------------------------------------
    def _write_sigmf_metadata(self, req: CaptureRequest, iq: np.ndarray, sigmf_path: Path) -> None:
        """Write SigMF metadata for raw IQ file."""
        sdr_info = self.orch.sdr.get_info() if hasattr(self.orch.sdr, "get_info") else {}
        
        sigmf_meta = {
            "global": {
                # Note: Stream uses CS16 format, but we store as CF32 for compatibility
                # Future: Store CS16 directly (ci16_le) to reduce disk I/O
                "core:datatype": "cf32_le",
                "core:sample_rate": int(req.sample_rate_sps),
                "core:version": "1.0.0",
                "core:num_channels": 1,
                "core:sha512": "",  # Could compute if needed
                "core:offset": 0,
                "core:description": f"SPEAR-Edge capture: {req.reason}",
                "core:author": "SPEAR-Edge",
                "core:meta_doi": "",
                "core:data_doi": "",
                "core:recorder": "SPEAR-Edge v1.0",
                "core:license": "",
                "core:hw": sdr_info.get("label", "Unknown SDR"),
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:global_index": 0,
                    "core:header_bytes": 0,
                    "core:frequency": int(req.freq_hz),
                    "core:datetime": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(req.ts)),
                }
            ],
            "annotations": [],
        }
        
        # Add RF configuration if available
        if req.meta:
            if req.meta.get("bandwidth_hz"):
                sigmf_meta["global"]["core:bandwidth"] = int(req.meta["bandwidth_hz"])
            if req.meta.get("gain_db") is not None:
                sigmf_meta["global"]["core:gain"] = float(req.meta["gain_db"])
            if req.meta.get("gain_mode"):
                sigmf_meta["global"]["core:gain_mode"] = str(req.meta["gain_mode"])
        
        sigmf_path.write_text(json.dumps(sigmf_meta, indent=2))

    def _generate_ml_features(self, iq: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Generate ML-ready features:
        - spectrogram.npy: deterministic spectrogram tensor (time_bins, freq_bins)
        - psd.npy: averaged power spectral density
        - stats.json: SNR, occupied BW, duty cycle, peak/avg power, burstiness
        """
        # Fixed parameters for deterministic ML features
        nfft = 1024
        hop = 256
        win = np.hanning(nfft).astype(np.float32)
        win_energy = float(np.sum(win * win))
        
        n = iq.size
        if n < nfft:
            # Return empty features if not enough samples
            return {
                "spectrogram": np.empty((0, nfft // 2), dtype=np.float32),
                "psd": np.empty(nfft // 2, dtype=np.float32),
                "stats": {
                    "snr_db": 0.0,
                    "occupied_bw_hz": 0.0,
                    "duty_cycle": 0.0,
                    "peak_power": 0.0,
                    "avg_power": 0.0,
                    "burstiness": 0.0,
                }
            }
        
        frames = 1 + (n - nfft) // hop
        S = np.empty((frames, nfft // 2), dtype=np.float32)
        
        # Compute spectrogram
        for k in range(frames):
            s = k * hop
            x = iq[s:s + nfft] * win
            X = np.fft.rfft(x, n=nfft)
            P = (np.abs(X) ** 2) / max(win_energy, 1e-12)
            SdB = 10.0 * np.log10(P + 1e-12)
            S[k, :] = SdB[: nfft // 2]
        
        # Average PSD
        psd = np.mean(S, axis=0)
        
        # Compute statistics
        # SNR: signal above noise floor
        noise_floor = np.percentile(S, 10)
        signal_power = np.percentile(S, 90)
        snr_db = float(signal_power - noise_floor) if signal_power > noise_floor else 0.0
        
        # Occupied bandwidth: frequency range with significant energy
        freq_bins = np.arange(nfft // 2) * (sample_rate / nfft)
        significant_mask = psd > (noise_floor + 3.0)  # 3 dB above noise
        if np.any(significant_mask):
            occupied_bw_hz = float(np.max(freq_bins[significant_mask]) - np.min(freq_bins[significant_mask]))
        else:
            occupied_bw_hz = 0.0
        
        # Duty cycle: fraction of time with significant energy
        time_energy = np.max(S, axis=1)  # Max power per time bin
        threshold = noise_floor + 6.0  # 6 dB above noise
        active_frames = np.sum(time_energy > threshold)
        duty_cycle = float(active_frames / frames) if frames > 0 else 0.0
        
        # Power statistics
        peak_power = float(np.max(S))
        avg_power = float(np.mean(S))
        
        # Burstiness: coefficient of variation of time-domain energy
        if np.std(time_energy) > 0:
            burstiness = float(np.std(time_energy) / (np.mean(time_energy) + 1e-12))
        else:
            burstiness = 0.0
        
        stats = {
            "snr_db": snr_db,
            "occupied_bw_hz": occupied_bw_hz,
            "duty_cycle": duty_cycle,
            "peak_power": peak_power,
            "avg_power": avg_power,
            "burstiness": burstiness,
        }
        
        return {
            "spectrogram": S,
            "psd": psd,
            "stats": stats,
        }

    def _write_operator_spectrogram(self, iq: np.ndarray, fs: int, req: CaptureRequest, out_path: str) -> None:
        """
        Generate operator-view spectrogram PNG with annotations.
        Falls back to PGM if PNG generation fails.
        """
        nfft = 1024
        hop = 256
        win = np.hanning(nfft).astype(np.float32)
        n = iq.size
        if n < nfft:
            raise ValueError("Not enough samples for spectrogram")
        
        frames = 1 + (n - nfft) // hop
        S = np.empty((frames, nfft // 2), dtype=np.float32)
        for k in range(frames):
            s = k * hop
            x = iq[s:s + nfft] * win
            X = np.fft.rfft(x)
            P = (np.abs(X) ** 2).astype(np.float32)
            SdB = 10.0 * np.log10(P + 1e-12)
            S[k, :] = SdB[: nfft // 2]
        
        # Contrast stretch
        lo = np.percentile(S, 5)
        hi = np.percentile(S, 99)
        img = np.clip((S - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        h, w = img.shape
        
        # Try to use PIL/Pillow for PNG, fallback to PGM
        try:
            from PIL import Image
            pil_img = Image.fromarray(img, mode='L')
            # Add annotations as text (simple approach)
            # For now, just save the image - annotations can be added later with PIL drawing
            pil_img.save(out_path, "PNG")
        except ImportError:
            # Fallback to PGM if PIL not available
            print("[CAPTURE] PIL not available, using PGM format")
            with open(out_path.replace('.png', '.pgm'), "wb") as f:
                f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
                f.write(img.tobytes())

    def _generate_vita49(self, req: CaptureRequest, iq: np.ndarray, out_path: str) -> None:
        """
        Generate VITA-49 VRT packets from IQ data.
        This is a simplified implementation - full VITA-49 would require proper packetization.
        For now, we'll create a placeholder structure.
        """
        # VITA-49 is complex - this is a placeholder
        # Full implementation would require proper packet headers, timestamps, stream IDs, etc.
        vita49_data = {
            "format": "VITA-49",
            "version": "1.0",
            "packet_count": 0,  # Would be computed from actual packetization
            "sample_rate": int(req.sample_rate_sps),
            "center_freq": int(req.freq_hz),
            "n_samples": int(iq.size),
            "note": "VITA-49 generation is a placeholder - full implementation requires packetization",
        }
        
        # Write placeholder metadata
        Path(out_path).write_text(json.dumps(vita49_data, indent=2))
        print("[CAPTURE] VITA-49 generation is a placeholder - full packetization not yet implemented")

    def _write_capture_json(self, req: CaptureRequest, iq: np.ndarray, ml_features: Dict[str, Any], 
                           json_path: Path, artifact_paths: Dict[str, Optional[str]], 
                           spectrogram_axes: Optional[Dict[str, Any]] = None, triage: Optional[Dict[str, Any]] = None, classification: Optional[Dict[str, Any]] = None) -> None:
        """
        Write comprehensive capture.json with all metadata:
        - Request provenance
        - RF configuration
        - Timing information
        - Derived stats
        - File references
        """
        sdr_info = self.orch.sdr.get_info() if hasattr(self.orch.sdr, "get_info") else {}
        sdr_config = getattr(self.orch, "sdr_config", None)
        
        # Request provenance
        request_provenance = {
            "reason": req.reason,
            "source_node": req.source_node,
            "scan_plan": req.scan_plan,
            "priority": req.priority,
            "original_timestamp": float(req.ts),
        }
        
        if req.meta:
            if req.meta.get("confidence") is not None:
                request_provenance["cue_confidence"] = float(req.meta["confidence"])
            if req.meta.get("classification"):
                request_provenance["classification"] = str(req.meta["classification"])
            if req.meta.get("timestamp"):
                request_provenance["cue_timestamp"] = float(req.meta["timestamp"])
        
        # RF configuration
        rf_config = {
            "center_freq_hz": int(req.freq_hz),
            "sample_rate_sps": int(req.sample_rate_sps),
            "rx_channel": req.rx_channel,
            "bandwidth_hz": req.meta.get("bandwidth_hz") if req.meta else None,
            "gain_mode": req.meta.get("gain_mode") if req.meta else None,
            "gain_db": req.meta.get("gain_db") if req.meta else None,
            "sdr_driver": sdr_info.get("driver", "unknown"),
            "sdr_serial": sdr_info.get("serial"),
            "sdr_label": sdr_info.get("label"),
        }
        
        # Timing
        timing = {
            "system_time": time.time(),
            "capture_timestamp": float(req.ts),
            "duration_s": req.duration_s,
            "n_samples": int(iq.size),
            "gps_time": None,  # Could be added if GPS available
            "clock_uncertainty": None,  # Could be estimated
        }
        
        # Derived stats (from ML features)
        derived_stats = ml_features["stats"].copy()
        derived_stats["n_samples"] = int(iq.size)
        derived_stats["sample_rate_sps"] = int(req.sample_rate_sps)

        # Quality metrics
        iq_stats = self._iq_stats(iq)
        quality = self._derive_quality(
            iq_stats=iq_stats,
            spec_stats=derived_stats,
            actual_duration_s=timing["duration_s"],
            requested_duration_s=req.duration_s,
        )
        
        # File references
        file_refs = {
            "iq_file": artifact_paths["iq_path"],
            "sigmf_meta": artifact_paths["sigmf_path"],
            "spectrogram_npy": artifact_paths["spectrogram_npy_path"],
            "psd_npy": artifact_paths["psd_npy_path"],
            "stats_json": artifact_paths["stats_json_path"],
            "thumbnail_png": artifact_paths["thumbnail_path"],
            "vita49_vrt": artifact_paths["vita49_path"],
        }
        
        capture_json = {
            "schema": "spear.edge.capture.v2",
            "version": "2.0",
            "request_provenance": request_provenance,
            "rf_configuration": rf_config,
            "timing": timing,
            "derived_stats": derived_stats,
            "quality": quality,
            "file_references": file_refs,
            "request": asdict(req),
        }
        
        # Add triage if available
        if triage is not None:
            capture_json["triage"] = triage
        
        # Add classification if available
        if classification is not None:
            capture_json["classification"] = classification
        
        # Add spectrogram axis metadata if available
        if spectrogram_axes:
            capture_json["spectrogram_axes"] = spectrogram_axes
        
        # Add ML feature contract metadata (critical for downstream ML)
        capture_json["ml_features"] = {
            "spectrogram_shape": [512, 512],
            "dtype": "float32",
            "normalized": "noise_floor",
        }
        
        json_path.write_text(json.dumps(capture_json, indent=2))

    def _iq_stats(self, iq: np.ndarray) -> Dict[str, Any]:
        """
        Compute basic RF health statistics from IQ.
        """
        iq = np.asarray(iq)

        # DC offsets
        dc_i = float(np.mean(iq.real))
        dc_q = float(np.mean(iq.imag))

        # Magnitude
        mag = np.abs(iq).astype(np.float64)

        rms = float(np.sqrt(np.mean(mag ** 2) + 1e-12))
        peak = float(np.max(mag))
        crest = float(peak / (rms + 1e-12))

        # Clipping proxy: fraction of samples within 0.5 dB of max
        if peak > 0:
            clip_threshold = peak * 0.944  # ~ -0.5 dB
            clip_fraction = float(np.mean(mag >= clip_threshold))
        else:
            clip_fraction = 0.0

        return {
            "dc_i": dc_i,
            "dc_q": dc_q,
            "rms": rms,
            "peak": peak,
            "crest_factor": crest,
            "clip_fraction": clip_fraction,
            "n_samples": int(iq.size),
        }

    def _iq_stats_from_file(
        self,
        iq_path: Path,
        n_samples: int,
        sample_size: int = 1_000_000,
    ) -> Dict[str, Any]:
        """
        Compute IQ stats from file by sampling (memory-efficient).
        Samples a subset of the file to compute statistics.
        """
        # Sample from beginning, middle, and end for representative stats
        sample_positions = [
            0,  # Beginning
            max(0, n_samples // 2 - sample_size // 2),  # Middle
            max(0, n_samples - sample_size),  # End
        ]
        
        all_samples = []
        for pos in sample_positions:
            if pos >= n_samples:
                continue
            # Read sample_size samples starting at pos
            read_size = min(sample_size, n_samples - pos)
            with open(iq_path, 'rb') as f:
                f.seek(pos * 8)  # complex64 = 8 bytes per sample
                data = f.read(read_size * 8)
                if len(data) == read_size * 8:
                    chunk = np.frombuffer(data, dtype=np.complex64)
                    all_samples.append(chunk)
        
        if not all_samples:
            # Fallback: read first chunk
            with open(iq_path, 'rb') as f:
                read_size = min(sample_size, n_samples)
                data = f.read(read_size * 8)
                if len(data) > 0:
                    chunk = np.frombuffer(data, dtype=np.complex64)
                    all_samples.append(chunk)
        
        if not all_samples:
            # Return default stats if file is empty
            return {
                "dc_i": 0.0,
                "dc_q": 0.0,
                "rms": 0.0,
                "peak": 0.0,
                "crest_factor": 0.0,
                "clip_fraction": 0.0,
                "n_samples": n_samples,
            }
        
        # Concatenate samples
        iq_sample = np.concatenate(all_samples)
        return self._iq_stats(iq_sample)

    def _derive_quality(
        self,
        iq_stats: Dict[str, Any],
        spec_stats: Dict[str, Any],
        actual_duration_s: float,
        requested_duration_s: float,
    ) -> Dict[str, Any]:
        """
        Derive capture quality flags and validity from RF stats.
        """
        warnings = []

        # Partial capture
        if actual_duration_s < requested_duration_s * 0.95:
            warnings.append("partial_capture")

        # Low SNR
        snr_db = spec_stats.get("snr_db", 0.0)
        if snr_db < 3.0:
            warnings.append("low_snr")

        # Clipping
        if iq_stats.get("clip_fraction", 0.0) > 0.01:
            warnings.append("clipping")

        # DC offset warning (relative to RMS)
        dc_i = abs(iq_stats.get("dc_i", 0.0))
        dc_q = abs(iq_stats.get("dc_q", 0.0))
        rms = iq_stats.get("rms", 0.0)
        if rms > 0 and (dc_i > 0.1 * rms or dc_q > 0.1 * rms):
            warnings.append("dc_offset")

        valid = True
        status = "ok" if not warnings else "degraded"

        return {
            "valid": valid,
            "status": status,
            "warnings": warnings,
            "dc_i": iq_stats.get("dc_i", 0.0),
            "dc_q": iq_stats.get("dc_q", 0.0),
            "rms": iq_stats.get("rms", 0.0),
            "crest_factor": iq_stats.get("crest_factor", 0.0),
            "clip_fraction": iq_stats.get("clip_fraction", 0.0),
            "snr_db": snr_db,
        }

    def _compute_triage(
        self,
        spec_ml: np.ndarray,
        derived_stats: Dict[str, Any],
        sample_rate_sps: int,
    ) -> Dict[str, Any]:
        """
        Lightweight signal triage from ML spectrogram (512x512).
        Determines presence, burstiness, and bandwidth class.
        """
        triage = {
            "signal_present": False,
            "bursty": False,
            "wideband": False,
            "occupied_bw_hz": 0.0,
            "likely_noise": False,
        }

        # Use SNR to decide if there's a signal at all
        snr_db = float(derived_stats.get("snr_db", 0.0))
        if snr_db < 3.0:
            triage["likely_noise"] = True
            return triage

        triage["signal_present"] = True

        # spec_ml is noise-floor normalized (median ~ 0)
        # Use threshold relative to noise
        activity_threshold_db = 6.0

        # Time axis: rows, Frequency axis: columns
        time_energy = np.max(spec_ml, axis=1)
        time_active = time_energy > activity_threshold_db
        duty = float(np.mean(time_active))

        # Bursty if active less than 30% of the time
        triage["bursty"] = duty < 0.3

        # Frequency occupancy
        freq_energy = np.max(spec_ml, axis=0)
        freq_active = freq_energy > activity_threshold_db
        active_bins = int(np.sum(freq_active))
        total_bins = freq_active.size

        # Convert bins to Hz
        occupied_fraction = active_bins / max(total_bins, 1)
        triage["occupied_bw_hz"] = float(occupied_fraction * sample_rate_sps)

        # Wideband if more than ~5% of spectrum is occupied
        triage["wideband"] = occupied_fraction > 0.05

        return triage

    def _write_artifacts(
        self,
        req: CaptureRequest,
        iq: np.ndarray,
        cap_dir: Optional[Path] = None,
        thumb_path: Optional[str] = None,
        spectrogram_axes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path, Path, Optional[str], Dict[str, Any]]:
        """
        Write capture artifacts in new tiered structure:
        capture_<id>/
        ├── iq/
        │   ├── samples.iq
        │   └── samples.sigmf-meta
        ├── features/
        │   ├── spectrogram.npy
        │   ├── psd.npy
        │   └── stats.json
        ├── interchange/
        │   └── vita49.vrt (optional)
        ├── capture.json
        └── thumbnails/
            └── spectrogram.png
        """
        # Use provided cap_dir if available, otherwise create it
        if cap_dir is None:
            ts = float(req.ts) if req.ts else time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(ts))
            base = (
                f"{stamp}_"
                f"{int(req.freq_hz)}Hz_"
                f"{int(req.sample_rate_sps)}sps_"
                f"{req.reason}"
            )
            cap_dir = self.out_dir / base
            cap_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories (if not already created)
        iq_dir = cap_dir / "iq"
        features_dir = cap_dir / "features"
        interchange_dir = cap_dir / "interchange"
        thumbnails_dir = cap_dir / "thumbnails"
        iq_dir.mkdir(exist_ok=True)
        features_dir.mkdir(exist_ok=True)
        interchange_dir.mkdir(exist_ok=True)
        thumbnails_dir.mkdir(exist_ok=True)

        # Write raw IQ
        iq_path = iq_dir / "samples.iq"
        iq.tofile(str(iq_path))

        # Write SigMF metadata
        sigmf_path = iq_dir / "samples.sigmf-meta"
        self._write_sigmf_metadata(req, iq, sigmf_path)

        # Features and thumbnails paths (already created in _execute above)
        spectrogram_npy_path = features_dir / "spectrogram.npy"
        psd_npy_path = features_dir / "psd.npy"
        stats_json_path = features_dir / "stats.json"
        
        # Load stats that were already written
        if stats_json_path.exists():
            with open(stats_json_path, "r", encoding="utf-8") as f:
                basic_stats = json.load(f)
        else:
            basic_stats = {}
        
        # Load spectrogram to compute PSD if needed (for backward compatibility)
        if spectrogram_npy_path.exists():
            spec_db = np.load(str(spectrogram_npy_path))
            psd = np.mean(spec_db, axis=0)
            if not psd_npy_path.exists():
                np.save(str(psd_npy_path), psd.astype(np.float32, copy=False))
        else:
            # Fallback: compute if not already done
            spec_db = None
            psd = None

        # Create ml_features dict for _write_capture_json compatibility
        ml_features = {
            "spectrogram": spec_db if spec_db is not None else np.empty((0, 512), dtype=np.float32),
            "psd": psd if psd is not None else np.empty(512, dtype=np.float32),
            "stats": basic_stats,
        }

        # Use provided thumb_path if available, otherwise try to find it
        if thumb_path:
            spec_path_out = thumb_path
        else:
            thumb_path_obj = thumbnails_dir / "spectrogram.png"
            spec_path_out = str(thumb_path_obj) if thumb_path_obj.exists() else None

        # Optional VITA-49 generation (for armed/high-value captures)
        vita49_path = None
        if req.reason == "tripwire_armed" or (req.meta and req.meta.get("confidence", 0) > 0.9):
            try:
                vita49_path = interchange_dir / "vita49.vrt"
                self._generate_vita49(req, iq, str(vita49_path))
            except Exception as e:
                print(f"[CAPTURE] Failed to generate VITA-49: {e}")

        # Write comprehensive capture.json
        capture_json_path = cap_dir / "capture.json"
        self._write_capture_json(req, iq, ml_features, capture_json_path, {
            "iq_path": str(iq_path),
            "sigmf_path": str(sigmf_path),
            "spectrogram_npy_path": str(spectrogram_npy_path),
            "psd_npy_path": str(psd_npy_path),
            "stats_json_path": str(stats_json_path),
            "thumbnail_path": spec_path_out,
            "vita49_path": str(vita49_path) if vita49_path else None,
        }, spectrogram_axes, None, None)

        # Return paths for backward compatibility
        return cap_dir, iq_path, capture_json_path, spec_path_out, ml_features["stats"]

    def _write_artifacts_from_disk(
        self,
        req: CaptureRequest,
        iq_path: Path,
        cap_dir: Path,
        thumb_path: str,
        spectrogram_axes: Dict[str, Any],
        actual_duration_s: float,
        triage: Optional[Dict[str, Any]] = None,
        classification: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path, Path, Optional[str], Dict[str, Any]]:
        """
        Write capture artifacts when IQ is already on disk (memory-efficient version).
        Similar to _write_artifacts but doesn't require IQ array in memory.
        """
        iq_dir = cap_dir / "iq"
        features_dir = cap_dir / "features"
        interchange_dir = cap_dir / "interchange"
        thumbnails_dir = cap_dir / "thumbnails"
        
        # IQ file is already written (iq_path)
        # Write SigMF metadata
        sigmf_path = iq_dir / "samples.sigmf-meta"
        
        # Get file size to determine sample count
        file_size = iq_path.stat().st_size
        n_samples = file_size // 8  # complex64 = 8 bytes per sample
        
        self._write_sigmf_metadata_from_disk(req, iq_path, n_samples, sigmf_path)
        
        # Features paths (already created in _execute)
        spectrogram_npy_path = features_dir / "spectrogram.npy"
        psd_npy_path = features_dir / "psd.npy"
        stats_json_path = features_dir / "stats.json"
        
        # Load stats that were already written
        if stats_json_path.exists():
            with open(stats_json_path, "r", encoding="utf-8") as f:
                basic_stats = json.load(f)
        else:
            basic_stats = {}
        
        # Load spectrogram to compute PSD if needed
        if spectrogram_npy_path.exists():
            spec_db = np.load(str(spectrogram_npy_path))
            psd = np.mean(spec_db, axis=0)
            if not psd_npy_path.exists():
                np.save(str(psd_npy_path), psd.astype(np.float32, copy=False))
        else:
            psd = None
        
        # Create ml_features dict for _write_capture_json compatibility
        ml_features = {
            "spectrogram": None,  # Not needed for JSON writing
            "psd": psd if psd is not None else np.empty(512, dtype=np.float32),
            "stats": basic_stats,
        }
        
        # Thumbnail path
        spec_path_out = thumb_path if thumb_path else None
        
        # Optional VITA-49 generation
        vita49_path = None
        if req.reason == "tripwire_armed" or (req.meta and req.meta.get("confidence", 0) > 0.9):
            try:
                vita49_path = interchange_dir / "vita49.vrt"
                # For VITA-49, we'd need to read IQ, but skip for now to save memory
                # self._generate_vita49(req, iq, str(vita49_path))
            except Exception as e:
                print(f"[CAPTURE] Failed to generate VITA-49: {e}")
        
        # Write comprehensive capture.json
        capture_json_path = cap_dir / "capture.json"
        self._write_capture_json_from_disk(
            req, n_samples, ml_features, capture_json_path, {
                "iq_path": str(iq_path),
                "sigmf_path": str(sigmf_path),
                "spectrogram_npy_path": str(spectrogram_npy_path),
                "psd_npy_path": str(psd_npy_path),
                "stats_json_path": str(stats_json_path),
                "thumbnail_path": spec_path_out,
                "vita49_path": str(vita49_path) if vita49_path else None,
            }, spectrogram_axes, actual_duration_s, triage, classification
        )
        
        return cap_dir, iq_path, capture_json_path, spec_path_out, ml_features["stats"]
    
    def _write_sigmf_metadata_from_disk(
        self, req: CaptureRequest, iq_path: Path, n_samples: int, sigmf_path: Path
    ) -> None:
        """Write SigMF metadata for raw IQ file (when IQ is on disk)."""
        sdr_info = self.orch.sdr.get_info() if hasattr(self.orch.sdr, "get_info") else {}
        
        sigmf_meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": int(req.sample_rate_sps),
                "core:version": "1.0.0",
                "core:num_channels": 1,
                "core:sha512": "",  # Could compute if needed
                "core:offset": 0,
                "core:description": f"SPEAR-Edge capture: {req.reason}",
                "core:author": "SPEAR-Edge",
                "core:meta_doi": "",
                "core:data_doi": "",
                "core:recorder": "SPEAR-Edge v1.0",
                "core:license": "",
                "core:hw": sdr_info.get("label", "Unknown SDR"),
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:global_index": 0,
                    "core:header_bytes": 0,
                    "core:frequency": int(req.freq_hz),
                    "core:datetime": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(req.ts)),
                }
            ],
            "annotations": [],
        }
        
        # Add RF configuration if available
        if req.meta:
            if req.meta.get("bandwidth_hz"):
                sigmf_meta["global"]["core:bandwidth"] = int(req.meta["bandwidth_hz"])
            if req.meta.get("gain_db") is not None:
                sigmf_meta["global"]["core:gain"] = float(req.meta["gain_db"])
            if req.meta.get("gain_mode"):
                sigmf_meta["global"]["core:gain_mode"] = str(req.meta["gain_mode"])
        
        sigmf_path.write_text(json.dumps(sigmf_meta, indent=2))
    
    def _write_capture_json_from_disk(
        self,
        req: CaptureRequest,
        n_samples: int,
        ml_features: Dict[str, Any],
        json_path: Path,
        artifact_paths: Dict[str, Optional[str]],
        spectrogram_axes: Dict[str, Any],
        actual_duration_s: float,
        triage: Optional[Dict[str, Any]] = None,
        classification: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write capture.json when IQ is on disk (memory-efficient)."""
        sdr_info = self.orch.sdr.get_info() if hasattr(self.orch.sdr, "get_info") else {}
        sdr_config = getattr(self.orch, "sdr_config", None)
        
        # Request provenance
        request_provenance = {
            "reason": req.reason,
            "source_node": req.source_node,
            "scan_plan": req.scan_plan,
            "priority": req.priority,
            "original_timestamp": float(req.ts),
        }
        
        if req.meta:
            if req.meta.get("confidence") is not None:
                request_provenance["cue_confidence"] = float(req.meta["confidence"])
            if req.meta.get("classification"):
                request_provenance["classification"] = str(req.meta["classification"])
            if req.meta.get("timestamp"):
                request_provenance["cue_timestamp"] = float(req.meta["timestamp"])
        
        # RF configuration
        rf_config = {
            "center_freq_hz": int(req.freq_hz),
            "sample_rate_sps": int(req.sample_rate_sps),
            "rx_channel": req.rx_channel,
            "bandwidth_hz": req.meta.get("bandwidth_hz") if req.meta else None,
            "gain_mode": req.meta.get("gain_mode") if req.meta else None,
            "gain_db": req.meta.get("gain_db") if req.meta else None,
            "sdr_driver": sdr_info.get("driver", "unknown"),
            "sdr_serial": sdr_info.get("serial"),
            "sdr_label": sdr_info.get("label"),
        }
        
        # Timing
        timing = {
            "system_time": time.time(),
            "capture_timestamp": float(req.ts),
            "duration_s": actual_duration_s,  # Use actual duration
            "n_samples": n_samples,
            "gps_time": None,
            "clock_uncertainty": None,
        }
        
        # Derived stats (from ML features)
        derived_stats = ml_features["stats"].copy()
        derived_stats["n_samples"] = n_samples
        derived_stats["sample_rate_sps"] = int(req.sample_rate_sps)

        # Quality metrics (compute from IQ file for memory efficiency)
        # Load a sample of IQ data to compute stats
        iq_path_str = artifact_paths.get("iq_path")
        if iq_path_str:
            iq_path_obj = Path(iq_path_str)
            iq_stats = self._iq_stats_from_file(iq_path_obj, n_samples)
        else:
            # Fallback: return default stats if IQ path not available
            iq_stats = {
                "dc_i": 0.0,
                "dc_q": 0.0,
                "rms": 0.0,
                "peak": 0.0,
                "crest_factor": 0.0,
                "clip_fraction": 0.0,
                "n_samples": n_samples,
            }
        quality = self._derive_quality(
            iq_stats=iq_stats,
            spec_stats=derived_stats,
            actual_duration_s=actual_duration_s,
            requested_duration_s=req.duration_s,
        )
        
        # File references
        file_refs = {
            "iq_file": artifact_paths["iq_path"],
            "sigmf_meta": artifact_paths["sigmf_path"],
            "spectrogram_npy": artifact_paths["spectrogram_npy_path"],
            "psd_npy": artifact_paths["psd_npy_path"],
            "stats_json": artifact_paths["stats_json_path"],
            "thumbnail_png": artifact_paths["thumbnail_path"],
            "vita49_vrt": artifact_paths["vita49_path"],
        }
        
        capture_json = {
            "schema": "spear.edge.capture.v2",
            "version": "2.0",
            "request_provenance": request_provenance,
            "rf_configuration": rf_config,
            "timing": timing,
            "derived_stats": derived_stats,
            "quality": quality,
            "file_references": file_refs,
            "request": asdict(req),
        }
        
        # Add triage if available
        if triage is not None:
            capture_json["triage"] = triage
        
        # Add classification if available
        if classification is not None:
            capture_json["classification"] = classification
        
        # Add spectrogram axis metadata if available
        if spectrogram_axes:
            capture_json["spectrogram_axes"] = spectrogram_axes
        
        # Add ML feature contract metadata
        capture_json["ml_features"] = {
            "spectrogram_shape": [512, 512],
            "dtype": "float32",
            "normalized": "noise_floor",
        }
        
        json_path.write_text(json.dumps(capture_json, indent=2))

    def _write_spectrogram_pgm(self, iq: np.ndarray, fs: int, out_path: str) -> None:
        nfft = 1024
        hop = 256
        win = np.hanning(nfft).astype(np.float32)
        n = iq.size
        if n < nfft:
            raise ValueError("Not enough samples for spectrogram")
        frames = 1 + (n - nfft) // hop
        S = np.empty((frames, nfft // 2), dtype=np.float32)
        for k in range(frames):
            s = k * hop
            x = iq[s:s + nfft] * win
            X = np.fft.rfft(x)
            P = (np.abs(X) ** 2).astype(np.float32)
            SdB = 10.0 * np.log10(P + 1e-12)
            S[k, :] = SdB[: nfft // 2]
        lo = np.percentile(S, 5)
        hi = np.percentile(S, 99)
        img = np.clip((S - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        h, w = img.shape
        with open(out_path, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
            f.write(img.tobytes())