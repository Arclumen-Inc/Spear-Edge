# spear_edge/core/scan/scan_task.py
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class ScanTask:
    """
    Consumes IQ from ring buffer and emits *truth* FFT frames at fixed FPS.

    Design goals:
      - Produce stable, ML-friendly spectral truth (no per-frame renormalization tricks).
      - Never block the FFT loop (subscribers can be slow; frames will be dropped/overwritten).
      - Keep UI-specific "make it pop" out of this module (we'll build a UI view task separately).
    """

    def __init__(
        self,
        ring,
        center_freq_hz: int,
        sample_rate_sps: int,
        fft_size: int = 2048,
        fps: float = 15.0,
        calibration_offset_db: float = 0.0,
    ):
        self.ring = ring
        self.center_freq_hz = int(center_freq_hz)
        self.sample_rate_sps = int(sample_rate_sps)
        self.fft_size = int(fft_size)
        self.fps = float(fps)

        # Subscribers receive dict frames. They may be sync or async functions.
        self._subs: List[Callable[[Dict[str, Any]], Any]] = []

        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Window & frequency axis are fixed for a given configuration.
        # Use Nuttall window to match SDR++ default
        try:
            from scipy.signal import windows
            self._win = windows.nuttall(self.fft_size).astype(np.float32)
        except ImportError:
            # Fallback to Hann if scipy not available
            self._win = np.hanning(self.fft_size).astype(np.float32)
        self._freqs = (
            np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / self.sample_rate_sps))
            + self.center_freq_hz
        ).astype(np.float64)
        # Cache the list to avoid rebuilding every frame
        self._freqs_list = self._freqs.tolist()

        # Normalization constants for stable scaling
        # Window coherent gain (for magnitude normalization)
        # Nuttall window coherent gain ~ 0.363 (matches SDR++ default)
        self._coherent_gain = float(np.sum(self._win)) / self.fft_size
        self._window_sum = float(np.sum(self._win))  # Sum of window for coherent gain normalization (SDR++ style)
        self._win_energy = float(np.sum(self._win * self._win))  # sum(win^2) - kept for reference
        self._eps = 1e-12
        
        # Spectrum averaging (EMA) for noise floor estimation only
        self._avg_alpha = 0.25  # averaging factor (tune 0.1..0.4)
        self._avg_db = None
        
        # SDR++-style smoothing parameters
        # FFT smoothing: 100 = alpha = 1/100 = 0.01 (heavy smoothing for display)
        # SNR smoothing: 20 = alpha = 1/20 = 0.05 (moderate smoothing for SNR)
        self._fft_smooth_alpha = 0.01  # FFT display smoothing (matches SDR++ smoothing=100)
        self._snr_smooth_alpha = 0.05  # SNR smoothing (matches SDR++ SNR smoothing=20)
        self._fft_smoothed = None  # Smoothed FFT for display
        self._snr_smoothed = None  # Smoothed SNR value
        
        # No calibration offset - display raw dBFS values from bladeRF
        # The bladeRF hardware should work correctly without mathematical adjustments
        self._calibration_offset_db = 0.0

        # Latest truth frame queue (optional consumer pattern; used by UI view task later)
        # "Latest wins": we overwrite if full.
        self.latest: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1)

    def subscribe(self, cb: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a callback to receive truth FFT frames."""
        self._subs.append(cb)

    def is_running(self) -> bool:
        return bool(self._running and self._task and not self._task.done())

    async def start(self) -> None:
        if self.is_running():
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="scan_fft_task")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _offer_latest(self, frame: Dict[str, Any]) -> None:
        """Overwrite queue with newest frame (never blocks)."""
        try:
            if self.latest.full():
                _ = self.latest.get_nowait()
            self.latest.put_nowait(frame)
        except Exception:
            # If queue behaves oddly, never let it break the FFT loop.
            pass

    def _deliver_to_subs(self, frame: Dict[str, Any]) -> None:
        """
        Deliver frames to subscribers without blocking the FFT loop.
        - Sync callbacks are called directly (should be fast).
        - Async callbacks are scheduled via create_task.
        Any subscriber failure is isolated.
        """
        for cb in list(self._subs):
            try:
                result = cb(frame)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                # Don't spam logs if a subscriber is flaky, but do show something useful.
                # If you prefer totally silent, remove this print.
                print("[SCAN] subscriber error:", repr(e))

    # ----------------------------
    # Main loop
    # ----------------------------

    async def _loop(self) -> None:
        period = 1.0 / max(1.0, float(self.fps))
        frame_count = 0
        empty_count = 0

        # Pre-allocate some arrays to reduce allocations (helps on Jetson).
        # Note: ring.pop returns a new array; we still minimize other intermediate allocations.
        while self._running:
            t0 = time.time()

            iq = self.ring.pop(self.fft_size)
            if iq.size < self.fft_size:
                # Not enough samples yet; yield briefly.
                empty_count += 1
                if empty_count % 100 == 0:
                    print(f"[SCAN] Waiting for samples... (empty {empty_count} times)")
                await asyncio.sleep(0.001)
                continue

            # Ensure complex64 for consistent FFT performance/memory
            if iq.dtype != np.complex64:
                iq = iq.astype(np.complex64, copy=False)

            # Remove DC offset before FFT (removes LO leakage and DC bias)
            # This prevents DC from raising the noise floor and making the FFT look "flat"
            iq = iq - np.mean(iq)

            # Window and FFT
            x = iq * self._win
            X = np.fft.fftshift(np.fft.fft(x, n=self.fft_size))

            # SDR++ coherent gain normalization (matches SDR++ behavior)
            # SDR++ divides by sum(window) instead of N to account for window energy loss
            # This ensures a full-scale sine wave peaks at exactly 0 dBFS
            # Formula: mag = |X| / sum(window), then 20*log10(mag)
            # 
            # SDR++ uses Nuttall window (coherent gain ≈ 0.363)
            # This normalization matches SDR++ and accounts for window energy loss
            # Note: Process gain from FFT size (SDR++ uses 65536 vs our configurable size) 
            # will affect absolute noise floor levels, but normalization is correct
            mag = np.abs(X) / self._window_sum
            spec_db = 20.0 * np.log10(mag + self._eps)

            # Average spectrum (EMA) for noise floor estimation only
            if self._avg_db is None or self._avg_db.shape[0] != spec_db.shape[0]:
                self._avg_db = spec_db.astype(np.float32)
            else:
                self._avg_db = (self._avg_alpha * spec_db + (1.0 - self._avg_alpha) * self._avg_db).astype(np.float32)

            # Robust noise floor estimate from averaged spectrum
            noise_floor_db = float(np.percentile(self._avg_db, 10))
            
            # SDR++-style FFT smoothing (smoothing=100, alpha=0.01)
            # Smooth the entire spectrum for display (reduces noise variance)
            if self._fft_smoothed is None or self._fft_smoothed.shape[0] != spec_db.shape[0]:
                self._fft_smoothed = spec_db.astype(np.float32)
            else:
                self._fft_smoothed = (self._fft_smooth_alpha * spec_db + (1.0 - self._fft_smooth_alpha) * self._fft_smoothed).astype(np.float32)
            
            # Calculate SNR (peak - noise floor)
            peak_power_db = float(np.max(spec_db))
            snr_db = peak_power_db - noise_floor_db
            
            # SDR++-style SNR smoothing (SNR smoothing=20, alpha=0.05)
            if self._snr_smoothed is None:
                self._snr_smoothed = snr_db
            else:
                self._snr_smoothed = (self._snr_smooth_alpha * snr_db + (1.0 - self._snr_smooth_alpha) * self._snr_smoothed)
            
            # Use smoothed FFT for display (matches SDR++ behavior)
            # Keep instant spectrum for waterfall (for temporal resolution)
            power_line = self._fft_smoothed.astype(np.float32)  # Smoothed for FFT display
            power_inst = spec_db.astype(np.float32)  # Instant for waterfall
            
            # Diagnostic: Check sample levels and peak power
            if frame_count % 150 == 0:  # Every 150 frames (~10 seconds at 15 fps)
                sample_mag_max = float(np.max(np.abs(iq)))
                sample_mag_mean = float(np.mean(np.abs(iq)))
                sample_mag_std = float(np.std(np.abs(iq)))
                peak_power_db = float(np.max(power_line))
                peak_power_idx = int(np.argmax(power_line))
                # Check if there are any bins significantly above noise floor
                signal_bins = np.sum(power_line > (noise_floor_db + 6.0))  # Bins 6+ dB above noise
                # Calculate FFT magnitude statistics for debugging
                mag_max = float(np.max(mag))
                mag_mean = float(np.mean(mag))
                # Calculate theoretical noise floor for comparison
                # For coherent gain normalization: mag = |X| / sum(window)
                # For white noise: |X| ≈ sqrt(N) per bin, so mag ≈ sqrt(N)/sum(window)
                # For Nuttall window: sum(window) ≈ 0.363*N, so mag ≈ sqrt(N)/(0.363*N) = 2.75/sqrt(N)
                # dB = 20*log10(2.75/sqrt(N)) = 20*log10(2.75) - 10*log10(N) = 8.79 - 10*log10(N)
                # For N=1024: theoretical ≈ 8.79 - 30.1 = -21.3 dBFS
                # For N=4096: theoretical ≈ 8.79 - 36.1 = -27.3 dBFS
                nuttall_coherent_gain = 0.363  # Approximate for Nuttall window
                theoretical_floor = 20.0 * np.log10(1.0 / nuttall_coherent_gain) - 10.0 * np.log10(self.fft_size)
                print(f"[SCAN] Diagnostics: samples max={sample_mag_max:.6f} mean={sample_mag_mean:.6f} std={sample_mag_std:.6f}, "
                      f"FFT mag max={mag_max:.6e} mean={mag_mean:.6e}, "
                      f"peak={peak_power_db:.1f} dBFS@bin{peak_power_idx}, floor={noise_floor_db:.1f} dBFS (theoretical={theoretical_floor:.1f} dBFS), "
                      f"range={peak_power_db - noise_floor_db:.1f} dB, SNR={self._snr_smoothed:.1f} dB (smoothed), signal_bins={signal_bins}")

            # Build truth frame
            # Do NOT send giant freqs array every frame (client can reconstruct)
            # FFT line uses smoothed spectrum (SDR++ style), waterfall uses instant
            frame: Dict[str, Any] = {
                "ts": time.time(),
                "center_freq_hz": self.center_freq_hz,
                "sample_rate_sps": self.sample_rate_sps,
                "fft_size": self.fft_size,
                # "freqs_hz": ...  # Removed - client can compute from center_freq, sample_rate, fft_size
                "power_dbfs": power_line.tolist(),  # Smoothed spectrum for FFT line (SDR++ smoothing=100)
                "power_inst_dbfs": power_inst.tolist(),  # Instant spectrum for waterfall
                "noise_floor_dbfs": noise_floor_db,
                "snr_db": float(self._snr_smoothed),  # Smoothed SNR (SDR++ SNR smoothing=20)
                # No calibration - using raw bladeRF values
                "calibration_offset_db": 0.0,
                "power_units": "dBFS",
            }

            # Offer to "latest" queue and subscribers (non-blocking)
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames (~3 seconds at 10 fps)
                # Display in dBm if calibrated (offset != 0), otherwise dBFS
                unit = "dBm" if abs(self._calibration_offset_db) > 0.1 else "dBFS"
                print(f"[SCAN] Processed {frame_count} frames, noise_floor={noise_floor_db:.1f} {unit} (offset={self._calibration_offset_db:.1f} dB)")
            
            self._offer_latest(frame)
            self._deliver_to_subs(frame)

            # Maintain target FPS
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, period - elapsed))
