# spear_edge/core/scan/scan_task.py
from __future__ import annotations

import asyncio
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

        # Max-hold state (FHSS visibility)
        self._max_hold = None
        self._max_hold_reset_s = 0.35  # Reset window (FHSS friendly)
        self._max_hold_last_reset = time.time()

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

            # Window and FFT
            x = iq * self._win
            X = np.fft.fftshift(np.fft.fft(x, n=self.fft_size))

            # Power (linear)
            # Normalize by window energy for stable scaling across FFT sizes/windows.
            # This is not perfect absolute calibration, but it is stable and ML-friendly.
            P = (np.abs(X) ** 2) / max(self._win_energy, self._eps)

            # Convert to dB "truth" scale (stable; no per-frame shifting)
            power_db = 10.0 * np.log10(P + self._eps)

            # Max-hold update (helps visualize FHSS hops)
            # DISABLED: Using instant power for power_dbfs instead of max-hold
            # now = time.time()
            # if (self._max_hold is None) or ((now - self._max_hold_last_reset) >= self._max_hold_reset_s):
            #     self._max_hold = power_db.copy()
            #     self._max_hold_last_reset = now
            # else:
            #     self._max_hold = np.maximum(self._max_hold, power_db)

            # Robust noise floor estimate from truth spectrum (for later consumers)
            noise_floor_db = float(np.percentile(power_db, 10))

            # Build truth frame
            # Do NOT send giant freqs array every frame (client can reconstruct)
            p = power_db.astype(np.float32).tolist()
            frame: Dict[str, Any] = {
                "ts": time.time(),
                "center_freq_hz": self.center_freq_hz,
                "sample_rate_sps": self.sample_rate_sps,
                "fft_size": self.fft_size,
                # "freqs_hz": ...  # Removed - client can compute from center_freq, sample_rate, fft_size
                "power_dbfs": p,  # instant (peak hold OFF)
                "power_inst_dbfs": p,  # instant (for waterfall)
                "noise_floor_dbfs": noise_floor_db,
            }

            # Offer to "latest" queue and subscribers (non-blocking)
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames (~3 seconds at 10 fps)
                print(f"[SCAN] Processed {frame_count} frames, noise_floor={noise_floor_db:.1f} dBFS")
            
            self._offer_latest(frame)
            self._deliver_to_subs(frame)

            # Maintain target FPS
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, period - elapsed))
