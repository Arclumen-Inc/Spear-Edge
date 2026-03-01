# FFT and Waterfall Code Reference

This document contains all code related to FFT processing, waterfall rendering, and SDR settings that influence the FFT and waterfall display in SPEAR-Edge.

---

## Table of Contents

1. [Backend FFT Processing](#backend-fft-processing)
2. [Frontend FFT/Waterfall Rendering](#frontend-fftwaterfall-rendering)
3. [SDR Sample Reading and Scaling](#sdr-sample-reading-and-scaling)
4. [SDR Configuration Settings](#sdr-configuration-settings)
5. [WebSocket Data Transmission](#websocket-data-transmission)
6. [Data Models](#data-models)
7. [Application Settings](#application-settings)
8. [RX Task (Sample Collection)](#rx-task-sample-collection)

---

## Backend FFT Processing

**File:** `spear_edge/core/scan/scan_task.py`

### Class: ScanTask

Main FFT processing class that consumes IQ samples from ring buffer and produces FFT frames.

```python
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
        
        # No calibration offset - display raw dBFS values from bladeRF
        # The bladeRF hardware should work correctly without mathematical adjustments
        self._calibration_offset_db = 0.0

        # Latest truth frame queue (optional consumer pattern; used by UI view task later)
        # "Latest wins": we overwrite if full.
        self.latest: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1)

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
            # RF ENGINEERING: Proper power spectral density normalization
            # CS16 samples are scaled to [-1, 1] by dividing by 32768.0 in read_samples()
            # Standard FFT power normalization: P = |X|^2 / (N * win_energy)
            # - Divide by FFT size (N) to get power per bin (not power spectral density, but normalized power)
            # - Divide by window energy to account for windowing loss (Hanning window reduces power)
            # This ensures power levels are independent of FFT size and properly scaled
            # For a full-scale signal, this should give power_db ≈ 0 dBFS
            P = (np.abs(X) ** 2) / (self.fft_size * max(self._win_energy, self._eps))

            # Convert to dBFS (dB relative to full scale)
            # Full-scale signal (|sample| = 1.0) → P ≈ 1.0 → power_db ≈ 0 dBFS
            # No calibration offset applied - display raw values from bladeRF hardware
            power_db = 10.0 * np.log10(P + self._eps)

            # Robust noise floor estimate from truth spectrum (for later consumers)
            noise_floor_db = float(np.percentile(power_db, 10))
            
            # Diagnostic: Check sample levels and peak power
            if frame_count % 150 == 0:  # Every 150 frames (~10 seconds at 15 fps)
                sample_mag_max = float(np.max(np.abs(iq)))
                sample_mag_mean = float(np.mean(np.abs(iq)))
                sample_mag_std = float(np.std(np.abs(iq)))
                peak_power_db = float(np.max(power_db))
                peak_power_idx = int(np.argmax(power_db))
                # Check if there are any bins significantly above noise floor
                signal_bins = np.sum(power_db > (noise_floor_db + 6.0))  # Bins 6+ dB above noise
                print(f"[SCAN] Diagnostics: samples max={sample_mag_max:.6f} mean={sample_mag_mean:.6f} std={sample_mag_std:.6f}, "
                      f"peak={peak_power_db:.1f} dBFS@bin{peak_power_idx}, floor={noise_floor_db:.1f} dBFS, "
                      f"range={peak_power_db - noise_floor_db:.1f} dB, signal_bins={signal_bins}")

            # Build truth frame
            # Do NOT send giant freqs array every frame (client can reconstruct)
            p_inst = power_db.astype(np.float32).tolist()
            
            # Power values are in dBFS (dB relative to full scale) - raw from bladeRF
            frame: Dict[str, Any] = {
                "ts": time.time(),
                "center_freq_hz": self.center_freq_hz,
                "sample_rate_sps": self.sample_rate_sps,
                "fft_size": self.fft_size,
                # "freqs_hz": ...  # Removed - client can compute from center_freq, sample_rate, fft_size
                "power_dbfs": p_inst,  # Power in dBFS (raw from bladeRF hardware)
                "power_inst_dbfs": p_inst,  # instant power (for waterfall)
                "noise_floor_dbfs": noise_floor_db,
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
```

### Key FFT Processing Details:

1. **Window Function**: Hanning window (`np.hanning(self.fft_size)`)
2. **FFT**: `np.fft.fftshift(np.fft.fft(x, n=self.fft_size))` - FFT with frequency shift
3. **Power Normalization**: `P = (np.abs(X) ** 2) / (self.fft_size * max(self._win_energy, self._eps))`
4. **dBFS Conversion**: `power_db = 10.0 * np.log10(P + self._eps)`
5. **Noise Floor**: 10th percentile of power spectrum
6. **Output Units**: dBFS (decibels relative to full scale)

---

## Frontend FFT/Waterfall Rendering

**File:** `spear_edge/ui/web/app.js`

### Constants

```javascript
// ------------------------------
// CONSTANTS
// ------------------------------
const FFT_HEIGHT_FRAC   = 0.40; // 40% FFT, 60% waterfall
const DB_MIN            = -140;
const DB_MAX            = 20;
const FFT_SMOOTH_ALPHA  = 0.18;
const WF_NOISE_PCT      = 0.20;
const WF_NF_SMOOTH      = 0.06;
const WF_MIN_REL_DB     = 5;
const WF_MAX_REL_DB     = 35;
const WF_GAMMA          = 1.35;
const WF_FADE_ALPHA     = 0.0010;
```

### State Variables

```javascript
let lastSpectrum        = null;
let smoothedNoiseFloor  = null;

// Calibration metadata (set from WebSocket hello message)
let globalCalibrationOffset = 0.0;
let globalPowerUnits = "dBFS";

// Waterfall display controls
let wfBrightness = 0;  // Offset in dB (-50 to +50)
let wfContrast = 1.0;  // Contrast multiplier (0.1 to 3.0)
```

### Main Spectrum Drawing Function

```javascript
function drawSpectrum(frame) {
  if (!frame) return;

  // ----------------------------
  // COMPATIBILITY SHIM
  // Accept older / alternate backend field names
  // ----------------------------
  if (!frame.power_dbfs && Array.isArray(frame.power_db)) frame.power_dbfs = frame.power_db;
  if (!frame.power_dbfs && Array.isArray(frame.power)) frame.power_dbfs = frame.power;

  if (!frame.power_inst_dbfs && Array.isArray(frame.power_inst_db)) frame.power_inst_dbfs = frame.power_inst_db;
  if (!frame.power_inst_dbfs && Array.isArray(frame.power_inst)) frame.power_inst_dbfs = frame.power_inst;

  if (!frame.freqs_hz && Array.isArray(frame.freqs)) frame.freqs_hz = frame.freqs;

  // Guard: must have at least a few bins
  if (!ctx || !canvas || !frame.power_dbfs || frame.power_dbfs.length < 8) return;
  
  if (canvas.width < 50 || canvas.height < 50) {
    resizeCanvas(true);
    return;
  }
  
  // Compute device-space coordinates once
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  const pxW = canvas.width;
  const pxH = canvas.height;
  const pxFftH = Math.floor(cssH * FFT_HEIGHT_FRAC * dpr);
  const pxWfH  = pxH - pxFftH;

  // CSS-space for FFT drawing
  const w = (canvas.width / dpr) || 1;
  const h = (canvas.height / dpr) || 1;
  const fftH = Math.floor(h * FFT_HEIGHT_FRAC);
  const wfH = h - fftH;

  // Pick sources: max-hold for FFT, instant for waterfall
  const fftArr = frame.power_dbfs;  // max-hold (for FFT line)
  const wfArr = frame.power_inst_dbfs || frame.power_dbfs;  // instant (for waterfall)
  
  // Diagnostic logging (throttled to every 2 seconds)
  if (!window._lastFftDiag || (Date.now() - window._lastFftDiag) > 2000) {
    const fftMin = Math.min(...fftArr.map(Number).filter(Number.isFinite));
    const fftMax = Math.max(...fftArr.map(Number).filter(Number.isFinite));
    const fftMean = fftArr.map(Number).filter(Number.isFinite).reduce((a, b) => a + b, 0) / fftArr.length;
    const noiseFloor = frame.noise_floor_dbfs !== undefined ? frame.noise_floor_dbfs : fftMin;
    console.log(`[FFT UI] Range: ${fftMin.toFixed(1)} to ${fftMax.toFixed(1)} dBFS, mean: ${fftMean.toFixed(1)} dBFS, floor: ${noiseFloor.toFixed(1)} dBFS, dynamic_range: ${(fftMax - noiseFloor).toFixed(1)} dB`);
    window._lastFftDiag = Date.now();
  }

  // Determine power units from frame metadata or global calibration FIRST
  const calibrationOffset = frame.calibration_offset_db !== undefined ? 
                            Number(frame.calibration_offset_db) : 
                            globalCalibrationOffset;
  const powerUnits = frame.power_units || 
                     globalPowerUnits ||
                     (Math.abs(calibrationOffset) > 0.1 ? "dBm" : "dBFS");

  // ================================
  // WATERFALL (DPR-CORRECT, STABLE)
  // Must happen FIRST - before FFT clearing/drawing
  // ================================
  if (frame.power_inst_dbfs && wfArr && wfArr.length > 0 && pxWfH > 0) {
    // Reset transform to identity for device-pixel operations
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Scroll waterfall down by 1 device pixel
    if (pxWfH > 1) {
      ctx.drawImage(
        canvas,
        0, pxFftH,
        pxW, pxWfH - 1,
        0, pxFftH + 1,
        pxW, pxWfH - 1
      );
    }

    // ---- Stable waterfall scaling (pre-WebGL behavior) ----
    // Use 10th percentile for noise floor (more robust than median for waterfall)
    const wfSorted = wfArr
      .map(Number)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const wfFloorIdx = Math.max(0, Math.floor(wfSorted.length * 0.10));
    const wfNoiseFloor = wfSorted.length ? wfSorted[wfFloorIdx] : -75;
    
    if (smoothedNoiseFloor == null) {
      smoothedNoiseFloor = wfNoiseFloor;
    } else {
      // Slow drift toward actual noise floor
      smoothedNoiseFloor = 0.95 * smoothedNoiseFloor + 0.05 * wfNoiseFloor;
    }

    // Waterfall range: noise floor - 15 dB to noise floor + 50 dB
    // This gives good visibility of signals while showing noise floor context
    const wfDbMin = smoothedNoiseFloor - 15;
    const wfDbMax = smoothedNoiseFloor + 50;
    const wfRange = Math.max(1, wfDbMax - wfDbMin);

    // Draw ONE row (device space)
    const row = ctx.createImageData(pxW, 1);
    const data = row.data;
    const nBins = wfArr.length;

    for (let x = 0; x < pxW; x++) {
      const idx = Math.min(Math.floor(x * nBins / pxW), nBins - 1);
      const db = Number(wfArr[idx]);

      if (!Number.isFinite(db)) continue;

      // Normalize to 0-1 range
      let t = (db - wfDbMin) / wfRange;
      
      // Apply brightness: shift the normalized value
      // Brightness is in dB, convert to normalized offset
      const brightnessOffset = wfBrightness / wfRange;
      t = t + brightnessOffset;
      
      // Apply contrast: center around 0.5, multiply, then restore
      t = ((t - 0.5) * wfContrast) + 0.5;
      
      // Clamp and apply gamma
      t = Math.max(0, Math.min(1, t));
      t = Math.pow(t, WF_GAMMA);

      const o = x * 4;
      data[o + 0] = Math.floor(30 * t);
      data[o + 1] = Math.floor(255 * t);
      data[o + 2] = Math.floor(10 * t);
      data[o + 3] = 255;
    }

    ctx.putImageData(row, 0, pxFftH);
    ctx.restore();
  }

  // ---------
  // Visual leveling (stable / clamped) for FFT
  // - Estimate noise floor from FFT trace ONLY
  // - Use a low percentile so brief FHSS energy doesn't pull the floor
  // - Clamp offset motion per frame to prevent hopping
  // - Adjust target based on units (dBm vs dBFS)
  // - CRITICAL: Don't push signals DOWN - only adjust if floor is too low
  // ---------
  const TARGET_FLOOR_DB = powerUnits === "dBm" ? -95.0 : -75.0; // -95 dBm for calibrated, -75 dBFS for uncalibrated (adjusted for typical bladeRF levels)
  const DB_MIN = -120.0;
  const floorSrc = fftArr;

  // robust percentile (5%)
  const sorted = floorSrc
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  const floorIdx = Math.max(0, Math.floor(sorted.length * 0.05));
  const noiseFloorRaw = sorted.length ? sorted[floorIdx] : (powerUnits === "dBm" ? -95 : -75);

  // desired offset to place floor at TARGET_FLOOR_DB
  // CRITICAL FIX: Only apply offset if noise floor is BELOW target (push up)
  // If noise floor is ABOVE target, don't push signals down - use minimal or no offset
  let desiredOffset = TARGET_FLOOR_DB - noiseFloorRaw;
  
  // Don't push signals down - if offset is negative (would push down), limit it
  // This prevents making signals harder to see when noise floor is already high
  if (desiredOffset < 0) {
    // Noise floor is higher than target - don't push down, use small positive offset or zero
    desiredOffset = Math.max(0, desiredOffset + 5.0); // Allow small adjustment but don't go negative
  }

  if (window._fftVisOffset === undefined) {
    window._fftVisOffset = desiredOffset;
  } else {
    // clamp movement (dB per rendered frame)
    const delta = desiredOffset - window._fftVisOffset;
    const MAX_STEP_DB = 0.25; // smaller = more stable
    const step = Math.max(-MAX_STEP_DB, Math.min(MAX_STEP_DB, delta));
    window._fftVisOffset += step;
  }

  const visOffset = window._fftVisOffset || 0.0;

  // Apply offset ONLY for FFT drawing
  const p = fftArr.map(v => {
    const x = Number(v) + visOffset;
    return Number.isFinite(x) ? x : DB_MIN;
  });

  // Fixed display range for FFT - widened to show more dynamic range
  // Typical bladeRF noise floor: -70 to -75 dBFS, signals: -50 to -10 dBFS
  const dbMin = -100;  // Extended lower to see noise floor better
  const dbMax = -10;   // Extended upper to see strong signals

  // Clear ONLY the FFT area (never the waterfall) - device space
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, pxFftH);
  ctx.restore();

  // Header
  ctx.fillStyle = "#00ff88";
  ctx.font = "12px monospace";
  ctx.textAlign = "left";
  if (Number.isFinite(frame.center_freq_hz)) {
    ctx.fillText("Center: " + (frame.center_freq_hz / 1e6).toFixed(3) + " MHz", 8, 14);
    ctx.fillText("Units: " + powerUnits, 8, 28);
  }

  // Center marker
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.beginPath();
  ctx.moveTo(w / 2, 0);
  ctx.lineTo(w / 2, fftH);
  ctx.stroke();

  // Smoothing for FFT trace
  const dbNow = p.map(v => clamp(v, dbMin, dbMax));
  if (!lastSpectrum || lastSpectrum.length !== dbNow.length)
    lastSpectrum = dbNow.slice();
  } else {
    for (let i = 0; i < dbNow.length; i++) {
      lastSpectrum[i] = FFT_SMOOTH_ALPHA * dbNow[i] + (1 - FFT_SMOOTH_ALPHA) * lastSpectrum[i];
    }
  }

  // FFT trace
  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 2;
  ctx.shadowColor = "#00ff88";
  ctx.shadowBlur = 6;
  ctx.beginPath();
  for (let i = 0; i < lastSpectrum.length; i++) {
    const x = (i / (lastSpectrum.length - 1)) * w;
    const t = (lastSpectrum[i] - dbMin) / (dbMax - dbMin);
    const y = (1 - clamp(t, 0, 1)) * fftH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Axes
  drawPowerAxis(ctx, fftH, w, dbMin, dbMax, powerUnits);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath();
  ctx.moveTo(0, fftH + 0.5);
  ctx.lineTo(w, fftH + 0.5);
  ctx.stroke();
  drawFreqAxis(ctx, fftH, w, frame);
}
```

### Key Frontend Display Settings:

1. **FFT Display Range**: -100 to -10 dBFS (90 dB range)
2. **FFT Smoothing**: Alpha = 0.18 (exponential moving average)
3. **FFT Auto-scaling**: Target floor = -75 dBFS (for dBFS), prevents pushing signals down
4. **Waterfall Range**: Noise floor - 15 dB to noise floor + 50 dB (65 dB dynamic range)
5. **Waterfall Gamma**: 1.35 (for color mapping)
6. **Waterfall Noise Floor**: 10th percentile, smoothed with 0.95/0.05 ratio

---

## SDR Sample Reading and Scaling

**File:** `spear_edge/core/sdr/bladerf_native.py`

### Sample Reading Function

```python
def read_samples(self, num_samples: int) -> np.ndarray:
    """
    Read samples from bladeRF.
    Returns empty array on timeout/error (never throws).
    
    For dual channel mode, returns channel 0 samples only.
    (Dual channel interleaved data: [ch0_i, ch0_q, ch1_i, ch1_q, ...])
    
    Uses CS16 format (int16 I/Q pairs) and converts to complex64.
    """
    if self.dev is None or not self._stream_active:
        return np.empty(0, dtype=np.complex64)

    n = int(num_samples)
    if n <= 0:
        return np.empty(0, dtype=np.complex64)

    # Ensure power-of-two (Edge requirement)
    # Round up to next power-of-two if needed
    if n & (n - 1) != 0:
        # Not a power-of-two, round up
        n = 1 << (n - 1).bit_length()
        logger.debug(f"Rounded read size to power-of-two: {n}")

    # Allocate CS16 buffer
    # For dual channel: 2 channels * 2 (I/Q) * n samples
    # For single channel: 2 (I/Q) * n samples
    buf_size = n * 2 * (2 if self.dual_channel_mode else 1)
    buf = (ctypes.c_int16 * buf_size)()

    # Read from bladeRF (blocking call)
    timeout_ms = 250  # 250ms timeout
    ret = _libbladerf.bladerf_sync_rx(
        self.dev,
        buf,
        n,  # Number of samples (per-channel)
        None,  # No metadata
        timeout_ms
    )

    if ret != 0:
        # Error or timeout
        return np.empty(0, dtype=np.complex64)

    # Convert CS16 buffer to numpy array
    arr = np.frombuffer(buf, dtype=np.int16)
    
    # Diagnostic: Log raw sample levels occasionally to verify gain is working
    if self._health_stats["successful_reads"] % 1000 == 0:  # Every 1000 reads
        raw_max = int(np.max(np.abs(arr)))
        raw_mean = int(np.mean(np.abs(arr)))
        logger.info(f"[SDR] Sample levels: raw_max={raw_max} (of 32767), raw_mean={raw_mean}, "
                   f"gain={self.gain_db:.1f} dB, lna_gain={self.lna_gain_db.get(self.rx_channel, 0)} dB")
    
    # RF ENGINEERING: SC16_Q11 format uses Q11 fixed-point (11 fractional bits)
    # For proper power calculations, we normalize samples to [-1, 1] range
    # Standard approach: scale = 1.0 / 32768.0 for int16 to float32 conversion
    # This ensures correct power levels for FFT processing
    scale = 1.0 / 32768.0  # Normalize int16 to [-1, 1] range

    if self.dual_channel_mode:
        # Dual channel: interleaved [ch0_i, ch0_q, ch1_i, ch1_q, ch0_i, ch0_q, ...]
        # Extract ch0 I/Q: indices 0, 1, 4, 5, 8, 9, ...
        i = arr[0::4].astype(np.float32, copy=False) * scale  # ch0 I
        q = arr[1::4].astype(np.float32, copy=False) * scale  # ch0 Q
    else:
        # Single channel: [I, Q, I, Q, ...]
        # I: indices 0, 2, 4, ... (even)
        # Q: indices 1, 3, 5, ... (odd)
        i = arr[0::2].astype(np.float32, copy=False) * scale
        q = arr[1::2].astype(np.float32, copy=False) * scale

    # Combine I and Q into complex array
    self._conv_buf_iq[:len(i)] = (i + 1j * q).astype(np.complex64)
    
    return self._conv_buf_iq[:len(i)].copy()
```

### Key SDR Settings That Affect FFT:

1. **Sample Scaling**: `1.0 / 32768.0` - Normalizes int16 CS16 samples to [-1, 1] float range
2. **Read Size**: Must be power-of-two (8192, 16384, 32768, etc.)
3. **Gain**: Main gain (0-60 dB) and LNA gain (0, 6, 12, 18, 24, 30 dB) affect signal levels
4. **Sample Rate**: Affects frequency resolution (bin width = sample_rate / fft_size)
5. **Bandwidth**: RF frontend bandwidth filter
6. **Center Frequency**: Sets the center of the FFT display

---

## SDR Configuration Settings

**File:** `spear_edge/core/sdr/bladerf_native.py`

### RF Configuration Order (CRITICAL)

The bladeRF requires a specific order for stream operations:

1. Set sample rate FIRST
2. Set bandwidth
3. Set frequency
4. Enable RX channel
5. **ONLY THEN** create and activate stream

```python
def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None, dual_channel: bool = False):
    """
    Configure RF parameters. Stream must be deactivated before tuning.
    """
    # ... validation ...
    
    # CRITICAL ORDER: Set RF parameters BEFORE stream setup
    for ch_num in channels_to_config:
        ch = BLADERF_CHANNEL_RX(ch_num)
        
        # 1. Sample rate FIRST
        self.dev.setSampleRate(SOAPY_SDR_RX, ch, float(self.sample_rate_sps))
        
        # 2. Bandwidth
        self.dev.setBandwidth(SOAPY_SDR_RX, ch, float(self.bandwidth_hz))
        
        # 3. Frequency
        self.dev.setFrequency(SOAPY_SDR_RX, ch, float(self.center_freq_hz))
        
        # 4. Enable RX
        self.dev.writeSetting("ENABLE_CHANNEL", "RX", "true")
        
        # 5. Gain mode and gain (if manual)
        if self.gain_mode == GainMode.MANUAL:
            self.set_gain(self.gain_db, channel=ch_num)
        
        # 6. LNA gain
        self.set_lna_gain(self.lna_gain_db.get(ch_num, 0), channel=ch_num)
        
        # 7. Bias-tee (BT200 external LNA)
        self.set_bt200_enabled(self.bt200_enabled.get(ch_num, False), channel=ch_num)
    
    # STREAM SETUP MUST HAPPEN LAST
    self._setup_stream()
```

### Gain Settings

```python
def set_gain(self, gain_db: float, channel: Optional[int] = None):
    """Set manual gain.
    
    CRITICAL: Gain can only be set when gain_mode is MANUAL.
    If AGC is enabled, this will disable AGC and set manual gain.
    """
    if self.dev is None:
        logger.warning("set_gain: Device not open")
        return
    
    self.gain_db = float(gain_db)
    ch_num = channel if channel is not None else self.rx_channel
    ch = BLADERF_CHANNEL_RX(ch_num)
    
    # CRITICAL FIX: Ensure gain_mode is MANUAL before setting gain
    if self.gain_mode != GainMode.MANUAL:
        logger.info(f"set_gain: Switching from {self.gain_mode} to MANUAL mode to set gain")
        self.gain_mode = GainMode.MANUAL
        ret_mode = _libbladerf.bladerf_set_gain_mode(self.dev, ch, BLADERF_GAIN_MGC)
        if ret_mode != 0:
            error_str = _libbladerf.bladerf_strerror(ret_mode).decode('utf-8', errors='ignore')
            logger.error(f"Failed to set gain mode to MANUAL for ch{ch_num}: {error_str}")
            return
    
    try:
        gain_int = int(self.gain_db)
        logger.debug(f"Setting gain for ch{ch_num}: {gain_int} dB")
        ret = _libbladerf.bladerf_set_gain(self.dev, ch, gain_int)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            logger.error(f"Failed to set gain for ch{ch_num} to {gain_int} dB: {error_str} (code: {ret})")
        else:
            logger.info(f"Successfully set gain for ch{ch_num}: {gain_int} dB")
    except Exception as e:
        logger.error(f"set_gain exception for ch{ch_num}: {e}", exc_info=True)
```

---

## WebSocket Data Transmission

**File:** `spear_edge/api/ws/live_fft_ws.py`

### Binary Frame Format

```python
# spear_edge/api/ws/live_fft_ws.py
import json
import struct
import time
import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

MAGIC = b"SPRF"
VERSION = 1
FLAG_HAS_INST = 0x01
HEADER_LEN = 32

_HDR = struct.Struct("<4sBBHIqIff")  
# 4s magic
# B  version
# B  flags
# H  header_len
# I  fft_size
# q  center_freq_hz
# I  sample_rate_sps
# f  ts (float32)
# f  noise_floor_dbfs

async def live_fft_ws(websocket: WebSocket, orchestrator):
    await websocket.accept()
    q = await orchestrator.bus.subscribe("live_spectrum", maxsize=2)

    # Tell client we'll send binary frames next
    await websocket.send_text(json.dumps({
        "type": "hello", 
        "proto": 1, 
        "binary": True,
        "calibration_offset_db": 0.0,
        "power_units": "dBFS",
    }))

    try:
        while True:
            # "Latest wins": if UI is slow, drop older frames
            evt = await q.get()
            try:
                while True:
                    evt = q.get_nowait()
            except Exception:
                pass

            # Pull fields
            fft_size = int(evt.fft_size)
            cf = int(evt.center_freq_hz)
            sr = int(evt.sample_rate_sps)
            ts = float(evt.ts if evt.ts is not None else time.monotonic())
            noise = float(evt.noise_floor_dbfs if evt.noise_floor_dbfs is not None else 0.0)

            # Arrays (convert lists -> float32 bytes)
            p0 = np.asarray(evt.power_dbfs, dtype=np.float32)
            inst = getattr(evt, "power_inst_dbfs", None)
            if inst is not None:
                p1 = np.asarray(inst, dtype=np.float32)
                flags = FLAG_HAS_INST
            else:
                p1 = None
                flags = 0

            # Safety: enforce expected length
            if p0.size != fft_size:
                fft_size = int(p0.size)
            if p1 is not None and p1.size != fft_size:
                p1 = None
                flags = 0

            header = _HDR.pack(
                MAGIC,
                VERSION,
                flags,
                HEADER_LEN,
                fft_size,
                cf,
                sr,
                np.float32(ts),
                np.float32(noise),
            )

            if p1 is None:
                payload = header + p0.tobytes(order="C")
            else:
                payload = header + p0.tobytes(order="C") + p1.tobytes(order="C")

            await websocket.send_bytes(payload)

    except WebSocketDisconnect:
        pass
    finally:
        await orchestrator.bus.unsubscribe("live_spectrum", q)
```

### Binary Frame Structure:

- **Header (32 bytes)**: Magic, version, flags, fft_size, center_freq, sample_rate, timestamp, noise_floor
- **Power Data**: float32 array of `power_dbfs` (max-hold for FFT)
- **Instant Power Data** (optional): float32 array of `power_inst_dbfs` (for waterfall)

---

## Data Models

**File:** `spear_edge/core/bus/models.py`

```python
@dataclass(frozen=True)
class LiveSpectrumFrame:
    ts: float
    center_freq_hz: int
    sample_rate_sps: int
    fft_size: int
    power_dbfs: List[float]  # max-hold (for FFT line)
    power_inst_dbfs: Optional[List[float]] = None  # instant (for waterfall)
    noise_floor_dbfs: Optional[float] = None
    freqs_hz: Optional[List[float]] = None  # Optional - client can compute
    calibration_offset_db: Optional[float] = None  # 0.0 = no calibration, dBFS
    power_units: Optional[str] = None  # "dBm" or "dBFS"
    meta: Optional[Dict[str, Any]] = None
```

---

## Application Settings

**File:** `spear_edge/settings.py`

```python
from pydantic import BaseModel
import os

class Settings(BaseModel):
    APP_NAME: str = "Spear Edge v1.0"
    HOST: str = os.getenv("SPEAR_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SPEAR_PORT", "8080"))

    # Scan defaults
    DEFAULT_CENTER_FREQ_HZ: int = int(os.getenv("SPEAR_CENTER_FREQ_HZ", "915000000"))
    DEFAULT_SAMPLE_RATE_SPS: int = int(os.getenv("SPEAR_SAMPLE_RATE_SPS", "2400000"))
    DEFAULT_FFT_SIZE: int = int(os.getenv("SPEAR_FFT_SIZE", "2048"))
    DEFAULT_FPS: float = float(os.getenv("SPEAR_FPS", "15.0"))
    
    # RF Calibration (disabled - using raw bladeRF values)
    CALIBRATION_OFFSET_DB: float = float(os.getenv("SPEAR_CALIBRATION_OFFSET_DB", "0.0"))

settings = Settings()
```

### Environment Variables:

- `SPEAR_CENTER_FREQ_HZ`: Default center frequency (default: 915 MHz)
- `SPEAR_SAMPLE_RATE_SPS`: Default sample rate (default: 2.4 MS/s)
- `SPEAR_FFT_SIZE`: Default FFT size (default: 2048)
- `SPEAR_FPS`: Default FFT frame rate (default: 15.0 fps)
- `SPEAR_CALIBRATION_OFFSET_DB`: Calibration offset (default: 0.0, disabled)

---

## RX Task (Sample Collection)

**File:** `spear_edge/core/scan/rx_task.py`

```python
class RxTask:
    """
    Continuously drains SDR into ring buffer using ONE dedicated thread.
    No intermediate queue, no asyncio drain loop.
    Ring buffer is thread-safe, so this is the lowest-jitter path.
    """

    def __init__(self, sdr, ring, chunk_size: int = 65536):
        self.sdr = sdr
        self.ring = ring
        self.chunk_size = int(chunk_size)

    def _read_thread(self):
        """
        Hard real-time-ish drain loop:
        - read a BIG chunk each call to reduce overhead
        - push immediately into ring
        - never block on other subsystems
        """
        # Choose a sane chunk:
        # - big enough to reduce call rate
        # - not so big that latency becomes awful
        chunk = max(16384, self.chunk_size)

        # If user sets very high SR, increase chunk automatically
        if getattr(self.sdr, "sample_rate_sps", 0) >= 10_000_000:
            chunk = max(chunk, 131072)

        while not self._stop_event.is_set():
            self.read_calls += 1
            iq = self.sdr.read_samples(chunk)

            if iq is None or iq.size == 0:
                # read_samples() returns empty on timeout/errors
                self.empty_reads += 1
                self._stop_event.wait(0.0001)
                continue

            # Push straight into ring (thread-safe)
            self.ring.push(iq)
```

### Key Settings:

- **Chunk Size**: Default 65536 samples (adjusts for high sample rates)
- **High Sample Rate**: Automatically increases chunk to 131072 for rates >= 10 MS/s

---

## Summary of Key Parameters

### Backend (FFT Processing):
- **Window**: Hanning
- **Normalization**: `P = |X|² / (N * win_energy)`
- **Noise Floor**: 10th percentile
- **Output Units**: dBFS

### Frontend (Display):
- **FFT Range**: -100 to -10 dBFS (90 dB)
- **FFT Smoothing**: Alpha = 0.18
- **FFT Auto-scaling**: Target floor = -75 dBFS (prevents pushing signals down)
- **Waterfall Range**: Noise floor - 15 dB to noise floor + 50 dB (65 dB)
- **Waterfall Gamma**: 1.35

### SDR (Sample Reading):
- **Sample Scaling**: 1.0 / 32768.0 (int16 → [-1, 1] float)
- **Read Size**: Power-of-two (8192, 16384, etc.)
- **Gain Range**: 0-60 dB (main), 0-30 dB (LNA)

### Settings (Defaults):
- **Center Frequency**: 915 MHz
- **Sample Rate**: 2.4 MS/s
- **FFT Size**: 2048
- **FPS**: 15.0

---

## Notes

1. **Critical Order**: SDR RF parameters must be set in specific order (sample rate → bandwidth → frequency → gain → stream)
2. **Power-of-Two**: All read sizes must be power-of-two for bladeRF hardware
3. **No Calibration**: System currently uses raw dBFS values (calibration offset = 0.0)
4. **Auto-scaling**: Frontend prevents pushing signals down when noise floor is high
5. **Waterfall**: Uses 10th percentile for noise floor estimation (more robust than median)
