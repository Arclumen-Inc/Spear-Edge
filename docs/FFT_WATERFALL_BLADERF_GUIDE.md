# FFT, Waterfall, and bladeRF Controls Guide

This document provides a comprehensive guide to implementing FFT processing, waterfall visualization, and bladeRF hardware controls as used in SPEAR-Edge. This guide is intended for developers implementing similar functionality in other projects, such as the Manual Direction Finding page for Tripwire.

---

## Table of Contents

1. [FFT Processing (Backend)](#fft-processing-backend)
2. [Waterfall Visualization (Frontend)](#waterfall-visualization-frontend)
3. [bladeRF Hardware Controls](#bladerf-hardware-controls)
4. [WebSocket Protocol](#websocket-protocol)
5. [Implementation Checklist](#implementation-checklist)

---

## FFT Processing (Backend)

### Overview

The FFT processing pipeline converts IQ samples from the SDR into frequency-domain power spectra suitable for display. The implementation prioritizes stability, accuracy, and performance on embedded systems like the Jetson Orin Nano.

### Key Components

#### 1. Window Function

**Window Type**: Hanning (Hann) window

```python
import numpy as np

fft_size = 2048  # Typical value
window = np.hanning(fft_size).astype(np.float32)
```

**Why Hanning?**
- Lower sidelobes than rectangular window (reduces spectral leakage)
- Better frequency resolution than Blackman/Nuttall
- Good balance between resolution and sidelobe suppression
- Coherent gain ≈ 0.5 (sum ≈ 0.5 * N)

#### 2. FFT Processing Pipeline

```python
# 1. Get IQ samples from ring buffer
iq = ring_buffer.pop(fft_size)  # Complex64 array

# 2. Ensure correct dtype and size
if iq.dtype != np.complex64:
    iq = iq.astype(np.complex64, copy=False)

# 3. Optional: DC offset removal (configurable)
# WARNING: Can distort wideband signals - disable for wideband
if dc_removal_enabled:
    dc_offset = np.mean(iq)
    iq -= dc_offset

# 4. Apply window
windowed = iq * window

# 5. Compute FFT and shift to center
fft_result = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))

# 6. Calculate magnitude and normalize
# Coherent gain normalization: divide by sum(window) to account for window energy loss
window_sum = np.sum(window)
magnitude = np.abs(fft_result) / window_sum

# 7. Convert to dBFS (dB relative to full scale)
eps = 1e-12  # Avoid log(0)
power_db = 20.0 * np.log10(magnitude + eps)
```

#### 3. Frequency Axis Calculation

```python
# Calculate frequency bins
sample_rate_sps = 2400000  # Example: 2.4 MS/s
center_freq_hz = 915000000  # Example: 915 MHz

freqs_hz = (
    np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_sps))
    + center_freq_hz
).astype(np.float64)

# Frequency resolution
freq_resolution_hz = sample_rate_sps / fft_size
```

#### 4. Noise Floor Estimation

**Adaptive Percentile Method**:

```python
# Exclude edge bins (first/last 5%) - they're elevated due to window sidelobes
exclude_pct = 0.05
exclude_count = int(len(power_db) * exclude_pct)
center_spectrum = power_db[exclude_count:-exclude_count]

# Detect wideband signals (>20% of bins above threshold)
preliminary_floor = np.percentile(center_spectrum, 2)
signal_threshold = preliminary_floor + 3.0
signal_bins = np.sum(center_spectrum > signal_threshold)
signal_pct = (signal_bins / len(center_spectrum)) * 100

# Adaptive percentile based on signal type
if signal_pct > 20.0:
    # Wideband: use 2nd percentile (excludes signal energy)
    noise_floor_db = np.percentile(center_spectrum, 2)
else:
    # Narrowband: use 10th percentile (standard method)
    noise_floor_db = np.percentile(center_spectrum, 10)
```

**Why Adaptive?**
- Wideband signals (e.g., analog video) spread power across many bins
- 10th percentile would be contaminated by signal energy
- 2nd percentile gives true noise floor for wideband signals
- 10th percentile works well for narrowband signals

#### 5. Spectrum Smoothing

**SDR++-style Exponential Moving Average**:

```python
# Smoothing alpha (0.0 to 1.0)
# Lower = more smoothing, higher = less smoothing
fft_smooth_alpha = 0.1  # Default: light smoothing (good for wideband)

# Initialize smoothed spectrum
if smoothed_spectrum is None:
    smoothed_spectrum = power_db.copy()
else:
    # EMA update
    smoothed_spectrum = (
        fft_smooth_alpha * power_db + 
        (1.0 - fft_smooth_alpha) * smoothed_spectrum
    ).astype(np.float32)
```

**Two Spectra**:
- **Smoothed** (`power_dbfs`): For FFT line display (reduces noise variance)
- **Instant** (`power_inst_dbfs`): For waterfall (preserves temporal resolution)

#### 6. Edge Bin Handling

**Problem**: Edge bins (first/last 2.5%) are elevated 20+ dB due to window sidelobes.

**Solution**: Zero edge bins for display:

```python
edge_zero_pct = 0.025  # 2.5% from each edge
edge_zero_count = int(len(power_db) * edge_zero_pct)
edge_zero_value = noise_floor_db - 10.0  # Below display

power_db[:edge_zero_count] = edge_zero_value
power_db[-edge_zero_count:] = edge_zero_value
```

#### 7. Frame Rate Control

```python
target_fps = 30.0  # Frames per second
period = 1.0 / target_fps

while running:
    t0 = time.time()
    
    # Process FFT...
    
    # Maintain target FPS
    elapsed = time.time() - t0
    await asyncio.sleep(max(0.0, period - elapsed))
```

### Output Frame Structure

```python
frame = {
    "ts": time.time(),  # Timestamp
    "center_freq_hz": center_freq_hz,
    "sample_rate_sps": sample_rate_sps,
    "fft_size": fft_size,
    "power_dbfs": smoothed_spectrum.tolist(),  # Smoothed for FFT line
    "power_inst_dbfs": instant_spectrum.tolist(),  # Instant for waterfall
    "noise_floor_dbfs": noise_floor_db,
    "snr_db": peak_power_db - noise_floor_db,
    "calibration_offset_db": 0.0,  # Display offset (0.0 = raw dBFS)
    "power_units": "dBFS",
}
```

---

## Waterfall Visualization (Frontend)

### Overview

The waterfall displays a time-frequency representation where each row represents one FFT frame, with color indicating power level. The waterfall scrolls downward as new frames arrive.

### Canvas Setup

```javascript
// Get canvas and context
const canvas = document.getElementById('spectrum-canvas');
const ctx = canvas.getContext('2d');

// Handle device pixel ratio for high-DPI displays
const dpr = window.devicePixelRatio || 1;
const cssW = canvas.clientWidth;
const cssH = canvas.clientHeight;

// Set canvas size in device pixels
canvas.width = cssW * dpr;
canvas.height = cssH * dpr;

// Scale context for CSS coordinates
ctx.scale(dpr, dpr);
```

### Layout

```javascript
// Split canvas: FFT on top, waterfall below
const FFT_HEIGHT_FRAC = 0.35;  // 35% for FFT, 65% for waterfall

const fftH = Math.floor(cssH * FFT_HEIGHT_FRAC);
const wfH = cssH - fftH;
```

### Waterfall Rendering

#### 1. Scrolling Mechanism

```javascript
function drawWaterfall(frame) {
    const pxW = canvas.width;  // Device pixels
    const pxH = canvas.height;
    const pxFftH = Math.floor(cssH * FFT_HEIGHT_FRAC * dpr);
    const pxWfH = pxH - pxFftH;
    
    // Reset transform to device-pixel space
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    
    // Scroll waterfall down by 1 device pixel
    if (pxWfH > 1) {
        ctx.drawImage(
            canvas,
            0, pxFftH,           // Source: top of waterfall
            pxW, pxWfH - 1,      // Source size
            0, pxFftH + 1,       // Destination: 1 pixel down
            pxW, pxWfH - 1       // Destination size
        );
    }
    
    // Draw new row at top of waterfall...
}
```

#### 2. Noise Floor Tracking

```javascript
// Use 10th percentile for waterfall noise floor
const wfArr = frame.power_inst_dbfs;  // Instant spectrum
const wfSorted = wfArr
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
const wfFloorIdx = Math.max(0, Math.floor(wfSorted.length * 0.10));
const wfNoiseFloor = wfSorted.length ? wfSorted[wfFloorIdx] : -75;

// Smooth noise floor (slow drift)
if (smoothedNoiseFloor == null) {
    smoothedNoiseFloor = wfNoiseFloor;
} else {
    smoothedNoiseFloor = 0.95 * smoothedNoiseFloor + 0.05 * wfNoiseFloor;
}
```

#### 3. Dynamic Range

```javascript
// Waterfall range: noise floor - 15 dB to noise floor + 50 dB
const wfDbMin = smoothedNoiseFloor - 15;
const wfDbMax = smoothedNoiseFloor + 50;
const wfRange = Math.max(1, wfDbMax - wfDbMin);
```

#### 4. Color Mapping

```javascript
// Normalize power to 0-1 range
let t = (db - wfDbMin) / wfRange;

// Apply brightness (in dB, convert to normalized offset)
const brightnessOffset = wfBrightness / wfRange;
t = t + brightnessOffset;

// Apply contrast (center around 0.5, multiply, restore)
t = ((t - 0.5) * wfContrast) + 0.5;

// Clamp and apply gamma correction
t = Math.max(0, Math.min(1, t));
t = Math.pow(t, WF_GAMMA);  // Typical gamma: 1.0-1.5

// Get color from palette
const color = getPaletteColor(t, waterfallPalette);
```

#### 5. Color Palettes

**Common Palettes**:

```javascript
// Classic (blue -> cyan -> green -> yellow -> red)
const classicPalette = [
    { r: 0, g: 0, b: 0 },        // Black (noise)
    { r: 0, g: 0, b: 128 },      // Dark blue
    { r: 0, g: 128, b: 255 },     // Blue
    { r: 0, g: 255, b: 255 },    // Cyan
    { r: 0, g: 255, b: 0 },      // Green
    { r: 255, g: 255, b: 0 },    // Yellow
    { r: 255, g: 128, b: 0 },    // Orange
    { r: 255, g: 0, b: 0 },      // Red
    { r: 255, g: 255, b: 255 },  // White (saturation)
];

// Interpolate between palette colors
function getPaletteColor(t, palette) {
    const idx = t * (palette.length - 1);
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, palette.length - 1);
    const frac = idx - i0;
    
    return {
        r: Math.round(palette[i0].r + frac * (palette[i1].r - palette[i0].r)),
        g: Math.round(palette[i0].g + frac * (palette[i1].g - palette[i0].g)),
        b: Math.round(palette[i0].b + frac * (palette[i1].b - palette[i0].b)),
    };
}
```

#### 6. Drawing New Row

```javascript
// Create ImageData for one row
const row = ctx.createImageData(pxW, 1);
const data = row.data;
const nBins = wfArr.length;

for (let x = 0; x < pxW; x++) {
    // Map pixel to frequency bin
    const idx = Math.min(Math.floor(x * nBins / pxW), nBins - 1);
    const db = Number(wfArr[idx]);
    
    if (!Number.isFinite(db)) continue;
    
    // Normalize, apply brightness/contrast/gamma
    let t = (db - wfDbMin) / wfRange;
    t = t + brightnessOffset;
    t = ((t - 0.5) * wfContrast) + 0.5;
    t = Math.max(0, Math.min(1, t));
    t = Math.pow(t, WF_GAMMA);
    
    // Get color
    const color = getPaletteColor(t, waterfallPalette);
    
    // Write to ImageData
    const o = x * 4;
    data[o + 0] = color.r;
    data[o + 1] = color.g;
    data[o + 2] = color.b;
    data[o + 3] = 255;  // Alpha
}

// Draw row at top of waterfall
ctx.putImageData(row, 0, pxFftH);
ctx.restore();
```

### FFT Line Rendering

#### 1. Display Range

```javascript
// Fixed 70 dB range, autoscales to noise floor
const FFT_VIEW_RANGE_DB = 70.0;
const FFT_FLOOR_MARGIN_DB = 5.0;  // Margin below noise floor
const FFT_REFERENCE_LEVEL_DBFS = -20.0;  // Top of display

// Use backend noise floor (more accurate)
const backendFloor = frame.noise_floor_dbfs;

// Smooth floor with asymmetrical attack/release
if (fftFloorSmoothed == null) {
    fftFloorSmoothed = backendFloor;
} else {
    const floorDelta = Math.abs(backendFloor - fftFloorSmoothed);
    if (floorDelta > 15.0) {
        // Large change: reset immediately
        fftFloorSmoothed = backendFloor;
    } else {
        // Smooth update
        const alpha = (backendFloor > fftFloorSmoothed) ? 0.1 : 0.02;
        fftFloorSmoothed = (1 - alpha) * fftFloorSmoothed + alpha * backendFloor;
    }
}

// Calculate display range
const dbMinTarget = fftFloorSmoothed - FFT_FLOOR_MARGIN_DB;
const dbMax = Math.min(FFT_REFERENCE_LEVEL_DBFS, dbMinTarget + FFT_VIEW_RANGE_DB);
const dbMin = dbMax - FFT_VIEW_RANGE_DB;
```

#### 2. Trace Modes

**Instant Mode** (default):
```javascript
const ATTACK = 0.55;  // Fast rise
const DECAY = 0.96;   // Slow fall

for (let i = 0; i < spectrum.length; i++) {
    const cur = lastSpectrum[i];
    const nxt = currentSpectrum[i];
    if (nxt > cur) {
        // Fast attack
        lastSpectrum[i] = ATTACK * nxt + (1 - ATTACK) * cur;
    } else {
        // Slow decay
        lastSpectrum[i] = DECAY * cur + (1 - DECAY) * nxt;
    }
}
```

**Peak Hold Mode**:
```javascript
const PEAK_DECAY = 0.995;  // Very slow decay

for (let i = 0; i < spectrum.length; i++) {
    if (currentSpectrum[i] > peakHoldSpectrum[i]) {
        peakHoldSpectrum[i] = currentSpectrum[i];  // Fast attack
    } else {
        peakHoldSpectrum[i] = PEAK_DECAY * peakHoldSpectrum[i] + 
                             (1 - PEAK_DECAY) * currentSpectrum[i];  // Slow decay
    }
}
```

#### 3. Drawing FFT Line

```javascript
// Map spectrum to canvas coordinates
ctx.strokeStyle = "#00ff88";  // Green color
ctx.lineWidth = 2;
ctx.shadowColor = "#00ff88";
ctx.shadowBlur = 6;

ctx.beginPath();
for (let i = 0; i < spectrum.length; i++) {
    const x = (i / (spectrum.length - 1)) * cssW;
    const t = (spectrum[i] - dbMin) / (dbMax - dbMin);
    const y = (1 - Math.max(0, Math.min(1, t))) * fftH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
}
ctx.stroke();
ctx.shadowBlur = 0;
```

---

## bladeRF Hardware Controls

### CRITICAL: Configuration Order

**bladeRF requires a specific order for RF parameter configuration. The stream MUST be created/activated AFTER all RF settings are applied.**

```python
# CORRECT ORDER:
# 1. Set gain mode (if manual)
# 2. Set sample rate FIRST
# 3. Set bandwidth
# 4. Set frequency
# 5. Configure and activate stream
# 6. Set gain (AFTER stream is active)
```

### Implementation

```python
def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None):
    """
    Configure RF parameters in CRITICAL ORDER.
    """
    # Step 1: Set gain mode (if manual)
    if self.gain_mode == GainMode.MANUAL:
        bladerf_set_gain_mode(self.dev, channel, BLADERF_GAIN_MGC)
    
    # Step 2: Set sample rate FIRST
    actual_rate = ctypes.c_uint32()
    bladerf_set_sample_rate(
        self.dev, channel, sample_rate_sps, 
        ctypes.byref(actual_rate)
    )
    
    # Step 3: Set bandwidth
    actual_bw = ctypes.c_uint32()
    bladerf_set_bandwidth(
        self.dev, channel, bandwidth_hz or sample_rate_sps,
        ctypes.byref(actual_bw)
    )
    
    # Step 4: Set frequency
    bladerf_set_frequency(self.dev, channel, center_freq_hz)
    
    # Step 5: Configure and activate stream
    self._setup_stream()
    
    # Step 6: Set gain AFTER stream is active
    if self.gain_mode == GainMode.MANUAL:
        time.sleep(0.05)  # Brief delay
        bladerf_set_gain(self.dev, channel, int(self.gain_db))
```

### Key Parameters

#### Sample Rate

- **Range**: Up to 61.44 MS/s (USB 3.0)
- **Jetson Orin Nano**: Practical limit ~30-40 MS/s
- **Recommended**: 2.4 - 20 MS/s for general use
- **High Performance**: 20-40 MS/s (requires USB 3.0)

#### Bandwidth

- **Default**: Matches sample rate
- **Range**: Limited by sample rate
- **Note**: Hardware may apply different actual bandwidth

#### Frequency

- **Range**: 47 MHz - 6 GHz
- **Resolution**: Hardware-dependent (typically 1 Hz)
- **Note**: Read back actual frequency - hardware may apply different value

#### Gain Control

**Gain Modes**:
- **MANUAL (MGC)**: Fixed gain, user-controlled
- **AGC**: Automatic gain control (if supported)

**Gain Range**:
- **Typical**: 0-60 dB
- **Recommended**: 15-35 dB for bladeRF 2.0
- **LNA Gain**: Automatically optimized by `bladerf_set_gain()` (0, 6, 12, 18, 24, 30 dB steps)

**CRITICAL**: Gain must be set AFTER stream is configured, or it will be clamped to 60 dB.

#### Sample Format

**Format**: `BLADERF_FORMAT_SC16_Q11` (CS16 - interleaved int16 I/Q)

- **Bytes per complex sample**: 4 (2 bytes I + 2 bytes Q)
- **Sample range**: [-2048, 2047] (11-bit signed)
- **Normalization**: Divide by 2048.0 to get [-1.0, 1.0) range
- **Read Size**: MUST be power-of-two (8192, 16384, 32768, etc.)

```python
# Read samples
samples = sdr.read_samples(8192)  # Power-of-two!

# Convert to complex64
iq = samples.astype(np.float32) / 2048.0
iq = iq[::2] + 1j * iq[1::2]  # Interleaved I/Q
iq = iq.astype(np.complex64)
```

### Stream Lifecycle

```python
# 1. Configure RF parameters (see tune() above)
# 2. Setup stream
def _setup_stream(self):
    # Configure stream format
    bladerf_sync_config(
        self.dev,
        channel,
        BLADERF_FORMAT_SC16_Q11,
        num_buffers=16,
        buffer_size=65536,
        num_transfers=8,
        stream_timeout_ms=1000,
    )
    
    # Enable RX module
    bladerf_enable_module(self.dev, channel, True)
    
    # Stream is now active
    self._stream_active = True

# 3. Read samples (blocking)
iq = sdr.read_samples(8192)  # Power-of-two!

# 4. Deactivate stream
def _deactivate_stream(self):
    if self._stream_active:
        bladerf_enable_module(self.dev, channel, False)
        self._stream_active = False
```

### USB Buffer Configuration

**CRITICAL**: Linux USB filesystem memory pool is limited to 16MB by default.

**Calculation**:
```python
# SC16_Q11 uses 4 bytes per complex sample
total_memory_bytes = buffer_size * 4 * num_buffers

# Example: 64K samples, 16 buffers
buffer_size = 65536
num_buffers = 16
total_memory = 65536 * 4 * 16 = 4,194,304 bytes = 4 MB
```

**Recommended Configurations**:

- **High Sample Rates (20-40 MS/s)**:
  - `buffer_size = 131072` (128K samples)
  - `num_buffers = 8`
  - `num_transfers = 4`
  - Total: 4 MB

- **Medium Sample Rates (10-20 MS/s)**:
  - `buffer_size = 65536` (64K samples)
  - `num_buffers = 16`
  - `num_transfers = 8`
  - Total: 4 MB

- **Low Sample Rates (<10 MS/s)**:
  - `buffer_size = 32768` (32K samples)
  - `num_buffers = 16`
  - `num_transfers = 8`
  - Total: 2 MB

### BT200 External LNA

**CRITICAL**: BT200 is NOT connected by default. Only enable if hardware is present.

- **Gain**: ~16-20 dB additional gain
- **Control**: Bias-tee on/off
- **Safety**: Auto-disable if system gain ≤ 5 dB (prevents clipping)
- **Default**: Disabled (`bt200_enabled=False`)

---

## WebSocket Protocol

### Frame Format

**Binary Protocol** (recommended for performance):

```
Header (32 bytes):
- Magic: 4 bytes ("SPRF")
- Version: 1 byte (1)
- Flags: 1 byte (bit 0 = has instant spectrum)
- Header length: 2 bytes (32)
- FFT size: 4 bytes (uint32)
- Center frequency: 8 bytes (int64)
- Sample rate: 4 bytes (uint32)
- Timestamp: 4 bytes (float32)
- Noise floor: 4 bytes (float32)

Payload:
- Smoothed spectrum: fft_size * 4 bytes (float32 array)
- Instant spectrum (if flag set): fft_size * 4 bytes (float32 array)
```

**JSON Protocol** (alternative, easier to debug):

```json
{
    "ts": 1234567890.123,
    "center_freq_hz": 915000000,
    "sample_rate_sps": 2400000,
    "fft_size": 2048,
    "power_dbfs": [ -80.5, -79.2, ... ],
    "power_inst_dbfs": [ -80.1, -78.9, ... ],
    "noise_floor_dbfs": -85.3,
    "snr_db": 15.2,
    "calibration_offset_db": 0.0,
    "power_units": "dBFS"
}
```

### Client Implementation

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/live_fft');

ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
        // Binary frame
        const frame = parseBinaryFrame(event.data);
        drawSpectrum(frame);
    } else {
        // JSON frame or hello message
        const msg = JSON.parse(event.data);
        if (msg.type === 'hello') {
            console.log('Connected, calibration:', msg.calibration_offset_db);
        }
    }
};

function parseBinaryFrame(buffer) {
    const dv = new DataView(buffer);
    
    // Read header
    const magic = String.fromCharCode(
        dv.getUint8(0), dv.getUint8(1), 
        dv.getUint8(2), dv.getUint8(3)
    );
    if (magic !== 'SPRF') return null;
    
    const version = dv.getUint8(4);
    const flags = dv.getUint8(5);
    const headerLen = dv.getUint16(6, true);
    const fftSize = dv.getUint32(8, true);
    const centerFreqHz = Number(dv.getBigInt64(12, true));
    const sampleRateSps = dv.getUint32(20, true);
    const ts = dv.getFloat32(24, true);
    const noiseFloor = dv.getFloat32(28, true);
    
    const hasInst = (flags & 0x01) !== 0;
    
    // Read spectra
    const off0 = headerLen;
    const power0 = new Float32Array(buffer, off0, fftSize);
    const power1 = hasInst ? new Float32Array(buffer, off0 + fftSize * 4, fftSize) : null;
    
    return {
        ts,
        center_freq_hz: centerFreqHz,
        sample_rate_sps: sampleRateSps,
        fft_size: fftSize,
        power_dbfs: Array.from(power0),
        power_inst_dbfs: power1 ? Array.from(power1) : null,
        noise_floor_dbfs: noiseFloor,
        power_units: 'dBFS',
    };
}
```

---

## Implementation Checklist

### Backend (FFT Processing)

- [ ] Implement ring buffer for IQ samples
- [ ] Apply Hanning window
- [ ] Compute FFT with fftshift
- [ ] Normalize by window sum (coherent gain)
- [ ] Convert to dBFS (20 * log10)
- [ ] Calculate frequency axis
- [ ] Implement adaptive noise floor (2nd/10th percentile)
- [ ] Add spectrum smoothing (EMA)
- [ ] Zero edge bins (first/last 2.5%)
- [ ] Output both smoothed and instant spectra
- [ ] Control frame rate (target FPS)
- [ ] Handle DC offset removal (optional, configurable)

### Frontend (Waterfall & FFT)

- [ ] Setup canvas with device pixel ratio handling
- [ ] Split canvas: FFT (35%) + Waterfall (65%)
- [ ] Implement waterfall scrolling (drawImage)
- [ ] Calculate waterfall noise floor (10th percentile)
- [ ] Implement dynamic range (noise floor ± range)
- [ ] Apply color palette with gamma correction
- [ ] Draw new waterfall row (ImageData)
- [ ] Implement FFT line autoscaling (70 dB range)
- [ ] Smooth noise floor tracking (asymmetrical)
- [ ] Implement trace modes (instant, peak, average)
- [ ] Draw FFT line with proper scaling
- [ ] Add frequency and power axes
- [ ] Handle WebSocket connection and frame parsing

### bladeRF Controls

- [ ] Implement correct configuration order
- [ ] Set sample rate FIRST
- [ ] Set bandwidth
- [ ] Set frequency
- [ ] Configure stream AFTER RF params
- [ ] Set gain AFTER stream is active
- [ ] Use power-of-two read sizes (8192, 16384, etc.)
- [ ] Handle SC16_Q11 format (divide by 2048.0)
- [ ] Configure USB buffers (stay under 16MB)
- [ ] Handle BT200 external LNA (optional)
- [ ] Read back actual hardware values
- [ ] Implement proper stream lifecycle

### WebSocket Protocol

- [ ] Implement binary frame format (or JSON)
- [ ] Send hello message with calibration info
- [ ] Transmit smoothed and instant spectra
- [ ] Include metadata (freq, sample rate, noise floor)
- [ ] Handle frame rate limiting (drop old frames)
- [ ] Parse frames on client side

### Performance Optimization

- [ ] Pre-allocate arrays (reduce allocations)
- [ ] Use NumPy vectorized operations
- [ ] Minimize Python object allocations
- [ ] Use `np.complex64` (not complex128)
- [ ] Downsample large FFTs for display (>4096 points)
- [ ] Use `requestAnimationFrame` for rendering
- [ ] Cache grid/axis drawing
- [ ] Throttle logging

---

## Common Issues and Solutions

### Issue: Stream returns 0 samples

**Causes**:
- Stream not activated
- Wrong read size (not power-of-two)
- Stream deactivated

**Solution**: Verify stream is active and use power-of-two read sizes.

### Issue: Sample rate reverts to 4 MHz

**Cause**: Stream activated before RF parameters set

**Solution**: Configure ALL RF parameters BEFORE setting up stream.

### Issue: Waterfall not scrolling

**Cause**: Canvas coordinate system mismatch

**Solution**: Use device-pixel coordinates for scrolling, CSS coordinates for drawing.

### Issue: FFT not rendering

**Causes**:
- Missing field names (`power_dbfs` vs `power_db`)
- WebSocket not connected
- Invalid data

**Solution**: Check WebSocket connection and frame structure.

### Issue: Noise floor too high

**Causes**:
- Edge bins included in calculation
- Wideband signal contaminating percentile

**Solution**: Exclude edge bins (5%) and use adaptive percentile (2nd for wideband).

---

## References

- **SPEAR-Edge Codebase**: `spear_edge/core/scan/scan_task.py` (FFT processing)
- **SPEAR-Edge Frontend**: `spear_edge/ui/web/app.js` (Waterfall rendering)
- **bladeRF Native Driver**: `spear_edge/core/sdr/bladerf_native.py` (Hardware controls)
- **WebSocket Handler**: `spear_edge/api/ws/live_fft_ws.py` (Protocol)
- **bladeRF Configuration Guide**: `docs/BLADERF_CONFIGURATION_GUIDE.md`

---

## Notes

- This guide is based on SPEAR-Edge implementation (2024)
- All code examples are simplified - see actual implementation for error handling
- Performance optimizations are critical for Jetson Orin Nano
- Test with real hardware (bladeRF 2.0 micro) for validation
- Calibration offset is display-only (backend uses true Q11 scaling)
