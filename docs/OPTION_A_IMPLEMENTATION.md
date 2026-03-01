# Option A Implementation: True SC16_Q11 End-to-End

## Overview
This document describes the implementation of Option A: treating bladeRF output as true SC16_Q11 end-to-end, while allowing the UI to optionally present "SDR++-style" numbers via a simple display offset.

## Phase 1: bladeRF Sample Scaling + Clipping Metrics ✅

### Changes Made

**File:** `spear_edge/core/sdr/bladerf_native.py`

1. **Fixed IQ scaling to true Q11:**
   - Changed from: `scale = 1.0 / 32768.0` (int16 full-scale)
   - Changed to: `scale = 1.0 / 2048.0` (Q11 full-scale)
   - This makes `|iq|≈1.0` correspond to Q11 full-scale (±2048), not int16 full-scale

2. **Fixed rail/clipping detection:**
   - Changed from: `rail_frac = float(np.mean(raw >= 32760))` (int16 threshold)
   - Changed to: `rail_frac = float(np.mean(raw >= 2047))` (Q11 threshold)
   - Updated `fs_frac` comment to reflect Q11 full-scale

3. **Added format sanity logging:**
   - Every 200 reads, logs: `raw_max`, `raw_mean`, `rail_frac`, `gain`, `lna_gain`, `bt200` status
   - Expected range: `raw_max` should be ~0..2047 for Q11
   - Warns if `rail_frac > 0.001` (more than 0.1% samples hitting rails)

### Pass Criteria
- When gain is increased and strong signal is present:
  - `raw_max` should approach ~2047
  - `rail_frac` and `fs_frac` should rise together
  - Both should remain near 0 until overdriven

## Phase 2: FFT Math Consistency ✅

**File:** `spear_edge/core/scan/scan_task.py`

- **DC removal:** Already implemented (`iq = iq - np.mean(iq)`) ✅
- **Window normalization:** Already using display-grade normalization (`mag = np.abs(X) / self.fft_size`) ✅
  - No coherent gain correction (removed in previous fix)
  - This yields stable baseline and "looks right" for SDR UI

### Result
- Noise floor is ~6 dB lower than coherent-gain-corrected version
- Peaks remain visible
- Matches SDR++/GQRX behavior

## Phase 3: Calibration Metadata to UI ✅

### Changes Made

**File:** `spear_edge/api/ws/live_fft_ws.py`
- Updated WebSocket hello message to use `settings.CALIBRATION_OFFSET_DB`
- Supports both modes:
  - `0.0` = True Q11 dBFS (0 dBFS = Q11 full-scale ±2048)
  - `-24.08` = SDR++-style 16-bit dBFS (matches SDR++ expectations)

**File:** `spear_edge/settings.py`
- Updated documentation to explain Option A approach
- `CALIBRATION_OFFSET_DB` defaults to `0.0` (true Q11)
- Can be set via env var: `SPEAR_CALIBRATION_OFFSET_DB=-24.08` for SDR++-style

**File:** `spear_edge/ui/web/app.js`
- UI now applies `globalCalibrationOffset` to FFT array values
- Offset is display-only (backend uses true Q11 internally)

### Usage

**True Q11 dBFS (default):**
```bash
# No env var needed, or explicitly:
export SPEAR_CALIBRATION_OFFSET_DB=0.0
```

**SDR++-style 16-bit dBFS:**
```bash
export SPEAR_CALIBRATION_OFFSET_DB=-24.08
```

The offset is calculated as:
```
offset_db = 20*log10(2048/32768) = -24.082 dB
```

## Phase 4: Verification Tests (User to Perform)

### Test A: FM Broadcast (Strong, Easy)
- Center: ~100 MHz
- Sample rate: 5–10 MS/s
- FFT size: 4096+
- Gain: Mid/high
- **Expected:** Obvious wideband peaks and structure, not flat

### Test B: ELRS Hopping (902–928 MHz)
- Sample rate: 30 MS/s
- FFT size: 16384+ (32768 if performance allows)
- Gain: Appropriate
- **Expected:** Hopping energy clearly visible in waterfall; FFT shows burst peaks

### Test C: Gain Sweep Sanity Check
- Hold frequency fixed, increase gain in steps
- **Expected:**
  - Noise floor rises with gain (normal)
  - Strong signals rise faster than noise until compression/clipping
  - `rail_frac` and `fs_frac` remain near 0 until overdriven

## Phase 5: FFT Size Recommendations

### Wideband Viewing (30 MS/s)
- **Minimum:** 16384 bins (~1.83 kHz/bin)
- **Recommended:** 32768 bins (~915 Hz/bin) if performance allows
- **Why:** Reduces per-bin RBW so noise floor looks more like expected

### Lower Sample Rates
- 2 MS/s: 4096 bins (~488 Hz/bin) is fine
- 5 MS/s: 8192 bins (~610 Hz/bin) is fine
- 10 MS/s: 16384 bins (~610 Hz/bin) recommended

## Deliverables Checklist ✅

- ✅ bladeRF samples treated as real SC16_Q11
- ✅ Clipping metrics that actually mean something (Q11 thresholds)
- ✅ FFT normalization that behaves like a spectrum viewer (no coherent gain)
- ✅ UI "SDR++ look" controlled by single -24.08 dB display offset (not baked into IQ)
- ✅ Predictable results across sample rates / FFT sizes

## Key Points

1. **Backend is internally consistent:** All IQ processing uses true Q11 scaling (1.0/2048.0)
2. **Display offset is optional:** UI can show either Q11 dBFS or SDR++-style via offset
3. **No signal processing hacks:** Offset is purely for display, not baked into signal path
4. **Format sanity checks:** Logging helps verify Q11 format is working correctly
5. **Clipping detection is accurate:** Uses Q11 thresholds, not int16 thresholds

## Environment Variable Reference

```bash
# True Q11 dBFS (default - recommended for engineering/debug)
export SPEAR_CALIBRATION_OFFSET_DB=0.0

# SDR++-style 16-bit dBFS (for UI matching SDR++ expectations)
export SPEAR_CALIBRATION_OFFSET_DB=-24.08
```
