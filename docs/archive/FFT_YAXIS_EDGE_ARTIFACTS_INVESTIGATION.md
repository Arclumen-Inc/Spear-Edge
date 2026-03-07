# FFT Y-Axis and Edge Artifacts Investigation

## Issue 1: Y-Axis dBFS Numbers Seem Incorrect

### Current Implementation Analysis

**Y-Axis Rendering (`drawPowerAxis`):**
```javascript
const dbMin = fftDbMinSmoothed;  // Autoscaled minimum
const dbMax = dbMin + FFT_VIEW_RANGE_DB;  // Fixed 55 dB span
// Labels show: dbMin, dbMin+13.75, dbMin+27.5, dbMin+41.25, dbMax
```

**Autoscaling Logic:**
```javascript
// Noise floor estimate (2nd percentile)
const floorNow = sortedF[Math.floor(sortedF.length * 0.02)];

// Smoothed floor tracking
fftFloorSmoothed = smoothed version of floorNow;

// Display range bottom
const dbMinTarget = fftFloorSmoothed - FFT_FLOOR_MARGIN_DB;  // Currently 0 dB margin
const dbMin = fftDbMinSmoothed;  // Clamped movement version
const dbMax = dbMin + 55;  // Fixed 55 dB span
```

### Problem Identified

**The Y-axis shows DISPLAY RANGE, not ABSOLUTE dBFS values.**

- The axis labels are relative to the autoscaled noise floor
- If noise floor is at -100 dBFS, axis shows -100 to -45 dBFS
- If noise floor is at -80 dBFS, axis shows -80 to -25 dBFS
- The labels don't represent actual signal power levels

**Example from screenshot:**
- Y-axis shows: -94, -108, -121 dBFS
- These are the DISPLAY RANGE (autoscaled), not the actual signal levels
- The actual ELRS signal might be at -50 dBFS, but it's displayed relative to noise floor

### Root Cause

The autoscaling is designed to keep the noise floor at the bottom of the display, which is good for visualization but makes the Y-axis labels misleading. The labels should either:
1. Show absolute dBFS values (fixed reference)
2. Be labeled as "dB above noise" or "relative dB"
3. Show both absolute and relative values

### Expected Behavior (SDR++/GQRX)

Professional SDR software typically:
- Shows **absolute dBFS values** on Y-axis (fixed reference level)
- Uses **reference level** control to adjust the top of the display
- Shows **span** control to adjust the vertical range
- The noise floor moves within the fixed scale, not the scale moving with the floor

---

## Issue 2: Edge Humps on FFT Spectrum

### Current FFT Processing Pipeline

```python
# 1. DC offset removal (BEFORE windowing)
iq = iq - np.mean(iq)

# 2. Windowing
x = iq * self._win  # Nuttall window

# 3. FFT
X = np.fft.fftshift(np.fft.fft(x, n=self.fft_size))

# 4. Normalization
mag = np.abs(X) / self._window_sum  # Divide by sum(window), not fft_size
spec_db = 20.0 * np.log10(mag + self._eps)
```

### Potential Causes of Edge Humps

#### 1. **Window Edge Effects (Most Likely)**

**Nuttall Window Characteristics:**
- High sidelobes (~93 dB down from main lobe)
- Edge bins (first/last 5-10% of spectrum) can show elevated levels
- This is a known artifact of windowed FFTs

**Evidence:**
- Humps appear at the edges (left and right sides)
- Consistent with window sidelobe leakage
- More pronounced with strong signals (ELRS transmitter)

**Why it happens:**
- Window function has non-zero values at edges
- Strong signals leak energy into sidelobes
- Edge bins accumulate leakage from multiple sources
- Nuttall window has worse sidelobes than Hanning/Hamming

#### 2. **DC Offset Removal Timing**

**Current:** DC removed BEFORE windowing
```python
iq = iq - np.mean(iq)  # Remove DC
x = iq * self._win      # Apply window
```

**Issue:** 
- DC offset removal changes the signal statistics
- Window is then applied to DC-corrected signal
- This can cause edge artifacts if DC removal isn't perfect

**Better approach:**
- Remove DC AFTER windowing (or use windowed DC removal)
- Or use a DC-blocking filter instead of mean subtraction

#### 3. **Normalization Formula**

**Current:** `mag = |X| / sum(window)`

**Issue:**
- Dividing by `window_sum` (coherent gain) instead of `fft_size`
- This normalization is correct for SDR++ compatibility
- But edge bins might be affected differently than center bins

**Alternative normalization:**
- `mag = |X| / fft_size` (standard FFT normalization)
- `mag = |X| / sqrt(sum(window^2))` (energy normalization)

#### 4. **FFT Shift Artifacts**

**Current:** `X = np.fft.fftshift(np.fft.fft(x, n=self.fft_size))`

**Issue:**
- `fftshift` moves DC to center
- Edge bins after shift are actually Nyquist/aliasing bins
- These bins can show artifacts from:
  - Anti-aliasing filter rolloff
  - Sample rate limitations
  - Strong signals near Nyquist frequency

#### 5. **Spectral Leakage from Strong Signals**

**ELRS Signal Characteristics:**
- FHSS (Frequency Hopping Spread Spectrum)
- Strong, narrowband signals hopping across spectrum
- Window sidelobes spread energy to adjacent bins
- Edge bins accumulate leakage from all hops

**Why edges are worse:**
- Edge bins are furthest from signal center
- Window sidelobes are strongest at edges
- Multiple hops contribute to edge bin energy

### Diagnostic Steps Needed

1. **Check actual signal levels:**
   - Log peak power in dBFS
   - Compare to Y-axis labels
   - Verify normalization is correct

2. **Check edge bin values:**
   - Log first/last 10 bins of spectrum
   - Compare to center bins
   - Check if humps are consistent or variable

3. **Test with different windows:**
   - Try Hanning window (lower sidelobes)
   - Compare edge artifacts
   - See if Nuttall is the culprit

4. **Check DC offset:**
   - Log mean IQ before/after DC removal
   - Verify DC removal is working
   - Check if edge artifacts correlate with DC

5. **Check normalization:**
   - Verify `window_sum` value
   - Compare to theoretical Nuttall window sum
   - Check if normalization is consistent across bins

---

## Recommended Fixes

### Fix 1: Y-Axis Labels

**Option A: Show Absolute dBFS (Recommended)**
- Use fixed reference level (e.g., -20 dBFS at top)
- Show absolute values: -20, -40, -60, -80, -100 dBFS
- Noise floor moves within fixed scale
- Matches SDR++ behavior

**Option B: Show Relative dB**
- Label Y-axis as "dB above noise"
- Keep current autoscaling
- Make it clear values are relative

**Option C: Dual Labels**
- Show absolute dBFS on left
- Show relative dB on right
- Best of both worlds

### Fix 2: Edge Humps

**Option A: Zero Edge Bins (Quick Fix)**
- Zero out first/last 2-5% of bins
- Hides artifacts but loses data
- Not ideal for signal analysis

**Option B: Better Window (Recommended)**
- Switch to Hanning or Blackman window
- Lower sidelobes = less edge leakage
- Trade-off: slightly wider main lobe

**Option C: Edge Filtering**
- Apply additional filtering to edge bins
- Reduce edge bin values by fixed amount
- Preserves data but reduces artifacts

**Option D: Fix DC Removal**
- Remove DC AFTER windowing
- Or use proper DC-blocking filter
- May reduce edge artifacts

**Option E: Normalization Fix**
- Verify normalization is correct
- Check if edge bins need different normalization
- May require per-bin normalization

---

## Next Steps

1. **Add diagnostic logging** to capture:
   - Actual peak power in dBFS
   - Edge bin values (first/last 10 bins)
   - Noise floor estimate
   - Window sum value
   - DC offset before/after removal

2. **Test with different windows:**
   - Compare Nuttall vs Hanning vs Blackman
   - Measure edge artifact levels
   - Choose best window for ELRS signals

3. **Fix Y-axis display:**
   - Implement fixed reference level
   - Show absolute dBFS values
   - Add reference level control

4. **Fix edge artifacts:**
   - Implement chosen solution (window change or filtering)
   - Verify artifacts are reduced
   - Test with ELRS signals

---

## Code Locations

**Y-Axis Rendering:**
- `spear_edge/ui/web/app.js`: `drawPowerAxis()` (line ~591)
- `spear_edge/ui/web/app.js`: Autoscaling logic (line ~913-948)

**FFT Processing:**
- `spear_edge/core/scan/scan_task.py`: FFT calculation (line ~167-185)
- `spear_edge/core/scan/scan_task.py`: Window initialization (line ~45-50)

**IQ Scaling:**
- `spear_edge/core/sdr/bladerf_native.py`: Sample conversion (line ~869-933)
- `spear_edge/settings.py`: IQ_SCALING_MODE setting

---

## Expected Values for ELRS Signal

**Typical ELRS Signal Levels:**
- Peak power: -40 to -60 dBFS (depending on gain)
- Noise floor: -80 to -100 dBFS (depending on FFT size)
- Signal-to-noise: 20-40 dB
- Bandwidth: ~500 kHz per hop

**If Y-axis shows -94 to -121 dBFS:**
- This suggests noise floor is around -100 dBFS
- ELRS signal should be 20-40 dB above floor
- Expected peak: -60 to -80 dBFS
- But display shows relative to floor, so peak appears at top of display

---

**Document Status**: Investigation Complete - Ready for Fix Implementation
