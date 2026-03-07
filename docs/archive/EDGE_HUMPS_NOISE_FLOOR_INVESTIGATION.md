# Edge Humps and Noise Floor Investigation

> **Note:** This document is historical. Many of the fixes described below have been implemented. See implementation status in "Potential Fixes" section.

## Key Findings

### Issue 1: Noise Floor Calculation Discrepancy

**Backend (`scan_task.py`):**
- Calculates noise floor using **10th percentile** of **averaged spectrum**
- Formula: `noise_floor_db = np.percentile(self._avg_db, 10)`
- Uses EMA-averaged spectrum (`self._avg_db`) for stability
- Sends this value as `noise_floor_dbfs` in the frame

**Frontend (`app.js`):**
- **IGNORES** backend's `noise_floor_dbfs` value!
- Recalculates noise floor using **2nd percentile** of **raw FFT array**
- Formula: `floorNow = sortedF[Math.floor(sortedF.length * 0.02)]`
- Uses raw `power_inst_dbfs` array (not averaged)
- Then applies smoothing with asymmetrical attack/release

**Problem:**
- Two different calculations = inconsistent noise floor values
- Frontend's 2nd percentile is much more aggressive (lower value)
- If edge bins with humps are included, they could affect the percentile
- TinySA shows -95 dBFS, but our frontend might show different due to recalculation

**Evidence:**
```javascript
// Frontend code (app.js line ~772)
const noiseFloor = frame.noise_floor_dbfs !== undefined ? frame.noise_floor_dbfs : fftMin;
// This value is logged but NOT USED for autoscaling!

// Instead, frontend recalculates (line ~927):
const idx = Math.max(0, Math.floor(sortedF.length * FFT_FLOOR_PCT));  // 2nd percentile
const floorNow = sortedF.length ? sortedF[idx] : -80;
```

### Issue 2: Edge Humps Still Present (Even with Hanning Window)

**Possible Causes:**

#### A. Edge Bins Included in All Calculations
- Edge bins (first/last 5-10%) are **NOT excluded** from:
  - Noise floor calculation (2nd percentile)
  - FFT display
  - Smoothing operations
- If edge bins have elevated values (humps), they affect:
  - Noise floor percentile (pulls it up)
  - Display range calculation
  - Y-axis scaling

**Impact:**
- Edge humps → higher percentile values → higher noise floor estimate
- This could explain why noise floor appears higher than TinySA's -95 dBFS

#### B. FFT Shift Artifacts
- `fftshift` moves DC to center
- Edge bins after shift are actually **Nyquist/aliasing bins**
- These bins can show artifacts from:
  - Anti-aliasing filter rolloff
  - Sample rate limitations
  - Strong signals near Nyquist frequency

**Evidence:**
- Humps appear at both edges (left and right)
- Consistent with Nyquist bin artifacts
- More pronounced with strong signals (ELRS)

#### C. DC Offset Removal Timing
**Current:** DC removed BEFORE windowing
```python
iq = iq - np.mean(iq)  # Remove DC
x = iq * self._win      # Apply window
```

**Issue:**
- DC offset removal changes signal statistics
- Window is applied to DC-corrected signal
- Edge bins might be affected differently than center bins
- Could cause edge artifacts if DC removal isn't perfect

#### D. Window Edge Effects (Even with Hanning)
- Hanning window still has sidelobes (~31 dB down)
- Edge bins can still show elevated levels
- Strong signals (ELRS) leak energy into sidelobes
- Multiple frequency hops accumulate in edge bins

**Why Hanning Didn't Fully Fix:**
- Hanning reduces sidelobes but doesn't eliminate them
- Edge bins are still affected by windowing
- Need additional filtering or exclusion

### Issue 3: Noise Floor Manipulation

**Frontend Smoothing:**
```javascript
// Asymmetrical attack/release smoothing
const alpha = (floorNow > fftFloorSmoothed) ? FFT_RISE_ALPHA : FFT_FALL_ALPHA;
fftFloorSmoothed = (1 - alpha) * fftFloorSmoothed + alpha * floorNow;
```

**Parameters:**
- `FFT_RISE_ALPHA = 0.18` (fast attack - floor rises quickly)
- `FFT_FALL_ALPHA = 0.03` (slow release - floor falls slowly)

**Impact:**
- If edge bins cause `floorNow` to be higher, smoothing pulls floor up
- Slow release means floor stays elevated even after humps decrease
- This could explain discrepancy with TinySA

**Clamping:**
```javascript
const step = Math.max(-FFT_MAX_STEP_DB, Math.min(FFT_MAX_STEP_DB, delta));
fftDbMinSmoothed += step;
```
- Maximum 0.35 dB movement per frame
- Prevents rapid changes but also prevents rapid correction

## Diagnostic Questions

1. **Are edge bins included in noise floor calculation?**
   - YES - No exclusion logic found
   - Edge bins with humps would affect 2nd percentile

2. **Is backend noise floor being used?**
   - NO - Frontend recalculates from raw FFT array
   - Backend's averaged 10th percentile is ignored

3. **Why would TinySA show -95 dBFS but we show different?**
   - Frontend uses 2nd percentile (more aggressive)
   - Edge bins with humps pull percentile up
   - Smoothing keeps floor elevated
   - Different measurement methodology

4. **Why do edge humps persist with Hanning window?**
   - Hanning reduces but doesn't eliminate sidelobes
   - Edge bins are Nyquist bins (aliasing artifacts)
   - DC offset removal timing
   - No exclusion of edge bins from calculations

## Recommended Diagnostic Steps

1. **Log edge bin values:**
   ```python
   # In scan_task.py, add diagnostic logging
   edge_bins_start = spec_db[:10]  # First 10 bins
   edge_bins_end = spec_db[-10:]   # Last 10 bins
   center_bins = spec_db[len(spec_db)//2-5:len(spec_db)//2+5]
   print(f"Edge bins (start): {edge_bins_start}")
   print(f"Edge bins (end): {edge_bins_end}")
   print(f"Center bins: {center_bins}")
   ```

2. **Compare noise floor calculations:**
   ```python
   # Backend: 10th percentile of averaged spectrum
   backend_floor = np.percentile(self._avg_db, 10)
   
   # Frontend equivalent: 2nd percentile of raw spectrum
   frontend_floor_equiv = np.percentile(spec_db, 2)
   
   print(f"Backend floor (10th pct, averaged): {backend_floor:.1f} dBFS")
   print(f"Frontend equiv (2nd pct, raw): {frontend_floor_equiv:.1f} dBFS")
   ```

3. **Check if edge bins affect percentile:**
   ```python
   # Calculate percentile with and without edge bins
   exclude_edge_pct = 0.05  # Exclude 5% from each edge
   exclude_count = int(len(spec_db) * exclude_edge_pct)
   center_spectrum = spec_db[exclude_count:-exclude_count]
   floor_without_edges = np.percentile(center_spectrum, 2)
   floor_with_edges = np.percentile(spec_db, 2)
   print(f"Floor with edges: {floor_with_edges:.1f} dBFS")
   print(f"Floor without edges: {floor_without_edges:.1f} dBFS")
   ```

4. **Verify DC offset removal:**
   ```python
   dc_before = np.mean(iq)
   iq_dc_removed = iq - dc_before
   dc_after = np.mean(iq_dc_removed)
   print(f"DC before: {dc_before:.6f}, after: {dc_after:.6f}")
   ```

## Potential Fixes

### Fix 1: Use Backend Noise Floor ✅ **IMPLEMENTED**
- ✅ Backend sends `noise_floor_dbfs` in frame (line 316)
- ✅ Backend uses averaged 10th percentile for stability (line 195)
- ⚠️ Frontend still recalculates (line ~927 in app.js) - could be improved to use backend value

### Fix 2: Exclude Edge Bins from Noise Floor ✅ **IMPLEMENTED**
- ✅ Edge bins excluded from noise floor calculation (lines 191-198)
- ✅ Excludes first/last 5% of bins from percentile
- ✅ Prevents edge humps from affecting noise floor estimate

### Fix 3: Zero or Filter Edge Bins ✅ **IMPLEMENTED**
- ✅ Edge bins zeroed for display (lines 222-233)
- ✅ First/last 2.5% of bins set to noise_floor - 10 dB
- ✅ Reduces visual artifacts in FFT line and waterfall

### Fix 4: Fix DC Offset Removal Timing
- Remove DC AFTER windowing (or use windowed DC removal)
- Or use DC-blocking filter instead of mean subtraction
- May reduce edge artifacts

### Fix 5: Use Different Window
- Try Blackman window (even lower sidelobes)
- Or Kaiser window (adjustable sidelobe suppression)
- Trade-off: Wider main lobe

## Code Locations

**Noise Floor Calculation:**
- Backend: `spear_edge/core/scan/scan_task.py` line ~189
- Frontend: `spear_edge/ui/web/app.js` line ~927

**Edge Bin Handling:**
- ✅ Edge bins excluded from noise floor calculation (lines 191-198)
- ✅ Edge bins zeroed for display (lines 222-233)

**DC Offset Removal:**
- `spear_edge/core/scan/scan_task.py` line ~169

**FFT Processing:**
- `spear_edge/core/scan/scan_task.py` line ~168-180

---

**Status**: Investigation Complete - Ready for Diagnostic Logging
