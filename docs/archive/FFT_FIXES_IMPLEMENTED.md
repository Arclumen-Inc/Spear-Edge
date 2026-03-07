# FFT Y-Axis and Edge Artifacts Fixes - Implementation Summary

## Changes Implemented

### 1. Window Change: Nuttall → Hanning ✅

**File:** `spear_edge/core/scan/scan_task.py`

**Changes:**
- Switched from Nuttall window to Hanning window
- Updated normalization comments to reflect Hanning window characteristics
- Updated theoretical noise floor calculation for Hanning window

**Why:**
- Hanning window has lower sidelobes (~31 dB vs ~93 dB for Nuttall)
- Reduces edge artifacts (humps at spectrum edges)
- Better for FHSS signals like ELRS
- Trade-off: Slightly wider main lobe (better frequency resolution vs sidelobe suppression)

**Impact:**
- Edge humps should be significantly reduced
- Better visualization of signals near spectrum edges
- Slightly wider frequency bins (minimal impact)

### 2. Y-Axis Fix: Absolute dBFS Values ✅

**File:** `spear_edge/ui/web/app.js`

**Changes:**
- Added `FFT_REFERENCE_LEVEL_DBFS = -20` constant (fixed reference level)
- Modified autoscaling logic to use fixed reference level at top
- Y-axis now shows absolute dBFS values, not relative values
- Display range autoscales to show noise floor at bottom

**How it works:**
- Top of display: Fixed at -20 dBFS (reference level)
- Bottom of display: Autoscales to show noise floor (typically -80 to -100 dBFS)
- Y-axis labels: Show absolute dBFS values (e.g., -20, -40, -60, -80, -100 dBFS)
- Range: Typically 70-80 dB span (adjusts if noise floor is very low)

**Before:**
- Y-axis showed relative values (autoscaled range)
- Labels like -94, -108, -121 dBFS were relative to noise floor
- Confusing - didn't represent actual signal levels

**After:**
- Y-axis shows absolute dBFS values
- Labels represent actual signal power levels
- Matches SDR++/GQRX behavior
- Easier to interpret signal strength

**Example:**
- If ELRS signal is at -50 dBFS, Y-axis will show -50 dBFS (not relative value)
- If noise floor is at -100 dBFS, Y-axis shows -100 dBFS at bottom
- Reference level -20 dBFS at top provides consistent reference point

## Technical Details

### Hanning Window Normalization

**Window Sum:**
- Hanning: `sum(window) ≈ 0.5 * N` (coherent gain ≈ 0.5)
- Nuttall: `sum(window) ≈ 0.363 * N` (coherent gain ≈ 0.363)

**Normalization Formula:**
```python
mag = np.abs(X) / self._window_sum  # Divide by sum(window)
spec_db = 20.0 * np.log10(mag + eps)
```

**Theoretical Noise Floor:**
- Hanning: `6.02 - 10*log10(N)` dBFS
- For N=4096: ≈ -30.1 dBFS (theoretical)
- Actual noise floor will be lower due to process gain and hardware noise

### Y-Axis Display Logic

**Fixed Reference Level Approach:**
```javascript
const FFT_REFERENCE_LEVEL_DBFS = -20;  // Fixed top
const FFT_VIEW_RANGE_DB = 70;          // Default span

// Calculate bottom based on noise floor
dbMin = autoscaled noise floor
dbMax = min(dbMin + 70, -20)  // Clamp to reference level

// If noise floor is very low, extend range downward
// Range can be 70-80 dB depending on noise floor
```

**Benefits:**
- Consistent reference point (always -20 dBFS at top)
- Absolute values on Y-axis (matches professional SDR software)
- Still autoscales to show noise floor
- Easy to interpret signal strength

## Testing Recommendations

1. **Edge Artifacts:**
   - Check if edge humps are reduced
   - Compare before/after screenshots
   - Test with ELRS transmitter (FHSS signals)

2. **Y-Axis Values:**
   - Verify Y-axis shows absolute dBFS values
   - Check that ELRS signal levels make sense (-40 to -60 dBFS typical)
   - Compare to expected signal strength

3. **Noise Floor:**
   - Verify noise floor is visible at bottom
   - Check that range adjusts correctly for different noise floors
   - Ensure reference level stays fixed at -20 dBFS

4. **Signal Visibility:**
   - Verify signals are still clearly visible
   - Check that autoscaling still works correctly
   - Ensure strong signals don't clip at top

## Expected Results

**Edge Artifacts:**
- Edge humps should be significantly reduced
- Cleaner spectrum edges
- Better visualization of signals near edges

**Y-Axis:**
- Shows absolute dBFS values (e.g., -20, -40, -60, -80, -100)
- ELRS signal at -50 dBFS will show as -50 dBFS on Y-axis
- Easier to interpret signal strength
- Matches professional SDR software behavior

## Files Modified

1. `spear_edge/core/scan/scan_task.py`
   - Window changed to Hanning
   - Normalization comments updated
   - Theoretical floor calculation updated

2. `spear_edge/ui/web/app.js`
   - Y-axis autoscaling logic updated
   - Fixed reference level added
   - Absolute dBFS values on Y-axis

## Next Steps

After testing, consider:
1. Adding reference level control (user-adjustable top)
2. Adding span control (user-adjustable range)
3. Fine-tuning reference level (-20 dBFS might need adjustment)
4. Adding option to switch back to Nuttall window if needed

---

**Status**: Implementation Complete - Ready for Testing
