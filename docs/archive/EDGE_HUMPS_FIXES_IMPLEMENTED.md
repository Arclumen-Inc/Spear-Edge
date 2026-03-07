# Edge Humps Fixes - Implementation Summary

## Fixes Implemented ✅

### Fix 1: Exclude Edge Bins from Noise Floor Calculation

**File:** `spear_edge/core/scan/scan_task.py`

**Changes:**
- Modified noise floor calculation to exclude first/last 5% of bins
- Edge bins are 20+ dB higher and don't represent true noise floor
- Uses center 90% of spectrum for noise floor estimation

**Code:**
```python
# EXCLUDE edge bins (first/last 5%) from noise floor calculation
exclude_pct = 0.05  # Exclude 5% from each edge
exclude_count = int(len(self._avg_db) * exclude_pct)
if exclude_count > 0 and exclude_count < len(self._avg_db) // 2:
    center_spectrum = self._avg_db[exclude_count:-exclude_count]
    noise_floor_db = float(np.percentile(center_spectrum, 10))
```

**Expected Impact:**
- Noise floor: -129.6 dBFS (vs -126.4 dBFS with edges)
- More accurate noise floor estimate
- Better autoscaling

### Fix 2: Zero Edge Bins for Display

**File:** `spear_edge/core/scan/scan_task.py`

**Changes:**
- Zero out first/last 2.5% of bins (5% total) in both FFT line and waterfall
- Edge bins set to noise floor - 10 dB (effectively hidden below display)
- Removes visual humps from display

**Code:**
```python
# ZERO edge bins (first/last 2.5%) for display to remove visual humps
edge_zero_pct = 0.025  # Zero 2.5% from each edge (5% total)
edge_zero_count = int(len(power_line) * edge_zero_pct)
if edge_zero_count > 0:
    edge_zero_value = noise_floor_db - 10.0
    power_line[:edge_zero_count] = edge_zero_value
    power_line[-edge_zero_count:] = edge_zero_value
    power_inst[:edge_zero_count] = edge_zero_value
    power_inst[-edge_zero_count:] = edge_zero_value
```

**Expected Impact:**
- No visual humps on edges
- Cleaner spectrum display
- Better signal visibility
- Edge bins hidden below display range

### Fix 3: Use Backend Noise Floor in Frontend

**File:** `spear_edge/ui/web/app.js`

**Changes:**
- Frontend now uses `frame.noise_floor_dbfs` from backend
- Backend uses 10th percentile of averaged spectrum (more stable)
- Fallback to frontend calculation if backend value not available
- Fallback also excludes edge bins for consistency

**Code:**
```javascript
// USE BACKEND NOISE FLOOR (more stable and accurate)
const backendFloor = frame.noise_floor_dbfs !== undefined ? frame.noise_floor_dbfs : null;

if (backendFloor !== null && Number.isFinite(backendFloor)) {
    // Use backend noise floor directly
    floorNow = backendFloor;
} else {
    // Fallback: calculate from frontend (excluding edge bins)
    const excludeCount = Math.floor(fftArr.length * 0.05);
    const centerSpectrum = fftArr.slice(excludeCount, -excludeCount)
        .map(Number)
        .filter(Number.isFinite)
        .sort((a, b) => a - b);
    const idx = Math.max(0, Math.floor(centerSpectrum.length * FFT_FLOOR_PCT));
    floorNow = centerSpectrum.length ? centerSpectrum[idx] : -80;
}
```

**Expected Impact:**
- Consistent noise floor values (backend and frontend match)
- More stable display (backend uses averaged spectrum)
- Better alignment with backend calculations

## Expected Results

### Before Fixes:
- Edge bins: 21-22 dB higher than center
- Visual humps on edges
- Noise floor affected by edge bins (+3.2 dB)
- Frontend recalculates noise floor (inconsistent)

### After Fixes:
- Edge bins: Zeroed/hidden below display
- No visual humps
- Noise floor: More accurate (-129.6 dBFS vs -126.4 dBFS)
- Consistent noise floor (backend and frontend match)

## Testing Recommendations

1. **Visual Check:**
   - Verify edge humps are gone
   - Check that spectrum edges are clean
   - Ensure signals are still visible

2. **Noise Floor:**
   - Check backend logs: should show -129.6 dBFS (without edges)
   - Check frontend console: should match backend value
   - Verify autoscaling works correctly

3. **Signal Visibility:**
   - Verify ELRS signals are still visible
   - Check that signal peaks are not affected
   - Ensure waterfall display is clean

## Files Modified

1. `spear_edge/core/scan/scan_task.py`
   - Exclude edge bins from noise floor calculation
   - Zero edge bins for display

2. `spear_edge/ui/web/app.js`
   - Use backend noise floor instead of recalculating
   - Fallback excludes edge bins for consistency

---

**Status**: Implementation Complete - Ready for Testing
