# Edge Humps Diagnostic Results

## Summary

**Edge humps are CONFIRMED and SIGNIFICANT:**
- Edge bins are **21-22 dB higher** than center bins
- This explains the visual humps on the FFT display
- Edge bins are affecting noise floor calculation by **3.2 dB**

## Diagnostic Data Analysis

### Edge Bin Elevation (CRITICAL FINDING)

**From Backend Logs:**
```
Edge bins (first 204): mean=-103.4 dBFS, max=-84.7 dBFS
Edge bins (last 204): mean=-103.6 dBFS, max=-85.3 dBFS
Center bins (204): mean=-125.3 dBFS, max=-114.7 dBFS
Edge elevation: start=21.9 dB, end=21.7 dB
```

**Analysis:**
- Edge bins are **21.9 dB and 21.7 dB higher** than center bins
- This is a MASSIVE elevation - explains the visual humps
- Edge bins max at -84.7 to -85.3 dBFS vs center max at -114.7 dBFS
- Edge bins mean at -103.4 to -103.6 dBFS vs center mean at -125.3 dBFS

**Conclusion:** Edge humps are real artifacts, not measurement errors.

### Noise Floor Impact

**From Backend Logs:**
```
Backend (10th pct, averaged): -126.4 dBFS
Frontend equiv (2nd pct, raw): -135.9 dBFS
Without edge bins (10th pct): -129.6 dBFS
Difference (with vs without edges): 3.2 dB
```

**Analysis:**
- Including edge bins raises noise floor by **3.2 dB**
- Edge bins pull the percentile calculation upward
- This affects autoscaling and Y-axis display

**Conclusion:** Edge bins are affecting noise floor calculation, but not dramatically (3.2 dB).

### Noise Floor Comparison

**Measurements:**
- Backend (10th pct, averaged): **-126.4 dBFS**
- Frontend (2nd pct, raw): **-135.9 dBFS**
- TinySA Ultra: **-95 dBFS** (user reported)

**Analysis:**
- Backend and Frontend use different percentiles (10th vs 2nd)
- Frontend's 2nd percentile is more aggressive (lower value)
- TinySA shows -95 dBFS, which is **31-41 dB higher** than our measurements
- This suggests different normalization or measurement methodology

**Possible Explanations:**
1. TinySA might use different normalization (energy vs coherent gain)
2. TinySA might measure at different reference point
3. Our normalization might be too aggressive
4. Different FFT sizes or window functions

**Conclusion:** Need to investigate normalization differences with TinySA.

### DC Offset

**From Backend Logs:**
```
DC Offset: before=0.000000, after=0.000000, removed=0.000000
```

**Analysis:**
- DC offset removal is working perfectly
- No DC bias present in signal
- DC offset is NOT contributing to edge artifacts

**Conclusion:** DC offset removal is not the cause of edge humps.

## Root Cause Analysis

### Why Edge Humps Exist

1. **Window Sidelobes (Most Likely)**
   - Even with Hanning window, sidelobes exist (~31 dB down)
   - Strong signals (ELRS) leak energy into sidelobes
   - Edge bins accumulate leakage from multiple frequency hops
   - **21-22 dB elevation** is consistent with sidelobe leakage

2. **FFT Shift Artifacts**
   - Edge bins after `fftshift` are Nyquist/aliasing bins
   - These bins can show artifacts from anti-aliasing filter rolloff
   - Sample rate limitations affect edge bins differently

3. **Spectral Leakage**
   - FHSS signals (ELRS) hop across spectrum
   - Window sidelobes spread energy to adjacent bins
   - Edge bins accumulate leakage from all hops
   - Multiple hops contribute to edge bin energy

### Why Hanning Window Didn't Fully Fix

- Hanning reduces sidelobes but doesn't eliminate them
- Edge bins are still affected by windowing
- Strong signals (ELRS) still leak into sidelobes
- Need additional filtering or exclusion

## Recommended Fixes

### Fix 1: Exclude Edge Bins from Noise Floor (HIGH PRIORITY)

**Why:**
- Edge bins are 21-22 dB higher than center
- They affect noise floor calculation by 3.2 dB
- Excluding them will give more accurate noise floor

**Implementation:**
- Exclude first/last 5% of bins from percentile calculation
- Use center 90% of spectrum for noise floor
- This matches what we tested: -129.6 dBFS without edges vs -126.4 dBFS with edges

### Fix 2: Zero or Filter Edge Bins for Display (HIGH PRIORITY)

**Why:**
- Edge bins are 21-22 dB higher than center
- They create visual humps that are misleading
- They don't represent real signal energy

**Options:**
- **Option A:** Zero out first/last 2-5% of bins for display
- **Option B:** Apply additional filtering to edge bins (reduce by fixed amount)
- **Option C:** Use different window (Blackman - even lower sidelobes)

**Recommendation:** Option A (zero edge bins) - simplest and most effective

### Fix 3: Use Backend Noise Floor (MEDIUM PRIORITY)

**Why:**
- Backend uses 10th percentile of averaged spectrum (more stable)
- Frontend recalculates using 2nd percentile (more aggressive)
- Backend's value is more accurate

**Implementation:**
- Frontend should use `frame.noise_floor_dbfs` instead of recalculating
- This will give consistent noise floor values
- Matches backend's more stable calculation

### Fix 4: Investigate Normalization (LOW PRIORITY)

**Why:**
- TinySA shows -95 dBFS vs our -126.4 dBFS
- 31 dB difference suggests different normalization
- Need to understand why

**Investigation:**
- Compare TinySA normalization method
- Check if TinySA uses different reference level
- Verify our normalization is correct

## Implementation Priority

1. **Fix JavaScript error** ✅ (Done - dbMax scope issue)
2. **Exclude edge bins from noise floor** (Next)
3. **Zero edge bins for display** (Next)
4. **Use backend noise floor in frontend** (After)
5. **Investigate normalization** (Later)

## Expected Results After Fixes

**After excluding edge bins from noise floor:**
- Noise floor: -129.6 dBFS (vs -126.4 dBFS currently)
- More accurate noise floor estimate
- Better autoscaling

**After zeroing edge bins for display:**
- No visual humps on edges
- Cleaner spectrum display
- Better signal visibility

**After using backend noise floor:**
- Consistent noise floor values
- Better alignment with backend calculations
- More stable display

---

**Status**: Diagnostic Complete - Ready for Fix Implementation
