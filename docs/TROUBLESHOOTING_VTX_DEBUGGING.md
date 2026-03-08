# VTX Signal Debugging: Complete Investigation Report

## Overview

This document details the comprehensive debugging process that led to the discovery and resolution of multiple critical bugs preventing VTX (FPV video transmitter) signals from being visible in the SPEAR-Edge FFT/waterfall display.

**Date**: 2024
**Issue**: VTX signals (5.917 GHz, 20-30 MHz wide) were completely invisible in the UI despite being clearly visible on a TinySA Ultra spectrum analyzer and in video goggles.

## Root Cause: Frequency Tuning Bug

### The Critical Bug

**Problem**: The SDR was not tuning to the correct frequency for signals above 4.29 GHz.

**Root Cause**: 
- `bladerf_set_frequency()` function signature in `bladerf_native.py` used `ctypes.c_uint32` instead of `ctypes.c_uint64`
- Frequencies above 4.29 GHz (the maximum value for a 32-bit unsigned integer) were truncated
- Example: 5910 MHz → 1615 MHz (completely wrong frequency)

**Impact**: 
- VTX signals and other high-frequency signals were completely invisible
- The SDR was tuned to the wrong frequency, so no signal was present at the expected location

**Fix**: Changed function signature to `ctypes.c_uint64` to support full frequency range (up to 6 GHz for bladeRF)

**Verification**: 
- Created `scripts/verify_frequency.py` to test frequency tuning accuracy
- Verified that frequency readback now matches requested value

### Why This Was Hard to Detect

1. **No Error Messages**: The function call succeeded, but with a truncated value
2. **Silent Failure**: The SDR appeared to be working (no errors), but was tuned incorrectly
3. **Frequency Readback Bug**: Some libbladeRF versions return incorrect readback values, masking the issue
4. **Wideband Signal Characteristics**: Wideband signals appear as plateaus, not sharp peaks, making it harder to notice they're missing

## Secondary Issues Discovered

While debugging the frequency issue, several other real bugs and improvements were identified:

### 1. USB Buffer Memory Calculation Error

**Problem**: Buffer size calculations incorrectly assumed 2 bytes per complex sample.

**Reality**: SC16_Q11 format uses 4 bytes per complex sample (2 bytes I + 2 bytes Q).

**Impact**: 
- USB buffer memory usage was 2x larger than calculated
- Caused `LIBUSB_ERROR_NO_MEM` errors at high sample rates
- Buffer configurations were too aggressive

**Fix**: Corrected all buffer size calculations to use 4 bytes/complex sample and reduced buffer counts to fit within 16MB USB memory limit.

### 2. Stale Tail Data in Preallocated Buffers

**Problem**: Preallocated conversion buffers contained stale data from previous reads.

**Impact**: 
- Only prefix of buffer was written, but downstream code could access full buffer
- Wideband signals appeared smeared or inconsistent
- Narrowband signals were less affected (smaller read sizes)

**Fix**: Explicitly zero unused tail of buffers before returning.

### 3. Race Condition in Gain Adjustments

**Problem**: Rapid gain slider movements caused `malloc_consolidate()` segfaults.

**Root Cause**: 
- Concurrent calls to `set_gain()` without proper thread synchronization
- UI slider was not debounced, causing many API calls per second
- Device pointer checked before acquiring lock, but not re-checked inside lock

**Fix**: 
- Added `threading.Lock()` for gain operations
- Implemented 200ms debouncing on gain slider
- Re-check device pointer inside lock before making libbladeRF calls

### 4. Device Close/Reopen Race Condition

**Problem**: Segmentation fault when switching sample rates.

**Root Cause**: RX task thread called `read_samples()` while device was being closed/reopened.

**Fix**: 
- Set `_stream_active = False` at start of reconfiguration
- Added 50ms delay after stream deactivation
- Reordered checks in `read_samples()` to prioritize `_stream_active` flag

### 5. Noise Floor Calculation for Wideband Signals

**Problem**: Noise floor calculation was contaminated by wideband signal energy.

**Details**:
- Used 10th percentile, which included signal energy for wideband signals (20-30 MHz VTX)
- Result: Artificially high noise floor, compressed display range, signals appeared flat

**Fix**: Implemented adaptive noise floor calculation:
- Detects wideband signals (>20% of bins 3+ dB above preliminary 2nd percentile)
- Uses 2nd percentile for wideband signals, 10th percentile for narrowband
- Uses instant spectrum for detection (not averaged) for accurate classification

### 6. DC Removal Too Aggressive

**Problem**: Block-mean DC removal was too aggressive for wide analog FM video signals.

**Impact**: 
- Subtracting mean of entire block can distort wideband signals
- Especially problematic if signal is tuned at DC in FFT span

**Fix**: Made DC removal configurable via `SPEAR_DC_REMOVAL` environment variable (default: `false`).

**Recommendation**: Tune LO off-center by 5-10 MHz instead of using DC removal for wideband signals.

### 7. UI Downsampling Method

**Problem**: UI downsampling used `max` method, which doesn't preserve energy spread for wideband signals.

**Fix**: Use `mean` method for wideband signals when downsampling from 65536 to 4096 points.

### 8. Display Range Compression

**Problem**: Display range autoscaling was compressing wideband signals.

**Fix**: 
- Calculate actual peak in data and ensure display range includes it with 5 dB margin
- Fixed 35 dB display range for wideband signals (when noise floor < -75 dBFS)

## Debugging Process

### Phase 1: Initial Investigation

1. **User Report**: VTX signal not visible in FFT/waterfall
2. **Initial Hypothesis**: Gain, sample rate, or FFT size issue
3. **Testing**: Created diagnostic scripts to test hardware directly
4. **Finding**: Signal was present in hardware, but not visible in UI

### Phase 2: Display Investigation

1. **Hypothesis**: UI display logic issue
2. **Testing**: Added extensive logging to UI and backend
3. **Finding**: Data was present in backend, but display range was compressed
4. **Fix Attempt**: Adjusted display range calculation
5. **Result**: Still flat, but now jumping around

### Phase 3: Noise Floor Investigation

1. **Hypothesis**: Noise floor calculation contaminated by signal
2. **Testing**: Compared noise floor with TX on vs. TX off
3. **Finding**: Noise floor was similar in both cases (contaminated)
4. **Fix**: Implemented adaptive noise floor calculation
5. **Result**: Improved, but signal still not clearly visible

### Phase 4: Buffer and Sample Processing

1. **Hypothesis**: Buffer size calculations or sample unpacking issues
2. **External Assessment**: User provided detailed assessment of buffer math errors
3. **Finding**: Multiple issues:
   - Buffer size calculations wrong (2 bytes vs. 4 bytes)
   - Stale tail data in buffers
   - DC removal too aggressive
4. **Fix**: Implemented all fixes
5. **Result**: Still not visible

### Phase 5: Hardware Truth Investigation

1. **Hypothesis**: Actual hardware parameters don't match requested values
2. **Testing**: Added hardware truth logging (requested vs. actual)
3. **Finding**: Frequency readback showed incorrect values
4. **Investigation**: Discovered `ctypes.c_uint32` bug in function signature
5. **Fix**: Changed to `ctypes.c_uint64`
6. **Result**: **BREAKTHROUGH** - Signal now visible!

## Key Lessons Learned

### 1. Hardware Truth First

Always verify actual hardware parameters match requested values. bladeRF may apply different values than requested, especially for bandwidth. Logging requested vs. actual values is essential for debugging.

### 2. Type Safety Matters

`ctypes` function signatures must match C header definitions exactly. A `uint32` vs. `uint64` difference can cause silent failures that are extremely difficult to detect.

### 3. Buffer Math is Critical

Incorrect buffer size calculations can cause subtle but critical issues:
- USB memory exhaustion
- Stale data contamination
- Timing weirdness
- Spectrum inconsistency

### 4. Wideband vs. Narrowband

Different signal types require different processing approaches:
- **Wideband signals** (20-30 MHz VTX): Appear as plateaus, need different noise floor calculation, DC removal can distort
- **Narrowband signals** (ELRS bursts): Appear as sharp peaks, standard processing works

### 5. Race Conditions are Subtle

Thread synchronization is critical for hardware device access. Rapid UI interactions can cause race conditions that only manifest under specific timing conditions.

### 6. Debugging Tools are Essential

Creating diagnostic scripts that bypass the application and test hardware directly was crucial for isolating the issue.

## Verification

After implementing all fixes:

1. **Frequency Tuning**: Verified with `scripts/verify_frequency.py`
   - Frequency readback now matches requested value
   - Tested at 5910 MHz, 5917 MHz, 5925 MHz

2. **Signal Visibility**: VTX signal now clearly visible in FFT/waterfall
   - Peak power: -72 to -74 dBFS (above noise floor)
   - SNR: ~18-19 dB
   - Wideband detection: Working correctly

3. **Hardware Truth Logging**: Confirms correct configuration
   - Sample rate: Matches requested
   - Bandwidth: May differ slightly (expected for bladeRF)
   - Frequency: Now matches requested (was wrong before)

## Files Modified

### Core SDR Driver
- `spear_edge/core/sdr/bladerf_native.py`
  - Frequency function signature fix (uint32 → uint64)
  - Buffer size calculation corrections
  - Stale tail data zeroing
  - Thread-safe gain operations
  - Device close/reopen for stream reconfiguration
  - Hardware truth logging
  - Q11 scaling locked

### FFT Processing
- `spear_edge/core/scan/scan_task.py`
  - Adaptive noise floor calculation
  - DC removal made optional
  - Default smoothing reduced (0.01 → 0.1)
  - Smoothing control API

### UI
- `spear_edge/ui/web/app.js`
  - Gain slider debouncing
  - Wideband downsampling (mean method)
  - Display range optimization
  - Enhanced logging
  - RX port display

- `spear_edge/ui/web/index.html`
  - Smoothing slider control
  - RX port display

### API
- `spear_edge/api/http/routes_tasking.py`
  - Smoothing control endpoint

### Settings
- `spear_edge/settings.py`
  - DC removal configuration option

### Diagnostic Scripts
- `scripts/verify_frequency.py` (new)
- `scripts/test_sdr_signal.py` (new)
- `scripts/diagnose_vtx.py` (new)
- `scripts/diagnose_vtx_edge.py` (new)
- `scripts/diagnose_display_issue.py` (new)

## Recommendations for Future Debugging

1. **Always log hardware truth**: Requested vs. actual parameters
2. **Test hardware directly**: Create diagnostic scripts that bypass the application
3. **Verify type signatures**: Check `ctypes` function signatures match C headers
4. **Check buffer math**: Verify all buffer size calculations are correct
5. **Test with known signals**: Use signals with known characteristics (frequency, bandwidth, power)
6. **Consider signal type**: Wideband vs. narrowband signals need different processing
7. **Monitor race conditions**: Add thread synchronization for hardware access
8. **Use external tools**: Compare with spectrum analyzers or other SDR software

## Conclusion

The frequency tuning bug was the primary blocker, but the debugging process revealed multiple real issues that have been fixed. All fixes remain valuable and improve overall system reliability. The key takeaway is to always verify hardware truth and ensure type safety in low-level bindings.
