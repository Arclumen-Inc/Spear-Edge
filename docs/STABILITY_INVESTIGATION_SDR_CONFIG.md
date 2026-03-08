# SDR Configuration Stability Investigation

## Overview

This document investigates stability issues when changing SDR settings, particularly crashes that occurred during debugging when modifying frequency, sample rate, gain, and other parameters.

## Identified Issues

### 1. **No Protection Against Concurrent Config Requests** ⚠️ CRITICAL

**Problem**: The `/api/live/sdr/config` endpoint has no locking or queuing mechanism. Multiple rapid config changes can cause:

- **Overlapping reconfigurations**: If user rapidly changes settings (e.g., typing frequency, moving slider), multiple API calls can be in flight simultaneously
- **Device close/reopen race**: Multiple requests can trigger device close/reopen simultaneously
- **Orchestrator lock contention**: While `start_scan()` is protected by `asyncio.Lock()`, the API endpoint itself is not, so multiple requests can queue up `start_scan()` calls

**Location**: `spear_edge/api/http/routes_tasking.py:234-301`

**Impact**: 
- Device can be closed while another thread is trying to use it
- Multiple `start_scan()` calls can execute concurrently (though protected by lock, they queue up)
- Device state can become inconsistent

**Evidence**:
```python
@router.post("/sdr/config")
async def set_sdr_config(req: SdrConfigRequest):
    # No locking or request queuing here
    # Multiple concurrent requests can execute simultaneously
    if scan_running and not only_gain_changed:
        await orchestrator.start_scan(...)  # Protected by lock, but requests queue up
```

### 2. **Insufficient Delay During Device Close/Reopen** ⚠️ HIGH

**Problem**: In `_setup_stream()`, when reconfiguring an active stream, there's only a 50ms delay before closing the device. This may not be enough if:

- RX task thread is in the middle of a long `read_samples()` call (timeout can be up to 300ms)
- USB transfer is in progress
- Multiple threads are accessing the device

**Location**: `spear_edge/core/sdr/bladerf_native.py:790-791`

**Current Code**:
```python
# Brief delay to allow any in-flight read_samples() calls to complete
time.sleep(0.05)  # Only 50ms!

# Close device to free USB buffers
if self.dev is not None:
    _libbladerf.bladerf_close(self.dev)
```

**Impact**:
- `read_samples()` can be called with an invalid device pointer → segfault
- USB transfers can be interrupted → corruption
- Race condition between setting `_stream_active = False` and actual device close

**Evidence**: The `read_samples()` method checks `_stream_active` and `self.dev`, but there's a window between setting the flag and closing the device where:
1. `_stream_active` is set to `False` (line 776)
2. RX task thread checks `_stream_active` → returns early (good)
3. BUT: If RX task is already inside `bladerf_sync_rx()` call, it can't be interrupted
4. Device is closed while `bladerf_sync_rx()` is still running → segfault

### 3. **No Error Recovery After Device Reopen Failure** ⚠️ HIGH

**Problem**: If `_open_device()` fails after closing the device, the device is left in a closed state (`self.dev = None`), but the code continues. Subsequent operations will fail or crash.

**Location**: `spear_edge/core/sdr/bladerf_native.py:799-800`

**Current Code**:
```python
# Reopen device
self._open_device()  # Can raise RuntimeError
logger.info("[STREAM] Device reopened")

# Continue with stream setup...
```

**Impact**:
- If `_open_device()` fails, `self.dev = None`
- Subsequent code tries to use `self.dev` → segfault or errors
- No rollback or recovery mechanism
- System left in inconsistent state

**Evidence**: `_open_device()` can raise `RuntimeError` if device open fails (line 247), but this exception is not caught in `_setup_stream()`.

### 4. **Gain Restore Timing Issue** ⚠️ MEDIUM

**Problem**: Gain is stored for restore after stream setup (line 849), but if stream setup fails, gain is never restored. Also, gain restore happens after stream is configured, which means there's a window where gain is incorrect.

**Location**: `spear_edge/core/sdr/bladerf_native.py:847-849`

**Current Code**:
```python
# Restore gain (must be after stream is configured, so we'll do it later)
# Store for after stream setup
pending_gain_restore = (saved_gain, saved_gain_mode, channels_to_config)
```

**Impact**:
- If stream setup fails, gain is not restored
- Gain is incorrect during stream setup window
- Minor issue, but can cause confusion

### 5. **Redundant Stream Deactivation** ⚠️ LOW

**Problem**: In `_setup_stream()`, `_stream_active` is set to `False` manually (line 776), then `_deactivate_stream()` is called (implicitly via device close), which also sets `_stream_active = False` (line 971). This is redundant but harmless.

**Location**: `spear_edge/core/sdr/bladerf_native.py:776, 971`

**Impact**: Minor - redundant operations, but no functional issue.

### 6. **UI Debouncing Only for Gain Slider** ⚠️ MEDIUM

**Problem**: Only the gain slider has debouncing (200ms). Other inputs (frequency, sample rate, bandwidth) can trigger rapid API calls if user is typing or changing values.

**Location**: `spear_edge/ui/web/app.js:3000-3011` (gain slider), but no debouncing for other inputs

**Impact**:
- Rapid frequency changes can cause multiple concurrent reconfigurations
- Sample rate changes can trigger multiple device close/reopen cycles
- Can exacerbate issue #1 (concurrent requests)

**Evidence**: 
- Gain slider: Has 200ms debounce timeout
- Frequency input: No debouncing, calls `applySdrConfig()` on every change
- Sample rate input: No debouncing, calls `applySdrConfig()` on every change

### 7. **No Request Cancellation** ⚠️ MEDIUM

**Problem**: If a config request is in progress and a new one arrives, the old request is not cancelled. Both can proceed, causing:
- Device closed by request 1, reopened by request 2, then request 1 tries to use it
- Multiple `start_scan()` calls queued up
- Inconsistent final state

**Location**: `spear_edge/api/http/routes_tasking.py:234-301`

**Impact**: Can cause crashes if device is closed/reopened multiple times rapidly.

## Root Cause Analysis

The primary issue is **lack of serialization** for SDR configuration changes. The system has multiple layers of protection (orchestrator lock, stream active flag), but they don't prevent all race conditions:

1. **API Layer**: No locking or request queuing
2. **Device Layer**: Device close/reopen is not atomic with respect to concurrent operations
3. **Thread Safety**: RX task thread can be reading while device is being closed

## Crash Scenarios

### Scenario 1: Rapid Frequency Changes
1. User types "5910" in frequency field
2. Each keystroke triggers `applySdrConfig()`
3. Request 1: Starts device close/reopen
4. Request 2: Arrives while device is closed, tries to use device → segfault

### Scenario 2: Sample Rate Change During Active Stream
1. User changes sample rate while scan is running
2. `_setup_stream()` sets `_stream_active = False`
3. RX task thread is in middle of `bladerf_sync_rx()` call (can't be interrupted)
4. Device is closed after 50ms delay
5. `bladerf_sync_rx()` tries to use closed device → segfault

### Scenario 3: Concurrent Config + Capture
1. User changes SDR config
2. Capture starts simultaneously
3. Both try to reconfigure device
4. Device closed by config change, capture tries to use it → segfault

### Scenario 4: Device Reopen Failure
1. User changes sample rate
2. Device is closed
3. `_open_device()` fails (device busy, USB issue, etc.)
4. `self.dev = None`
5. Code continues, tries to use `self.dev` → segfault

## Recommended Fixes

### Fix 1: Add Request Serialization (CRITICAL)
**Priority**: HIGH
**Location**: `spear_edge/api/http/routes_tasking.py`

Add a lock or semaphore to serialize SDR config requests:
```python
# In bind() function, create a lock
_config_lock = asyncio.Lock()

@router.post("/sdr/config")
async def set_sdr_config(req: SdrConfigRequest):
    async with _config_lock:  # Serialize all config requests
        # ... existing code ...
```

### Fix 2: Increase Delay and Add Synchronization (HIGH)
**Priority**: HIGH
**Location**: `spear_edge/core/sdr/bladerf_native.py`

Increase delay and add proper synchronization:
```python
# Set stream inactive FIRST
self._stream_active = False
self._stream_configured = False

# Deactivate stream
# ... disable modules ...

# Wait longer for in-flight operations (300ms = max read timeout)
time.sleep(0.3)  # Increased from 0.05

# Additional check: ensure no reads are in progress
# Could add a read counter or event to track active reads
```

### Fix 3: Add Error Recovery (HIGH)
**Priority**: HIGH
**Location**: `spear_edge/core/sdr/bladerf_native.py`

Wrap device reopen in try/except and add recovery:
```python
try:
    self._open_device()
    logger.info("[STREAM] Device reopened")
except RuntimeError as e:
    logger.error(f"[STREAM] Failed to reopen device: {e}")
    # Attempt recovery: try once more after delay
    time.sleep(0.1)
    try:
        self._open_device()
        logger.info("[STREAM] Device reopened on retry")
    except RuntimeError as e2:
        logger.error(f"[STREAM] Device reopen failed after retry: {e2}")
        raise  # Re-raise to fail fast
```

### Fix 4: Add UI Debouncing for All Inputs (MEDIUM)
**Priority**: MEDIUM
**Location**: `spear_edge/ui/web/app.js`

Add debouncing for frequency, sample rate, and bandwidth inputs:
```javascript
let configDebounceTimeout;
function applySdrConfigDebounced() {
    clearTimeout(configDebounceTimeout);
    configDebounceTimeout = setTimeout(() => {
        applySdrConfig();
    }, 300); // 300ms debounce for all config changes
}
```

### Fix 5: Add Request Cancellation (MEDIUM)
**Priority**: MEDIUM
**Location**: `spear_edge/api/http/routes_tasking.py`

Track in-flight requests and cancel old ones:
```python
_inflight_config_request = None

@router.post("/sdr/config")
async def set_sdr_config(req: SdrConfigRequest):
    # Cancel previous request if still in flight
    if _inflight_config_request and not _inflight_config_request.done():
        _inflight_config_request.cancel()
    
    async with _config_lock:
        _inflight_config_request = asyncio.create_task(_do_config(req))
        try:
            return await _inflight_config_request
        except asyncio.CancelledError:
            return {"ok": False, "error": "cancelled"}
```

### Fix 6: Add Read Operation Tracking (LOW)
**Priority**: LOW
**Location**: `spear_edge/core/sdr/bladerf_native.py`

Track active read operations to ensure none are in progress before closing device:
```python
self._active_reads = 0
self._read_lock = threading.Lock()

def read_samples(self, num_samples: int) -> np.ndarray:
    with self._read_lock:
        self._active_reads += 1
    try:
        # ... existing read code ...
    finally:
        with self._read_lock:
            self._active_reads -= 1

# In _setup_stream(), wait for reads to complete:
while self._active_reads > 0:
    time.sleep(0.01)
```

## Testing Recommendations

1. **Rapid Config Changes**: Rapidly change frequency, sample rate, gain multiple times
2. **Config During Active Stream**: Change settings while scan is running
3. **Concurrent Operations**: Change config while capture is in progress
4. **Device Busy**: Simulate device busy condition (open in another process)
5. **USB Disconnect**: Simulate USB disconnect during reconfiguration
6. **Stress Test**: Send 10+ config requests per second for 30 seconds

## Priority Summary

1. **CRITICAL**: Add request serialization (Fix 1)
2. **HIGH**: Increase delay and add error recovery (Fixes 2, 3)
3. **MEDIUM**: Add UI debouncing and request cancellation (Fixes 4, 5)
4. **LOW**: Add read operation tracking (Fix 6)

## Files to Modify

1. `spear_edge/api/http/routes_tasking.py` - Add request serialization
2. `spear_edge/core/sdr/bladerf_native.py` - Improve device close/reopen safety
3. `spear_edge/ui/web/app.js` - Add debouncing for all inputs

## Notes

- The orchestrator's `asyncio.Lock()` protects `start_scan()`, but doesn't prevent multiple config requests from queuing up
- The `_stream_active` flag helps, but there's still a race condition window
- Device close/reopen is necessary for USB buffer management, but needs better synchronization
- UI debouncing would help reduce the frequency of concurrent requests
