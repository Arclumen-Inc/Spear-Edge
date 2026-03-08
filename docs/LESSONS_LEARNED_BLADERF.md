# Lessons Learned: bladeRF Configuration and Stability

## Overview

This document captures critical lessons learned during development and debugging of SPEAR-Edge's bladeRF integration. These issues and solutions are applicable to any project using bladeRF hardware, particularly the bladeRF 2.0 micro (xA4/xA9) on embedded systems like the Jetson Orin Nano.

**Target Audience**: SPEAR Tripwire project and other bladeRF users  
**Hardware**: bladeRF 2.0 micro (xA4/xA9), Jetson Orin Nano  
**Library**: libbladeRF (via ctypes)

---

## Critical Issues Discovered

### 1. Frequency Tuning Type Mismatch (CRITICAL BUG)

**Problem**: Frequencies above 4.29 GHz were silently truncated, causing the SDR to tune to completely wrong frequencies.

**Root Cause**: 
- `bladerf_set_frequency()` function signature used `ctypes.c_uint32` instead of `ctypes.c_uint64`
- Maximum value for `uint32` is 4,294,967,295 (4.29 GHz)
- Frequencies above this were truncated (e.g., 5910 MHz → 1615 MHz)

**Impact**: 
- VTX signals and other high-frequency signals were completely invisible
- No error messages - function call succeeded but with wrong value
- Extremely difficult to detect without hardware truth logging

**Fix**:
```python
# WRONG:
_libbladeRF.bladerf_set_frequency.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]

# CORRECT:
_libbladeRF.bladerf_set_frequency.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
```

**Lesson**: Always verify `ctypes` function signatures match C header definitions exactly. Type mismatches can cause silent failures.

**Verification**: Always log requested vs. actual frequency values after configuration.

---

### 2. USB Buffer Memory Calculation Error

**Problem**: Buffer size calculations incorrectly assumed 2 bytes per complex sample, causing USB memory exhaustion.

**Root Cause**:
- SC16_Q11 format uses **4 bytes per complex sample** (2 bytes I + 2 bytes Q)
- Calculations assumed 2 bytes per sample
- USB buffer memory usage was 2x larger than calculated
- Exceeded 16MB USB memory limit (`usbfs_memory_mb`)

**Impact**:
- `LIBUSB_ERROR_NO_MEM` errors at high sample rates
- Stream failures and crashes
- Unpredictable behavior

**Fix**:
```python
# WRONG:
# USB memory = buffer_size (samples) × 2 bytes/sample × num_buffers

# CORRECT:
# USB memory = buffer_size (samples) × 4 bytes/complex_sample × num_buffers
# Format: SC16_Q11 (4 bytes per complex sample: 2 bytes I + 2 bytes Q)

# Example:
buffer_size = 262144  # 256K samples
num_buffers = 4
# Actual memory: 256K × 4 bytes × 4 buffers = 4MB (not 2MB!)
```

**Lesson**: Always verify buffer size calculations match the actual data format. SC16_Q11 uses 4 bytes per complex sample, not 2.

**Recommendation**: Calculate and log total USB memory usage to ensure it stays under 16MB limit.

---

### 3. Device Close/Reopen Memory Leak

**Problem**: `bladerf_sync_config()` does not free old USB buffers on reconfiguration, causing memory leak.

**Root Cause**:
- libbladeRF allocates USB buffers via `bladerf_sync_config()`
- These buffers are NOT freed when `bladerf_sync_config()` is called again
- Buffers are only freed when `bladerf_close()` is called
- Repeated reconfigurations without closing device cause USB memory exhaustion

**Impact**:
- USB memory pool exhaustion after multiple reconfigurations
- `LIBUSB_ERROR_NO_MEM` errors
- System instability

**Fix**: Close and reopen device when reconfiguring active stream:
```python
if self._stream_active:
    # Save current RF state
    saved_freq = self.center_freq_hz
    saved_rate = self.sample_rate_sps
    # ... save other parameters ...
    
    # Set stream inactive FIRST
    self._stream_active = False
    
    # Deactivate stream
    # ... disable RX modules ...
    
    # Wait for in-flight operations (300ms = max read timeout)
    time.sleep(0.3)
    
    # Close device (frees USB buffers)
    _libbladeRF.bladerf_close(self.dev)
    self.dev = None
    
    # Reopen device
    self._open_device()
    
    # Restore RF parameters
    # ... restore frequency, sample rate, etc. ...
```

**Lesson**: libbladeRF does not automatically free buffers on reconfiguration. Always close device when changing stream configuration.

**Recommendation**: Implement device close/reopen for any stream reconfiguration, not just sample rate changes.

---

### 4. Race Condition During Device Close/Reopen

**Problem**: Segmentation fault when changing sample rate or other settings while stream is active.

**Root Cause**:
- RX task thread calls `read_samples()` while device is being closed/reopened
- Device pointer becomes invalid (`self.dev = None`) while `bladerf_sync_rx()` is still running
- Insufficient delay between setting `_stream_active = False` and closing device

**Impact**:
- Segmentation faults
- Application crashes
- Data corruption

**Fix**:
```python
# Set stream inactive FIRST
self._stream_active = False

# Deactivate stream
# ... disable RX modules ...

# CRITICAL: Wait long enough for in-flight reads to complete
# Max read timeout is 300ms, so wait at least that long
time.sleep(0.3)

# Now safe to close device
_libbladeRF.bladerf_close(self.dev)

# In read_samples(), check stream state FIRST:
if not self._stream_active or self.dev is None:
    return np.empty(0, dtype=np.complex64)
```

**Lesson**: Always wait for in-flight operations to complete before closing device. Check stream state at the beginning of read operations.

**Recommendation**: Use a delay equal to or greater than the maximum read timeout (300ms for high sample rates).

---

### 5. Concurrent Configuration Requests

**Problem**: Multiple rapid config changes can cause overlapping device close/reopen operations.

**Root Cause**:
- No serialization of SDR configuration requests
- Multiple API calls can execute simultaneously
- Device can be closed by one request while another tries to use it

**Impact**:
- Race conditions
- Device state inconsistencies
- Crashes

**Fix**: Serialize all configuration requests:
```python
# In API endpoint:
_config_lock = asyncio.Lock()

@router.post("/sdr/config")
async def set_sdr_config(req: SdrConfigRequest):
    async with _config_lock:  # Serialize all requests
        # ... configuration logic ...
```

**Lesson**: Hardware device access must be serialized. Multiple concurrent reconfigurations will cause race conditions.

**Recommendation**: Use a lock or semaphore to serialize all device configuration operations.

---

### 6. Stale Tail Data in Preallocated Buffers

**Problem**: Preallocated conversion buffers contained stale data from previous reads.

**Root Cause**:
- Only prefix of buffer was written with new data
- Downstream code could access full buffer, including stale tail data
- Wideband signals appeared smeared or inconsistent

**Impact**:
- Wideband signals look smeared
- Inconsistent spectrum display
- False signal detections

**Fix**: Explicitly zero unused tail:
```python
valid_len = len(i)
self._conv_buf_iq[valid_len:] = 0  # Zero unused tail
return self._conv_buf_iq[:valid_len]
```

**Lesson**: Always zero unused portions of preallocated buffers, or return only the valid portion.

**Recommendation**: Either return a copy of the valid portion, or zero the tail before returning.

---

### 7. Noise Floor Calculation for Wideband Signals

**Problem**: Noise floor calculation was contaminated by wideband signal energy.

**Root Cause**:
- Used 10th percentile for noise floor calculation
- Wideband signals (20-30 MHz) occupy many bins
- 10th percentile included signal energy, not just noise

**Impact**:
- Artificially high noise floor
- Compressed display range
- Signals appeared flat

**Fix**: Adaptive noise floor calculation:
```python
# Detect wideband signals (>20% of bins 3+ dB above preliminary floor)
# Use instant spectrum for detection (not averaged)
signal_bins = np.sum(center_spectrum_inst > (prelim_floor + 3.0))
is_wideband = (signal_bins / len(center_spectrum_inst)) > 0.20

# Use different percentile based on signal type
if is_wideband:
    noise_floor = np.percentile(center_spectrum_avg, 2)  # 2nd percentile
else:
    noise_floor = np.percentile(center_spectrum_avg, 10)  # 10th percentile
```

**Lesson**: Wideband and narrowband signals require different processing approaches. Noise floor calculation must account for signal type.

**Recommendation**: Implement adaptive algorithms that detect signal characteristics and adjust processing accordingly.

---

## Best Practices

### 1. Hardware Truth Logging

**Always log requested vs. actual hardware parameters**:
```python
# After configuration, read back actual values
actual_freq = ctypes.c_uint64()
ret = _libbladeRF.bladerf_get_frequency(self.dev, ch, ctypes.byref(actual_freq))

# Log hardware truth
logger.info(
    f"RX{ch_num} configured: "
    f"req_sr={requested_rate:.0f} act_sr={actual_rate.value:.0f} "
    f"req_bw={requested_bw:.0f} act_bw={actual_bw.value:.0f} "
    f"req_fc={requested_freq:.0f} act_fc={actual_freq.value:.0f} "
    f"gain_mode={gain_mode} gain={gain}"
)
```

**Why**: bladeRF may apply different values than requested, especially for bandwidth. This is critical for debugging.

---

### 2. Type Safety in ctypes Bindings

**Always verify function signatures match C headers exactly**:
```python
# Check libbladeRF.h for exact types:
# bladerf_frequency is uint64_t, not uint32_t
_libbladeRF.bladerf_set_frequency.argtypes = [
    ctypes.c_void_p,      # bladerf *dev
    ctypes.c_int,         # bladerf_channel channel
    ctypes.c_uint64       # bladerf_frequency frequency (NOT uint32!)
]
```

**Why**: Type mismatches cause silent failures that are extremely difficult to detect.

---

### 3. USB Buffer Sizing

**Calculate buffer memory correctly**:
```python
# SC16_Q11 format: 4 bytes per complex sample
bytes_per_complex_sample = 4

# Total USB memory:
total_memory = buffer_size * bytes_per_complex_sample * num_buffers

# Must be < 16MB (usbfs_memory_mb limit)
assert total_memory < 16 * 1024 * 1024, "USB memory limit exceeded"
```

**Why**: Incorrect calculations lead to USB memory exhaustion and stream failures.

---

### 4. Stream Lifecycle Order

**CRITICAL: bladeRF requires specific order for stream operations**:
1. Set sample rate FIRST
2. Set bandwidth
3. Set frequency
4. Enable RX channel
5. **ONLY THEN** create and activate stream

**NEVER** activate stream before RF parameters are set.

**Example**:
```python
# CORRECT ORDER:
self.dev.setSampleRate(SOAPY_SDR_RX, ch, float(self.sample_rate_sps))
self.dev.setBandwidth(SOAPY_SDR_RX, ch, float(self.bandwidth_hz))
self.dev.setFrequency(SOAPY_SDR_RX, ch, float(self.center_freq_hz))
# Enable RX
self.dev.writeSetting("ENABLE_CHANNEL", "RX", "true")
# STREAM SETUP MUST HAPPEN LAST
self._setup_stream()
```

---

### 5. Read Size Requirements

**ALWAYS use power-of-two read sizes**:
```python
# bladeRF requires power-of-two read sizes
if n & (n - 1) != 0:
    # Not a power-of-two, round up
    n = 1 << (n - 1).bit_length()
```

**Why**: Non-power-of-two sizes cause buffer timeouts and stream failures.

---

### 6. Error Recovery

**Implement error recovery for device operations**:
```python
try:
    self._open_device()
except RuntimeError as e:
    logger.error(f"Failed to reopen device: {e}")
    # Attempt recovery: try once more after delay
    time.sleep(0.1)
    try:
        self._open_device()
    except RuntimeError as e2:
        logger.error(f"Device reopen failed after retry: {e2}")
        raise  # Re-raise to fail fast
```

**Why**: Device operations can fail due to USB issues, system load, etc. Retry logic improves reliability.

---

### 7. Thread Safety

**Protect hardware access with locks**:
```python
# Serialize gain operations
self._gain_lock = threading.Lock()

def set_gain(self, gain_db: float):
    with self._gain_lock:
        # Re-check device inside lock
        if self.dev is None:
            return
        # ... set gain ...
```

**Why**: Multiple threads (UI, capture, scan) may access hardware concurrently. Locks prevent race conditions.

---

### 8. UI Debouncing

**Debounce rapid user input**:
```javascript
// Debounce config changes to prevent rapid API calls
let configDebounceTimer = null;
function applySdrConfigDebounced() {
    clearTimeout(configDebounceTimer);
    configDebounceTimer = setTimeout(() => {
        applySdrConfig();
    }, 300); // 300ms debounce
}
```

**Why**: Rapid input changes can trigger many concurrent API calls, causing race conditions and crashes.

---

## Configuration Recommendations

### USB Buffer Configuration

For stable operation, use conservative buffer counts that fit within 16MB limit:

```python
# Sample rate > 40 MS/s:
buffer_size = 262144  # 256K samples (1MB per buffer)
num_buffers = 4       # 4MB total

# Sample rate 20-40 MS/s:
buffer_size = 131072  # 128K samples (512KB per buffer)
num_buffers = 8       # 4MB total

# Sample rate 10-20 MS/s:
buffer_size = 65536   # 64K samples (256KB per buffer)
num_buffers = 16      # 4MB total

# Sample rate < 10 MS/s:
buffer_size = 32768   # 32K samples (128KB per buffer)
num_buffers = 32      # 4MB total
```

**Key Points**:
- Total memory must be < 16MB
- Buffer size must be larger than read size
- Use power-of-two sizes
- Conservative counts are better than aggressive

---

### Read Size Configuration

Read sizes should be ~75% of buffer size to prevent timeouts:

```python
# For buffer_size = 262144 (256K):
max_read_size = 196608  # 192K (75% of 256K)

# For buffer_size = 131072 (128K):
max_read_size = 98304   # 96K (75% of 128K)
```

**Why**: Read sizes larger than buffer size cause timeouts. Smaller reads provide safety margin.

---

### Timeout Configuration

Adaptive timeout based on sample rate and read size:

```python
expected_read_time_ms = (num_samples / sample_rate) * 1000
multiplier = 4.0 if sample_rate > 20_000_000 else 3.0
timeout_ms = max(50, min(300, int(expected_read_time_ms * multiplier)))
```

**Why**: High sample rates need longer timeouts due to USB transfer overhead.

---

## Common Pitfalls

### 1. Assuming 2 Bytes Per Sample

**WRONG**: SC16_Q11 uses 2 bytes per sample  
**CORRECT**: SC16_Q11 uses 4 bytes per complex sample (2 bytes I + 2 bytes Q)

### 2. Not Closing Device on Reconfiguration

**WRONG**: Call `bladerf_sync_config()` multiple times without closing device  
**CORRECT**: Close and reopen device when reconfiguring active stream

### 3. Insufficient Delay During Close/Reopen

**WRONG**: 50ms delay before closing device  
**CORRECT**: 300ms delay (matches max read timeout)

### 4. No Hardware Truth Logging

**WRONG**: Assume requested values match actual values  
**CORRECT**: Always read back and log actual hardware values

### 5. Type Mismatches in ctypes

**WRONG**: Use `ctypes.c_uint32` for frequencies  
**CORRECT**: Use `ctypes.c_uint64` for frequencies (supports up to 6 GHz)

### 6. Non-Power-of-Two Read Sizes

**WRONG**: Use arbitrary read sizes (e.g., 10000)  
**CORRECT**: Always use power-of-two sizes (8192, 16384, etc.)

### 7. No Request Serialization

**WRONG**: Allow concurrent configuration requests  
**CORRECT**: Serialize all device configuration operations

---

## Testing Recommendations

### 1. Frequency Verification

Always verify frequency tuning accuracy:
```python
# Test at various frequencies, especially > 4.29 GHz
test_frequencies = [915e6, 2400e6, 5800e6, 5910e6]
for freq in test_frequencies:
    set_frequency(freq)
    actual = get_frequency()
    assert abs(actual - freq) < 1e6, f"Frequency mismatch: {actual} vs {freq}"
```

### 2. Rapid Configuration Changes

Test rapid config changes to verify stability:
```python
# Rapidly change frequency, sample rate, gain
for i in range(10):
    change_frequency(random_freq())
    change_sample_rate(random_rate())
    await asyncio.sleep(0.1)
```

### 3. Concurrent Requests

Test concurrent requests to verify serialization:
```python
# Send multiple config requests simultaneously
tasks = [send_config(freq) for freq in frequencies]
results = await asyncio.gather(*tasks)
# All should succeed without crashes
```

### 4. Stream Restart

Test stream restart during active operation:
```python
# Start stream
start_scan(freq=915e6, rate=10e6)

# Change settings while stream is active
change_frequency(2400e6)
change_sample_rate(20e6)

# Verify no crashes, stream restarts correctly
```

---

## Summary

### Critical Issues
1. ✅ Frequency type mismatch (uint32 → uint64)
2. ✅ USB buffer memory calculation error (2 bytes → 4 bytes)
3. ✅ Device close/reopen memory leak
4. ✅ Race condition during device close/reopen
5. ✅ Concurrent configuration requests

### Best Practices
1. ✅ Always log hardware truth (requested vs. actual)
2. ✅ Verify ctypes function signatures match C headers
3. ✅ Calculate USB buffer memory correctly (4 bytes/complex sample)
4. ✅ Follow correct stream lifecycle order
5. ✅ Use power-of-two read sizes
6. ✅ Implement error recovery
7. ✅ Protect hardware access with locks
8. ✅ Debounce rapid user input

### Configuration
- Conservative USB buffer counts (fit within 16MB)
- Read sizes ~75% of buffer size
- Adaptive timeouts based on sample rate
- Device close/reopen for stream reconfiguration

---

## References

- libbladeRF Documentation: https://github.com/Nuand/bladeRF
- bladeRF Hardware Manual: https://www.nuand.com/bladeRF-doc/
- USB Memory Limits: `/sys/module/usbcore/parameters/usbfs_memory_mb`

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: SPEAR-Edge Development Team
