# bladeRF Configuration Guide

## Overview

This guide provides best practices and recommended configurations for bladeRF 2.0 micro (xA4/xA9) hardware, specifically for use on embedded systems like the Jetson Orin Nano.

**Target Audience**: SPEAR Tripwire project and other bladeRF users  
**Hardware**: bladeRF 2.0 micro (xA4/xA9)  
**Library**: libbladeRF (via ctypes or SoapySDR)

---

## Hardware Specifications

### bladeRF 2.0 micro

- **Frequency Range**: 47 MHz - 6 GHz
- **Sample Rate**: Up to 61.44 MS/s
- **ADC Resolution**: 12-bit
- **RX Channels**: 2
- **TX Channels**: 2 (not used in SPEAR-Edge)
- **Interface**: USB 3.0
- **Power**: USB powered
- **IQ Format**: SC16_Q11 (4 bytes per complex sample)

### SC16_Q11 Format Details

- **Bytes per complex sample**: 4 (2 bytes I + 2 bytes Q)
- **Sample range**: [-2048, 2047] (11-bit signed)
- **Normalization**: Divide by 2048.0 to get [-1.0, 1.0) range
- **Endianness**: Little-endian (Intel x86/ARM)

---

## USB Buffer Configuration

### Memory Limit

**CRITICAL**: Linux USB filesystem memory pool is limited to 16MB by default:
```bash
# Check current limit:
cat /sys/module/usbcore/parameters/usbfs_memory_mb
# Default: 16

# To increase (requires root, not recommended):
# echo 32 > /sys/module/usbcore/parameters/usbfs_memory_mb
```

**Recommendation**: Keep total USB buffer memory under 16MB.

### Buffer Size Calculation

**CRITICAL**: SC16_Q11 uses 4 bytes per complex sample, not 2!

```python
# Total USB memory calculation:
total_memory_bytes = buffer_size * 4 * num_buffers

# Example:
buffer_size = 262144      # 256K samples
num_buffers = 4
total_memory = 262144 * 4 * 4 = 4,194,304 bytes = 4 MB
```

### Recommended Configurations

#### Very High Sample Rates (>40 MS/s)

```python
buffer_size = 262144      # 256K samples (1MB per buffer)
num_buffers = 4           # 4 buffers
num_transfers = 2         # Must be < num_buffers
total_memory = 4 MB       # Well under 16MB limit
```

**Use Case**: 40-60 MS/s sample rates  
**Read Size**: Up to 196K samples (75% of buffer size)

#### High Sample Rates (20-40 MS/s)

```python
buffer_size = 131072      # 128K samples (512KB per buffer)
num_buffers = 8           # 8 buffers
num_transfers = 4         # Must be < num_buffers
total_memory = 4 MB       # Well under 16MB limit
```

**Use Case**: 20-40 MS/s sample rates  
**Read Size**: Up to 96K samples (75% of buffer size)

#### Medium Sample Rates (10-20 MS/s)

```python
buffer_size = 65536       # 64K samples (256KB per buffer)
num_buffers = 16          # 16 buffers
num_transfers = 8         # Must be < num_buffers
total_memory = 4 MB       # Well under 16MB limit
```

**Use Case**: 10-20 MS/s sample rates  
**Read Size**: Up to 48K samples (75% of buffer size)

#### Low Sample Rates (<10 MS/s)

```python
buffer_size = 32768       # 32K samples (128KB per buffer)
num_buffers = 32          # 32 buffers
num_transfers = 16        # Must be < num_buffers
total_memory = 4 MB       # Well under 16MB limit
```

**Use Case**: <10 MS/s sample rates  
**Read Size**: Up to 24K samples (75% of buffer size)

### Buffer Configuration Rules

1. **Total memory must be < 16MB**
2. **Buffer size must be larger than read size** (prevent timeouts)
3. **Use power-of-two sizes** (hardware requirement)
4. **num_transfers must be < num_buffers** (libbladeRF requirement)
5. **Conservative counts are better** (stability over performance)

---

## Read Size Configuration

### Power-of-Two Requirement

**CRITICAL**: bladeRF requires power-of-two read sizes. Non-power-of-two sizes cause buffer timeouts.

```python
# Valid read sizes:
8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576

# Invalid read sizes (will cause timeouts):
10000, 20000, 50000, etc.
```

### Recommended Read Sizes

Read sizes should be ~75% of buffer size to provide safety margin:

```python
# For buffer_size = 262144 (256K):
max_read_size = 196608  # 192K (75% of 256K)

# For buffer_size = 131072 (128K):
max_read_size = 98304   # 96K (75% of 128K)

# For buffer_size = 65536 (64K):
max_read_size = 49152   # 48K (75% of 64K)

# For buffer_size = 32768 (32K):
max_read_size = 24576   # 24K (75% of 32K)
```

### Adaptive Read Sizing

Calculate read size based on sample rate and desired chunk duration:

```python
# Aim for 8-15ms chunks (good balance)
chunk_duration_ms = 0.010  # 10ms
chunk = int(sample_rate_sps * chunk_duration_ms)

# Round up to next power-of-two
if chunk & (chunk - 1) != 0:
    chunk = 1 << (chunk - 1).bit_length()

# Clamp to maximum (75% of buffer size)
chunk = min(chunk, max_read_size)
```

---

## Timeout Configuration

### Adaptive Timeout

Timeout should scale with expected read time:

```python
expected_read_time_ms = (num_samples / sample_rate_sps) * 1000
multiplier = 4.0 if sample_rate_sps > 20_000_000 else 3.0
timeout_ms = max(50, min(300, int(expected_read_time_ms * multiplier)))
```

**Why**:
- High sample rates need longer timeouts (USB transfer overhead)
- Minimum 50ms prevents false timeouts
- Maximum 300ms prevents hanging on errors

### Example Timeouts

```python
# 10 MS/s, 96K samples:
expected = (96000 / 10e6) * 1000 = 9.6 ms
timeout = 9.6 * 3.0 = 28.8 ms → 50 ms (clamped to minimum)

# 40 MS/s, 192K samples:
expected = (192000 / 40e6) * 1000 = 4.8 ms
timeout = 4.8 * 4.0 = 19.2 ms → 50 ms (clamped to minimum)

# 60 MS/s, 192K samples:
expected = (192000 / 60e6) * 1000 = 3.2 ms
timeout = 3.2 * 4.0 = 12.8 ms → 50 ms (clamped to minimum)
```

---

## Stream Lifecycle

### CRITICAL: Correct Order

bladeRF requires a specific order for stream operations:

1. **Set sample rate FIRST**
2. **Set bandwidth**
3. **Set frequency**
4. **Enable RX channel**
5. **ONLY THEN** create and activate stream

**NEVER** activate stream before RF parameters are set.

### Example Implementation

```python
def configure_rf(self, freq, rate, bw, channel):
    ch = BLADERF_CHANNEL_RX(channel)
    
    # Step 1: Set sample rate FIRST
    actual_rate = ctypes.c_uint32()
    ret = _libbladeRF.bladerf_set_sample_rate(
        self.dev, ch, rate, ctypes.byref(actual_rate)
    )
    if ret != 0:
        raise RuntimeError(f"Failed to set sample rate: {ret}")
    
    # Step 2: Set bandwidth
    actual_bw = ctypes.c_uint32()
    ret = _libbladeRF.bladerf_set_bandwidth(
        self.dev, ch, bw, ctypes.byref(actual_bw)
    )
    if ret != 0:
        raise RuntimeError(f"Failed to set bandwidth: {ret}")
    
    # Step 3: Set frequency
    ret = _libbladeRF.bladerf_set_frequency(self.dev, ch, freq)
    if ret != 0:
        raise RuntimeError(f"Failed to set frequency: {ret}")
    
    # Step 4: Enable RX channel
    ret = _libbladeRF.bladerf_enable_module(self.dev, ch, True)
    if ret != 0:
        raise RuntimeError(f"Failed to enable RX module: {ret}")
    
    # Step 5: Configure and activate stream (LAST)
    self._setup_stream()
```

---

## Device Close/Reopen Strategy

### When to Close/Reopen

**Always close and reopen device when**:
- Reconfiguring active stream (sample rate, bandwidth changes)
- Changing buffer configuration
- After USB errors or timeouts

**Why**: `bladerf_sync_config()` does not free old USB buffers. Only `bladerf_close()` frees them.

### Implementation

```python
def _setup_stream(self):
    if self._stream_active:
        # Save current RF state
        saved_freq = self.center_freq_hz
        saved_rate = self.sample_rate_sps
        saved_bw = self.bandwidth_hz
        saved_gain = self.gain_db
        
        # Set stream inactive FIRST
        self._stream_active = False
        
        # Deactivate stream
        _libbladeRF.bladerf_enable_module(self.dev, ch, False)
        
        # CRITICAL: Wait for in-flight operations (300ms = max read timeout)
        time.sleep(0.3)
        
        # Close device (frees USB buffers)
        _libbladeRF.bladerf_close(self.dev)
        self.dev = None
        
        # Reopen device
        self._open_device()
        
        # Restore RF parameters
        self._configure_rf(saved_freq, saved_rate, saved_bw, channel)
        self.set_gain(saved_gain)
    
    # Now configure new stream
    _libbladeRF.bladerf_sync_config(...)
```

### Error Recovery

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

---

## Type Safety in ctypes Bindings

### Critical Function Signatures

Always verify function signatures match C header definitions:

```python
# Frequency: uint64_t (NOT uint32_t!)
_libbladeRF.bladerf_set_frequency.argtypes = [
    ctypes.c_void_p,      # bladerf *dev
    ctypes.c_int,         # bladerf_channel channel
    ctypes.c_uint64       # bladerf_frequency frequency
]

# Sample rate: uint32_t
_libbladeRF.bladerf_set_sample_rate.argtypes = [
    ctypes.c_void_p,      # bladerf *dev
    ctypes.c_int,         # bladerf_channel channel
    ctypes.c_uint32,      # unsigned int frequency
    ctypes.POINTER(ctypes.c_uint32)  # unsigned int *actual
]

# Bandwidth: uint32_t
_libbladeRF.bladerf_set_bandwidth.argtypes = [
    ctypes.c_void_p,      # bladerf *dev
    ctypes.c_int,         # bladerf_channel channel
    ctypes.c_uint32,      # unsigned int bandwidth
    ctypes.POINTER(ctypes.c_uint32)  # unsigned int *actual
]
```

### Verification

Always read back and verify actual values:

```python
# Set frequency
ret = _libbladeRF.bladerf_set_frequency(self.dev, ch, freq)

# Read back actual frequency
actual_freq = ctypes.c_uint64()
ret = _libbladeRF.bladerf_get_frequency(self.dev, ch, ctypes.byref(actual_freq))

# Log hardware truth
logger.info(
    f"Frequency: requested={freq} actual={actual_freq.value} "
    f"diff={abs(actual_freq.value - freq)/1e6:.3f} MHz"
)
```

---

## Thread Safety

### Serialize Hardware Access

```python
# Create lock for gain operations
self._gain_lock = threading.Lock()

def set_gain(self, gain_db: float):
    with self._gain_lock:
        # Re-check device inside lock
        if self.dev is None:
            return
        
        ch = BLADERF_CHANNEL_RX(self.rx_channel)
        ret = _libbladeRF.bladerf_set_gain(self.dev, ch, int(gain_db))
        if ret != 0:
            raise RuntimeError(f"Failed to set gain: {ret}")
```

### Serialize Configuration Requests

```python
# In API endpoint:
_config_lock = asyncio.Lock()

@router.post("/sdr/config")
async def set_sdr_config(req: SdrConfigRequest):
    async with _config_lock:  # Serialize all requests
        # ... configuration logic ...
```

---

## Sample Processing

### SC16_Q11 Conversion

```python
# Read raw buffer (int16 interleaved I/Q)
buf = (ctypes.c_int16 * (n * 2))()
ret = _libbladeRF.bladerf_sync_rx(self.dev, buf, n, None, timeout_ms)

# Convert to NumPy array
arr = np.frombuffer(buf, dtype=np.int16)

# Deinterleave I and Q
i = arr[0::2].astype(np.float32)
q = arr[1::2].astype(np.float32)

# Normalize (Q11 format: divide by 2048.0)
scale = 1.0 / 2048.0
i *= scale
q *= scale

# Create complex array
iq = i + 1j * q
```

### Zero Unused Tail

```python
# If using preallocated buffer, zero unused tail
valid_len = len(i)
self._conv_buf_iq[:valid_len] = iq
self._conv_buf_iq[valid_len:] = 0  # Zero unused tail
return self._conv_buf_iq[:valid_len]
```

---

## Performance Optimization

### Ring Buffer Sizing

```python
# Adaptive ring buffer size based on sample rate
ring_duration = 0.3 if sample_rate_sps > 20_000_000 else 0.5
ring_size = int(sample_rate_sps * ring_duration)
```

**Why**: High sample rates need smaller buffers to save memory.

### Pre-allocate Buffers

```python
# Pre-allocate conversion buffers for maximum expected size
max_expected_size = 1048576  # 1M samples
self._conv_buf_i = np.empty(max_expected_size, dtype=np.float32)
self._conv_buf_q = np.empty(max_expected_size, dtype=np.float32)
self._conv_buf_iq = np.empty(max_expected_size, dtype=np.complex64)
```

**Why**: Avoids reallocation when chunk sizes change, improving performance.

---

## Troubleshooting

### LIBUSB_ERROR_NO_MEM

**Symptom**: `LIBUSB_ERROR_NO_MEM` errors at high sample rates

**Cause**: USB buffer memory exceeds 16MB limit

**Fix**:
1. Reduce `num_buffers` and `num_transfers`
2. Reduce `buffer_size` if possible
3. Close and reopen device to free old buffers
4. Verify calculation: `buffer_size * 4 * num_buffers < 16MB`

### Stream Returns 0 Samples

**Symptom**: `read_samples()` returns empty array

**Causes**:
1. Stream not activated
2. Read size not power-of-two
3. Stream deactivated during read
4. Device closed during read

**Fix**:
1. Verify stream lifecycle order
2. Check read size is power-of-two
3. Check `_stream_active` flag before reading
4. Ensure device is not closed during read

### Frequency Tuning Incorrect

**Symptom**: SDR tunes to wrong frequency (especially >4.29 GHz)

**Cause**: `ctypes.c_uint32` instead of `ctypes.c_uint64` in function signature

**Fix**: Change function signature to use `ctypes.c_uint64`

### Segmentation Fault on Config Change

**Symptom**: Crash when changing sample rate or other settings

**Cause**: Device closed while RX task thread is reading

**Fix**:
1. Set `_stream_active = False` first
2. Wait 300ms for in-flight reads to complete
3. Then close device
4. Check stream state at beginning of `read_samples()`

---

## Testing Checklist

- [ ] Frequency tuning accuracy (especially >4.29 GHz)
- [ ] USB buffer memory calculations (verify <16MB)
- [ ] Power-of-two read sizes
- [ ] Stream lifecycle order
- [ ] Device close/reopen recovery
- [ ] Concurrent configuration requests
- [ ] Rapid config changes
- [ ] Config changes during active stream
- [ ] Hardware truth logging (requested vs. actual)
- [ ] Thread safety (multiple threads accessing device)

---

## References

- libbladeRF Documentation: https://github.com/Nuand/bladeRF
- bladeRF Hardware Manual: https://www.nuand.com/bladeRF-doc/
- USB Memory Limits: `/sys/module/usbcore/parameters/usbfs_memory_mb`
- SC16_Q11 Format: See libbladeRF.h header file

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: SPEAR-Edge Development Team
