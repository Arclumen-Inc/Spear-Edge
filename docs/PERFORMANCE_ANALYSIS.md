# SPEAR-Edge Performance Analysis & Optimization

## Performance Status

**Previous Limit:** System struggled above ~10 MS/s  
**Current Target:** 30-40 MS/s  
**Hardware Capability:** bladeRF 2.0 micro supports up to 61.44 MS/s (USB 3.0)  
**Jetson Orin Nano:** Should handle 30-40 MS/s with proper optimization

## Optimizations Implemented

### Phase 1: Quick Wins (COMPLETED)
✅ **Adaptive USB Buffer Sizing** - Buffers now scale with sample rate  
✅ **Adaptive Read Chunk Sizing** - Chunks scale up to 1M samples for high rates  
✅ **Adaptive Read Timeout** - Timeout scales with expected read time

### Phase 2: Medium Effort (COMPLETED)
✅ **Ring Buffer Optimization** - Reduced lock contention, optimized copies  
✅ **Memory Allocation Reduction** - Pre-allocated work arrays, buffer reuse  
✅ **FFT Pipeline Optimization** - In-place operations, pre-allocated arrays

### Expected Performance
- **Before**: ~10 MS/s maximum
- **After Phase 1**: 20-25 MS/s expected
- **After Phase 2**: 30-35 MS/s expected
- **Target**: 30-40 MS/s (achievable with hardware)

## Bottleneck Analysis

### 1. USB Buffer Configuration (HIGH PRIORITY)

**Current State:**
```python
# bladerf_native.py:685-688
num_buffers = 64        # Fixed, regardless of sample rate
buffer_size = 131072    # Fixed 128KB buffer
num_transfers = 16      # Fixed
stream_timeout_ms = 5000
```

**Issues:**
- **Fixed buffer size** doesn't scale with sample rate
- At 30 MS/s, 131072 samples = 4.37ms of data (very small)
- USB 3.0 can handle much larger transfers efficiently
- Not enough buffers for high-rate streaming

**Impact:**
- Frequent USB transfers (small buffers = more overhead)
- Higher risk of buffer underruns
- Increased CPU overhead from transfer management

**Optimization:**
- **Adaptive buffer sizing** based on sample rate
- For 30-40 MS/s: buffer_size should be 262144-524288 (8-16ms of data)
- Increase num_buffers to 128-256 for high rates
- Increase num_transfers to 32-64 for better USB pipeline

**Expected Gain:** 2-3x throughput improvement

---

### 2. Read Chunk Size (HIGH PRIORITY)

**Current State:**
```python
# orchestrator.py:224-229
chunk = int(sample_rate_sps * 0.02)  # 20ms
chunk = max(chunk, 32768)
chunk = min(chunk, 262144)

# rx_task.py:76-82
chunk = max(16384, self.chunk_size)
if sample_rate >= 10_000_000:
    chunk = max(chunk, 131072)
```

**Issues:**
- **Clamped to 262144** max - too small for 30-40 MS/s
- At 30 MS/s: 0.02 * 30e6 = 600k samples, but clamped to 262144 (8.7ms)
- Small chunks = more read calls = more overhead
- More lock contention on ring buffer

**Impact:**
- Excessive read calls (every 8.7ms at 30 MS/s)
- Lock contention on ring buffer
- CPU overhead from frequent context switches

**Optimization:**
- **Remove or increase max clamp** to 524288-1048576 for high rates
- Target 10-20ms chunks (300k-600k samples at 30 MS/s)
- Larger chunks = fewer calls = less overhead

**Expected Gain:** 1.5-2x reduction in read overhead

---

### 3. Read Timeout (MEDIUM PRIORITY)

**Current State:**
```python
# bladerf_native.py:796
timeout_ms = 250  # 250ms timeout
```

**Issues:**
- **250ms is too long** for high-rate streaming
- At 30 MS/s, 250ms = 7.5M samples queued (if timeout occurs)
- Long timeouts mask performance issues
- Should fail fast if USB is struggling

**Impact:**
- Delayed error detection
- Large sample gaps when timeouts occur
- Poor responsiveness to USB issues

**Optimization:**
- **Adaptive timeout** based on sample rate
- For 30 MS/s: timeout = 50-100ms (1.5-3M samples)
- Timeout should be ~2-3x expected read time
- Fail fast to detect issues early

**Expected Gain:** Better error detection, reduced latency

---

### 4. Ring Buffer Lock Contention (HIGH PRIORITY)

**Current State:**
```python
# ring_buffer.py:26-66
def push(self, samples):
    with self.lock:  # Thread-safe but blocking
        # ... copy operations ...
```

**Issues:**
- **Single lock** for all operations
- Lock held during memory copies (can be slow)
- At high rates, push/pop compete for lock
- Lock metrics show contention but not optimized

**Impact:**
- Lock contention causes delays
- RX task blocks waiting for lock
- FFT task blocks waiting for lock
- Reduced effective throughput

**Optimization:**
- **Lock-free ring buffer** using atomic operations
- Or: **Separate read/write locks** (readers-writer pattern)
- Or: **Double-buffering** to reduce contention
- Minimize time holding lock (pre-copy data outside lock)

**Expected Gain:** 1.5-2x reduction in lock overhead

---

### 5. Memory Allocations (MEDIUM PRIORITY)

**Current State:**
```python
# bladerf_native.py:780-783
if self._conv_buf_iq is None or len(self._conv_buf_iq) != n:
    self._conv_buf_i = np.empty(n, dtype=np.float32)
    self._conv_buf_q = np.empty(n, dtype=np.float32)
    self._conv_buf_iq = np.empty(n, dtype=np.complex64)

# scan_task.py:151
iq = self.ring.pop(self.fft_size)  # New array each time
```

**Issues:**
- **Reallocation** when chunk size changes
- **New array** from ring.pop() every FFT frame
- Multiple temporary arrays in FFT pipeline
- NumPy operations create intermediate arrays

**Impact:**
- Memory allocation overhead
- Garbage collection pressure
- Cache misses from new allocations

**Optimization:**
- **Pre-allocate buffers** for max expected chunk size
- **Reuse arrays** instead of creating new ones
- **In-place operations** where possible
- **Memory pool** for frequently allocated arrays

**Expected Gain:** 10-20% reduction in CPU overhead

---

### 6. FFT Processing Pipeline (MEDIUM PRIORITY)

**Current State:**
```python
# scan_task.py:166-170
iq = iq - np.mean(iq)  # DC removal (creates new array)
x = iq * self._win      # Windowing (creates new array)
X = np.fft.fftshift(np.fft.fft(x, n=self.fft_size))  # FFT
```

**Issues:**
- **Multiple array copies** in hot path
- DC removal creates new array
- Windowing creates new array
- FFT creates new array
- Smoothing operations create new arrays

**Impact:**
- Memory bandwidth usage
- CPU cache pressure
- Slower FFT processing

**Optimization:**
- **In-place operations** where possible
- **Pre-allocate work arrays** and reuse
- **Combine operations** to reduce passes
- **Vectorized operations** for better CPU utilization

**Expected Gain:** 15-25% faster FFT processing

---

### 7. Ring Buffer Size (LOW PRIORITY)

**Current State:**
```python
# orchestrator.py:220
ring_size = int(sample_rate_sps * 0.5)  # 0.5 seconds
```

**Issues:**
- **0.5 seconds** = 15M samples at 30 MS/s = 120MB
- Large memory footprint
- May be excessive for high rates

**Impact:**
- High memory usage
- Cache misses from large buffer
- Slower lock operations on large buffer

**Optimization:**
- **Reduce to 0.2-0.3 seconds** for high rates
- Still provides good buffering
- Reduces memory footprint

**Expected Gain:** Reduced memory usage, slightly faster operations

---

### 8. USB Transfer Optimization (HIGH PRIORITY)

**Current State:**
- No explicit USB buffer tuning
- No USB transfer size optimization
- Blocking reads (can't overlap transfers)

**Issues:**
- **Synchronous reads** block until complete
- No pipelining of USB transfers
- USB 3.0 capabilities not fully utilized
- No DMA optimization

**Impact:**
- USB bandwidth not fully utilized
- CPU waits for USB transfers
- No overlap between transfers

**Optimization:**
- **Asynchronous USB transfers** (if libbladeRF supports)
- **Larger transfer sizes** (match buffer_size)
- **USB buffer tuning** (kernel-level)
- **DMA optimization** (if available)

**Expected Gain:** 1.5-2x USB throughput

---

### 9. CPU Affinity & Threading (MEDIUM PRIORITY)

**Current State:**
- RX task runs in dedicated thread (good)
- FFT task runs in async loop (good)
- No CPU affinity set
- No thread priority set

**Issues:**
- **No CPU pinning** - threads can migrate
- **No real-time priority** - can be preempted
- Thread scheduling overhead

**Impact:**
- Thread migration causes cache misses
- Preemption causes jitter
- Reduced real-time performance

**Optimization:**
- **Pin RX thread** to specific CPU core
- **Pin FFT thread** to different core
- **Set real-time priority** for RX thread
- **CPU isolation** (isolcpus kernel parameter)

**Expected Gain:** 10-15% reduction in jitter, better real-time performance

---

### 10. NumPy FFT vs GPU FFT (LOW PRIORITY)

**Current State:**
- Uses NumPy FFT (CPU-based)
- Jetson Orin Nano has GPU available
- No GPU acceleration

**Issues:**
- **CPU FFT** is slower than GPU
- GPU is underutilized
- FFT is compute-intensive

**Impact:**
- CPU bottleneck for FFT
- Slower frame processing

**Optimization:**
- **GPU-accelerated FFT** (cuFFT or similar)
- Offload FFT to GPU
- Async GPU operations

**Expected Gain:** 2-3x faster FFT (if GPU available)

---

## Recommended Optimization Priority

### Phase 1: Quick Wins (High Impact, Low Risk)
1. **Increase read chunk size** (remove/increase max clamp)
2. **Adaptive USB buffer sizing** (based on sample rate)
3. **Reduce read timeout** (adaptive based on sample rate)

**Expected Result:** 2-3x improvement, should reach 20-25 MS/s

### Phase 2: Medium Effort (High Impact, Medium Risk)
4. **Optimize ring buffer** (lock-free or double-buffering)
5. **Reduce memory allocations** (pre-allocate, reuse)
6. **USB transfer optimization** (larger transfers, pipelining)

**Expected Result:** Additional 1.5-2x improvement, should reach 30-35 MS/s

### Phase 3: Advanced (Medium Impact, Higher Risk)
7. **CPU affinity & threading** (pinning, priorities)
8. **FFT pipeline optimization** (in-place, vectorized)
9. **GPU FFT acceleration** (if available)

**Expected Result:** Additional 1.2-1.5x improvement, should reach 35-40 MS/s

---

## Implementation Strategy

### Step 1: Measurement & Profiling
- Add performance counters (read rate, buffer usage, lock contention)
- Profile with cProfile/py-spy at different sample rates
- Measure USB throughput directly
- Identify exact bottleneck at 10 MS/s

### Step 2: Incremental Optimization
- Implement Phase 1 optimizations
- Test at 15, 20, 25 MS/s
- Measure improvement at each step
- Stop when target (30-40 MS/s) is reached

### Step 3: Validation
- Long-duration stability tests
- Memory leak checks
- CPU usage monitoring
- USB error rate monitoring

---

## Key Metrics to Monitor

1. **USB Throughput:** Actual samples/second received
2. **Buffer Underruns:** Count of empty reads
3. **Lock Contention:** Time waiting for locks
4. **CPU Usage:** Per-core utilization
5. **Memory Usage:** Peak and average
6. **Read Latency:** Time per read_samples() call
7. **FFT Latency:** Time per FFT frame

---

## Notes

- **Jetson Orin Nano** has 6 CPU cores - can dedicate cores to RX/FFT
- **USB 3.0** theoretical max is 5 Gbps, bladeRF uses ~480 Mbps at 30 MS/s
- **Memory bandwidth** may be a limiting factor (DDR5 should handle it)
- **Thermal throttling** could affect sustained performance

---

## Conclusion

The system should be capable of 30-40 MS/s with proper optimization. The main bottlenecks are:

1. **USB buffer configuration** (too small, not adaptive)
2. **Read chunk size** (too small, clamped too low)
3. **Ring buffer lock contention** (single lock bottleneck)
4. **Memory allocations** (too many temporary arrays)

Addressing these should provide the needed 3-4x performance improvement to reach the target.
