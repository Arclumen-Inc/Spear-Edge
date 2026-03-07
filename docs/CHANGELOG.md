# SPEAR-Edge Changelog

## [Unreleased] - Performance Optimization Release

### Performance Improvements

#### High Sample Rate Support (30-40 MS/s)
- **Adaptive USB Buffer Sizing**: USB buffers now scale with sample rate
  - Low rates (<10 MS/s): 128KB buffers, 64 buffers, 16 transfers
  - Medium rates (10-20 MS/s): 256KB buffers, 96 buffers, 24 transfers
  - High rates (>20 MS/s): 512KB buffers, 128 buffers, 32 transfers
  - Reduces USB overhead and improves throughput at high rates

- **Adaptive Read Chunk Sizing**: Read chunks now scale with sample rate
  - Low rates: 20ms chunks (clamped to 32K-262K samples)
  - High rates (>20 MS/s): 15ms chunks, up to 1M samples
  - Reduces read call overhead and improves efficiency

- **Adaptive Read Timeout**: Timeout now scales with expected read time
  - Calculated as 3x expected read time
  - Clamped to 50-200ms range
  - Fails fast if USB is struggling, improving error detection

- **Optimized Ring Buffer**: Reduced lock contention
  - Pre-convert samples outside lock to reduce lock hold time
  - Optimized memory copy operations
  - Pre-allocate output arrays for wraparound cases
  - Reduces contention between RX and FFT threads

- **Reduced Memory Allocations**: Pre-allocated work arrays
  - FFT pipeline uses pre-allocated work arrays
  - Conversion buffers pre-allocated for max expected size (1M samples)
  - In-place operations where possible
  - Reduces GC pressure and improves cache locality

- **Optimized Ring Buffer Size**: Adaptive based on sample rate
  - High rates (>20 MS/s): 0.3 seconds (reduced from 0.5)
  - Low rates: 0.5 seconds (unchanged)
  - Reduces memory footprint while maintaining buffering

### Technical Details

#### Files Modified
- `spear_edge/core/orchestrator/orchestrator.py`: Adaptive chunk sizing, ring buffer sizing
- `spear_edge/core/sdr/bladerf_native.py`: Adaptive USB buffers, adaptive timeout, buffer pre-allocation
- `spear_edge/core/scan/rx_task.py`: Improved chunk size handling
- `spear_edge/core/scan/ring_buffer.py`: Lock contention optimization
- `spear_edge/core/scan/scan_task.py`: Pre-allocated work arrays, in-place operations

#### Expected Performance
- **Before**: ~10 MS/s maximum
- **After Phase 1**: 20-25 MS/s expected
- **After Phase 2**: 30-35 MS/s expected
- **Target**: 30-40 MS/s (achievable with hardware)

### Backward Compatibility
- All changes are backward compatible
- Low sample rates (<10 MS/s) use same configuration as before
- No API changes
- No breaking changes to existing functionality

### Notes
- Performance improvements are most noticeable at sample rates >10 MS/s
- USB 3.0 connection required for high rates
- Jetson Orin Nano should handle 30-40 MS/s with these optimizations
- Monitor CPU usage and USB health metrics at high rates

---

## Previous Releases

(Historical changelog entries would go here)
