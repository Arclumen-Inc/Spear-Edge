# SPEAR-Edge Changelog

## [Unreleased] - Critical Bug Fixes & Wideband Signal Support

### Critical Bug Fixes

#### Frequency Tuning Bug (CRITICAL)
- **Issue**: SDR was not tuning to correct frequency for signals above 4.29 GHz
  - Root cause: `bladerf_set_frequency()` function signature used `ctypes.c_uint32` instead of `ctypes.c_uint64`
  - Result: Frequencies above 4.29 GHz were truncated (e.g., 5910 MHz → 1615 MHz)
  - Impact: VTX signals and other high-frequency signals were completely invisible
- **Fix**: Changed function signature to `ctypes.c_uint64` to support full frequency range
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`
- **Testing**: Verified with `scripts/verify_frequency.py` - frequency readback now matches requested value

#### USB Buffer Memory Calculation Bug
- **Issue**: Buffer size calculations incorrectly assumed 2 bytes per complex sample
  - Actual SC16_Q11 format uses 4 bytes per complex sample (2 bytes I + 2 bytes Q)
  - Result: USB buffer memory usage was 2x larger than calculated, causing `LIBUSB_ERROR_NO_MEM` errors
- **Fix**: Corrected all buffer size calculations to use 4 bytes/complex sample
- **Impact**: Reduced buffer counts to fit within 16MB USB memory limit
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`

#### Stale Tail Data in Preallocated Buffers
- **Issue**: Preallocated conversion buffers contained stale data from previous reads
  - Only prefix of buffer was written, but downstream code could access full buffer
  - Result: Wideband signals appeared smeared or inconsistent
- **Fix**: Explicitly zero unused tail of buffers before returning
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`

#### Race Condition in Gain Adjustments
- **Issue**: Rapid gain slider movements caused `malloc_consolidate()` segfaults
  - Root cause: Concurrent calls to `set_gain()` without proper thread synchronization
  - UI slider was not debounced, causing many API calls per second
- **Fix**: 
  - Added `threading.Lock()` for gain operations
  - Implemented 200ms debouncing on gain slider
  - Re-check device pointer inside lock before making libbladeRF calls
- **Files Modified**: 
  - `spear_edge/core/sdr/bladerf_native.py`
  - `spear_edge/ui/web/app.js`

#### Device Close/Reopen Race Condition
- **Issue**: Segmentation fault when switching sample rates
  - Root cause: RX task thread called `read_samples()` while device was being closed/reopened
- **Fix**: 
  - Set `_stream_active = False` at start of reconfiguration
  - Added 50ms delay after stream deactivation to allow in-flight reads to complete
  - Reordered checks in `read_samples()` to prioritize `_stream_active` flag
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`

### Wideband Signal Support Improvements

#### Adaptive Noise Floor Calculation
- **Issue**: Noise floor calculation was contaminated by wideband signal energy
  - Used 10th percentile, which included signal energy for wideband signals (20-30 MHz VTX)
  - Result: Artificially high noise floor, compressed display range, signals appeared flat
- **Fix**: Implemented adaptive noise floor calculation
  - Detects wideband signals (>20% of bins 3+ dB above preliminary 2nd percentile)
  - Uses 2nd percentile for wideband signals, 10th percentile for narrowband
  - Uses instant spectrum for detection (not averaged) for accurate classification
- **Files Modified**: `spear_edge/core/scan/scan_task.py`

#### DC Removal Made Optional
- **Issue**: Block-mean DC removal was too aggressive for wide analog FM video signals
  - Subtracting mean of entire block can distort wideband signals, especially if tuned at DC
- **Fix**: Made DC removal configurable via `SPEAR_DC_REMOVAL` environment variable (default: `false`)
  - For wideband signals, DC removal is disabled by default
  - For narrowband signals, can be enabled if needed
  - Recommendation: Tune LO off-center by 5-10 MHz instead of using DC removal for wideband
- **Files Modified**: 
  - `spear_edge/core/scan/scan_task.py`
  - `spear_edge/settings.py`

#### UI Downsampling Improvements
- **Issue**: UI downsampling used `max` method, which doesn't preserve energy spread for wideband signals
- **Fix**: Use `mean` method for wideband signals when downsampling from 65536 to 4096 points
  - Preserves total energy across frequency bins
  - Better visibility for wideband signals (20-30 MHz VTX)
- **Files Modified**: `spear_edge/ui/web/app.js`

#### Display Range Optimization
- **Issue**: Display range autoscaling was compressing wideband signals
- **Fix**: 
  - Calculate actual peak in data and ensure display range includes it with 5 dB margin
  - Fixed 35 dB display range for wideband signals (when noise floor < -75 dBFS)
  - Prevents signal clipping while maintaining stability
- **Files Modified**: `spear_edge/ui/web/app.js`

#### FFT Smoothing Control
- **Issue**: Heavy FFT smoothing (alpha=0.01) was flattening wideband signals
- **Fix**: 
  - Reduced default smoothing from 0.01 to 0.1
  - Added UI slider for runtime adjustment (0.01-1.0)
  - Added API endpoint: `POST /api/live/smoothing`
- **Files Modified**: 
  - `spear_edge/core/scan/scan_task.py`
  - `spear_edge/ui/web/app.js`
  - `spear_edge/ui/web/index.html`
  - `spear_edge/api/http/routes_tasking.py`

### Hardware Truth Logging

#### Requested vs. Actual Parameter Logging
- **Issue**: No visibility into actual hardware parameters vs. requested values
  - bladeRF may apply different values than requested (especially bandwidth)
  - Critical for wideband signals where actual bandwidth matters significantly
- **Fix**: Added comprehensive hardware truth logging
  - Logs requested vs. actual: sample rate, bandwidth, frequency, gain mode, gain
  - Format: `RX0 configured: req_sr=30.72e6 act_sr=30.72e6 req_bw=28e6 act_bw=27e6 req_fc=5800e6 act_fc=5800e6 gain_mode=MGC gain=30`
  - Stores actual values in `_actual_sample_rate`, `_actual_bandwidth`, `_actual_frequency`
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`

#### Frequency Readback Workaround
- **Issue**: Some libbladeRF versions return incorrect frequency values on readback
- **Fix**: Added workaround to detect and handle incorrect readback values
  - If readback differs by >100 MHz from requested, uses requested value
  - Logs warning when workaround is applied
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`

### IQ Scaling Lock

#### Q11 Format Locked for Debugging
- **Issue**: Configurable IQ scaling mode created ambiguity during debugging
- **Fix**: Locked scaling to Q11 format (1.0 / 2048.0) for consistency
  - Removed `SPEAR_IQ_SCALING_MODE` environment variable
  - All IQ samples now use consistent Q11 normalization
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`

### USB Buffer Configuration Refinement

#### Conservative Buffer Sizing
- **Issue**: Previous buffer configurations were too aggressive, causing USB memory exhaustion
- **Fix**: Reduced buffer counts to fit within 16MB USB memory limit
  - Very high rate (>40 MS/s): 4 buffers × 256K samples × 4 bytes = 4MB
  - High rate (20-40 MS/s): 8 buffers × 128K samples × 4 bytes = 4MB
  - Medium rate (10-20 MS/s): 16 buffers × 64K samples × 4 bytes = 4MB
  - Low rate (<10 MS/s): 32 buffers × 32K samples × 4 bytes = 4MB
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`

#### Device Close/Reopen for Stream Reconfiguration
- **Issue**: `bladerf_sync_config()` does not free old buffers on reconfiguration, causing memory leak
- **Fix**: Implemented device close/reopen when reconfiguring active stream
  - Saves current RF state (frequency, sample rate, bandwidth, gain)
  - Closes device (frees all USB buffers)
  - Reopens device
  - Restores RF parameters
  - Proceeds with new stream configuration
- **Files Modified**: `spear_edge/core/sdr/bladerf_native.py`

### UI Improvements

#### RX Port Display
- Added display of active RX channel in SDR Controls panel
- Shows "RX0" or "RX1" based on configuration
- **Files Modified**: 
  - `spear_edge/ui/web/app.js`
  - `spear_edge/ui/web/index.html`

#### Enhanced WebSocket Logging
- Added detailed logging for received FFT frames
- Logs: frame size, data range, mean, noise floor, downsampling method
- Helps diagnose display issues
- **Files Modified**: `spear_edge/ui/web/app.js`

### Diagnostic Tools

#### New Diagnostic Scripts
- `scripts/verify_frequency.py`: Verifies frequency tuning accuracy
- `scripts/test_sdr_signal.py`: Tests SDR configuration and signal detection
- `scripts/diagnose_vtx.py`: Direct bladeRF hardware testing for VTX signals
- `scripts/diagnose_vtx_edge.py`: Matches Edge's exact configuration for VTX diagnosis
- `scripts/diagnose_display_issue.py`: Compares raw data with Edge's processed output

### Lessons Learned

#### Debugging Process
1. **Hardware Truth First**: Always verify actual hardware parameters match requested values
2. **Buffer Math Matters**: Incorrect buffer size calculations can cause subtle but critical issues
3. **Wideband vs. Narrowband**: Different signal types require different processing approaches
4. **Type Safety**: `ctypes` function signatures must match C header definitions exactly
5. **Race Conditions**: Thread synchronization is critical for hardware device access

#### Root Cause Analysis
- The frequency bug (uint32 → uint64) was the primary blocker for VTX visibility
- However, the debugging process revealed multiple real issues:
  - Buffer size calculation errors
  - Stale tail data contamination
  - Race conditions in gain adjustments
  - Noise floor calculation issues for wideband signals
- All fixes remain valuable and improve overall system reliability

### Backward Compatibility
- All changes are backward compatible
- No API changes
- No breaking changes to existing functionality
- DC removal can be enabled via environment variable if needed

### Testing Recommendations
- Test with known signals at various frequencies (especially >4.29 GHz)
- Verify frequency readback matches requested value
- Test wideband signals (20-30 MHz) with noise floor calculation
- Test gain adjustments with rapid slider movements
- Monitor USB memory usage at high sample rates

---

## [Unreleased] - Performance Optimization & ML Dashboard Release

### New Features

#### ML Dashboard
- **New ML Management Interface**: Dedicated web UI for ML operations
  - Accessible at `/ml` or via "ML Dashboard" link in main UI
  - Capture management with grid view and thumbnails
  - Label editing (single and batch)
  - Model export/import functionality
  - Model testing on captures
  - Statistics dashboard

- **Capture Label Management**:
  - View all captures with thumbnails
  - Filter by label, source, search
  - Edit labels inline or batch update
  - Delete captures (single or batch)
  - Label validation against class_labels.json

- **Model Management**:
  - Export current model as ZIP (includes model + labels + metadata)
  - Import new models from ZIP
  - View available models
  - Test model on specific captures
  - Current model information display

- **Statistics**:
  - Total captures count
  - Labeled/unlabeled counts
  - Label distribution chart
  - Auto-refresh every 30 seconds

- **Quick Training**:
  - Fine-tune model on 1-2 captures (5-15 min)
  - Real-time progress tracking
  - Only trains classification head (fast)
  - Suitable for on-device training on Jetson
  - Automatic model saving

#### ML API Endpoints
- `GET /api/ml/captures` - List captures with filtering
- `GET /api/ml/captures/{capture_dir}` - Get capture details
- `GET /api/ml/captures/{capture_dir}/thumbnail` - Get thumbnail
- `POST /api/ml/captures/{capture_dir}/label` - Update label
- `POST /api/ml/captures/batch-label` - Batch update labels
- `POST /api/ml/captures/{capture_dir}/delete` - Delete capture
- `POST /api/ml/captures/batch-delete` - Batch delete
- `GET /api/ml/models` - List models
- `GET /api/ml/models/current` - Get current model info
- `POST /api/ml/models/export` - Export model
- `POST /api/ml/models/import` - Import model
- `POST /api/ml/models/test` - Test model
- `GET /api/ml/class-labels` - Get class labels
- `GET /api/ml/stats` - Get statistics
- `POST /api/ml/train/quick` - Start quick training (fine-tuning)
- `GET /api/ml/train/status/{job_id}` - Get training status
- `POST /api/ml/train/cancel/{job_id}` - Cancel training

## [Previous] - Performance Optimization Release

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
