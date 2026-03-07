# SPEAR-Edge Optimization and Cleanup Report

**Date:** 2025-01-XX  
**Scope:** Complete codebase analysis for optimizations, dead code, and outdated information

---

## Executive Summary

This report documents findings from a comprehensive analysis of the SPEAR-Edge codebase. The analysis identified:
- **3 instances of dead code** (unused methods/functions)
- **2 duplicate endpoints** (redundant WebSocket routes)
- **5+ optimization opportunities** (performance improvements)
- **10+ outdated documentation references** (needs updates)
- **1 TODO item** that should be addressed or removed

---

## 1. Dead Code

### 1.1 Unused Method: `_capture_iq()` in `capture_manager.py`

**Location:** `spear_edge/core/capture/capture_manager.py:729-812`

**Issue:** The method `_capture_iq()` is defined but never called. The codebase uses `_capture_iq_to_disk()` instead, which writes directly to disk (memory-efficient).

**Evidence:**
- `_capture_iq()` loads entire capture into memory as numpy array
- `_capture_iq_to_disk()` writes directly to disk (used in `_execute()`)
- No references to `_capture_iq()` found in codebase

**Recommendation:** 
- **Remove** `_capture_iq()` method (lines 729-812)
- Keep `_capture_iq_to_disk()` as it's the production path

**Impact:** Low risk - method is unused, removal will reduce code complexity

---

### 1.2 Unused Method: `_write_operator_spectrogram()` in `capture_manager.py`

**Location:** `spear_edge/core/capture/capture_manager.py:951-993`

**Issue:** Method is defined but never called. The codebase uses `save_spectrogram_thumbnail()` from `spectrogram.py` instead.

**Evidence:**
- Method exists but no references found
- `_execute()` uses `save_spectrogram_thumbnail()` from `spectrogram.py`
- Similar functionality already implemented elsewhere

**Recommendation:**
- **Remove** `_write_operator_spectrogram()` method (lines 951-993)
- Functionality is already provided by `spectrogram.py`

**Impact:** Low risk - redundant implementation

---

### 1.3 Unused Method: `_write_spectrogram_pgm()` in `capture_manager.py`

**Location:** `spear_edge/core/capture/capture_manager.py:1694-1717`

**Issue:** Method is defined but never called. PNG generation is handled by `spectrogram.py`.

**Evidence:**
- No references found in codebase
- PGM format is legacy/fallback, not used in production

**Recommendation:**
- **Remove** `_write_spectrogram_pgm()` method (lines 1694-1717)
- If PGM support is needed, it should be in `spectrogram.py`

**Impact:** Low risk - unused legacy code

---

## 2. Duplicate Code / Redundant Implementations

### 2.1 Duplicate WebSocket Endpoints

**Location:** `spear_edge/app.py:115-121`

**Issue:** Two WebSocket endpoints (`/ws/tripwire` and `/ws/tripwire_link`) both call the same handler function.

```python
@app.websocket("/ws/tripwire")
async def ws_tripwire(websocket: WebSocket):
    await tripwire_link_ws(websocket, app.state.orchestrator)

@app.websocket("/ws/tripwire_link")
async def ws_tripwire_link(websocket: WebSocket):
    await tripwire_link_ws(websocket, app.state.orchestrator)
```

**Recommendation:**
- **Keep** `/ws/tripwire_link` (more descriptive name)
- **Remove** `/ws/tripwire` endpoint
- Update any frontend code that uses `/ws/tripwire` to use `/ws/tripwire_link`

**Impact:** Medium - need to verify frontend usage before removal

**Action Required:** Check `app.js` for WebSocket connections to `/ws/tripwire`

---

### 2.2 Duplicate Capture IQ Methods

**Location:** `spear_edge/core/capture/capture_manager.py`

**Issue:** Two methods with similar functionality:
- `_capture_iq()` - loads into memory (unused)
- `_capture_iq_to_disk()` - writes to disk (used)

**Status:** Already identified in Section 1.1 - remove `_capture_iq()`

---

## 3. Optimization Opportunities

### 3.1 Memory Optimization: Pre-allocated Buffers

**Location:** `spear_edge/core/sdr/bladerf_native.py:212-215`

**Status:** ✅ **Already Optimized** - Pre-allocated conversion buffers are reused

**Current Implementation:**
```python
self._conv_buf_i: Optional[np.ndarray] = None
self._conv_buf_q: Optional[np.ndarray] = None
self._conv_buf_iq: Optional[np.ndarray] = None
```

**Recommendation:** Keep as-is, this is good practice

---

### 3.2 FFT Processing: Reduce Diagnostic Logging Frequency

**Location:** `spear_edge/core/scan/scan_task.py:236`

**Issue:** Diagnostic logging every 150 frames (~10 seconds at 15 fps) may be too frequent for production.

**Current:**
```python
if frame_count % 150 == 0:  # Every 150 frames (~10 seconds at 15 fps)
    # ... extensive diagnostic logging ...
```

**Recommendation:**
- Reduce frequency to every 300-500 frames (20-33 seconds)
- Or make it configurable via environment variable
- Or disable by default, enable with `SPEAR_DEBUG_FFT=1`

**Impact:** Low - reduces log verbosity in production

---

### 3.3 SDR Health Logging: Reduce Frequency

**Location:** `spear_edge/core/sdr/bladerf_native.py:883`

**Issue:** Health diagnostics logged every 200 reads may be too frequent.

**Current:**
```python
if self._health_stats["successful_reads"] % 200 == 0:  # Every 200 reads
    # ... extensive logging ...
```

**Recommendation:**
- Increase to every 1000 reads (less frequent)
- Or make conditional on actual issues (only log when clipping detected)

**Impact:** Low - reduces log verbosity

---

### 3.4 Ring Buffer: Lock Contention Optimization

**Location:** `spear_edge/core/scan/ring_buffer.py`

**Issue:** Lock metrics are collected but may not be used effectively.

**Recommendation:**
- Review if lock contention metrics are actually used
- If not used, remove metrics collection to reduce overhead
- If used, ensure they're being monitored/alerted on

**Action Required:** Check if `get_lock_metrics()` is called anywhere

---

### 3.5 Capture Manager: Reduce Print Statements

**Location:** `spear_edge/core/capture/capture_manager.py` (throughout)

**Issue:** Extensive use of `print()` statements instead of proper logging.

**Recommendation:**
- Migrate to Python `logging` module
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Allow log level configuration via environment variable

**Impact:** Medium - improves debuggability and production logging

**Priority:** Should-fix (already identified in ASSESSMENT.md)

---

### 3.6 Frontend: Reduce Console Logging

**Location:** `spear_edge/ui/web/app.js`

**Issue:** Extensive `console.log()` statements in production code.

**Recommendation:**
- Use conditional logging: `if (DEBUG) console.log(...)`
- Or use a logging utility that respects log levels
- Reduce frequency of frame key logging (currently every 2 seconds)

**Impact:** Low - improves browser console performance

---

## 4. Outdated Information

### 4.1 Documentation: Edge Humps Investigation

**Location:** `docs/EDGE_HUMPS_NOISE_FLOOR_INVESTIGATION.md`

**Issue:** Document describes "Potential Fixes (Not Implemented Yet)" but some fixes have been implemented.

**Status Check Required:**
- Fix 1 (Use Backend Noise Floor): ✅ **IMPLEMENTED** - Backend sends `noise_floor_dbfs`
- Fix 2 (Exclude Edge Bins): ✅ **IMPLEMENTED** - Edge bins excluded in `scan_task.py:191-198`
- Fix 3 (Zero Edge Bins): ✅ **IMPLEMENTED** - Edge bins zeroed in `scan_task.py:222-233`

**Recommendation:**
- Update document to reflect implemented fixes
- Mark document as "Historical" or "Partially Implemented"
- Create new document for remaining unimplemented fixes

---

### 4.2 Documentation: Option A Implementation

**Location:** `docs/OPTION_A_IMPLEMENTATION.md`

**Issue:** Document describes implementation phases, but current codebase may have diverged.

**Status Check Required:**
- Verify Phase 1 (Q11 scaling) is still accurate
- Verify Phase 2 (FFT math) matches current implementation
- Verify Phase 3 (Calibration metadata) is still used

**Recommendation:**
- Review and update document to match current implementation
- Or mark as "Historical Reference"

---

### 4.3 Documentation: LIBBLADERF Migration

**Location:** `docs/LIBBLADERF_MIGRATION_SUMMARY.md`

**Status:** ✅ **Current** - Migration is complete, document is accurate

**Recommendation:** Keep as-is, good reference document

---

### 4.4 Documentation: Changelog Duplication

**Location:** 
- `docs/CHANGELOG.txt`
- `technical_data_package/CHANGELOG.txt`

**Issue:** Two identical changelog files exist.

**Recommendation:**
- Consolidate into single changelog in `docs/`
- Remove duplicate from `technical_data_package/`
- Or clearly mark one as "source of truth"

---

### 4.5 Code Comments: Outdated References

**Location:** `spear_edge/core/orchestrator/orchestrator.py:641`

**Issue:** TODO comment for `recent_tripwire_hit` implementation.

```python
# TODO: Implement recent_tripwire_hit if needed
```

**Recommendation:**
- Either implement the feature
- Or remove TODO if not needed
- Or document why it's deferred

**Impact:** Low - just cleanup

---

### 4.6 Settings: Unused Configuration Options

**Location:** `spear_edge/settings.py`

**Issue:** `CALIBRATION_OFFSET_DB` and `IQ_SCALING_MODE` may not be actively used.

**Status Check Required:**
- Verify `CALIBRATION_OFFSET_DB` is used (check `scan_task.py`)
- Verify `IQ_SCALING_MODE` is used (check `bladerf_native.py`)

**Current Usage:**
- `IQ_SCALING_MODE`: ✅ Used in `bladerf_native.py:854-876`
- `CALIBRATION_OFFSET_DB`: ⚠️ Set to 0.0 in `scan_task.py:246` (hardcoded, not using settings)

**Recommendation:**
- Fix `scan_task.py` to use `settings.CALIBRATION_OFFSET_DB` instead of hardcoded 0.0
- Or remove unused setting if calibration is not needed

---

## 5. Code Quality Issues

### 5.1 Error Handling: Inconsistent Exception Handling

**Location:** Throughout codebase

**Issue:** Mix of `print()` statements and proper exception handling.

**Recommendation:**
- Standardize on Python `logging` module
- Use structured exception handling
- Add error recovery where appropriate

**Priority:** Should-fix (already identified in ASSESSMENT.md)

---

### 5.2 Type Hints: Incomplete Type Annotations

**Location:** Throughout codebase

**Issue:** Some functions lack type hints, especially in older code.

**Recommendation:**
- Add type hints to public APIs
- Use `mypy` for type checking (optional, but recommended)

**Priority:** Nice-to-have

---

### 5.3 Import Organization: Unused Imports

**Location:** Various files

**Issue:** Some files may have unused imports.

**Recommendation:**
- Run `pylint` or `flake8` to identify unused imports
- Remove unused imports
- Use `isort` to organize imports

**Priority:** Nice-to-have

---

## 6. Performance Optimizations

### 6.1 FFT Processing: Reduce Array Copies

**Location:** `spear_edge/core/scan/scan_task.py`

**Status:** ✅ **Already Optimized** - Uses in-place operations where possible

**Current:**
- Pre-allocated window (`self._win`)
- In-place operations for DC removal
- Reuses arrays where possible

**Recommendation:** Keep as-is

---

### 6.2 WebSocket: Reduce JSON Serialization Overhead

**Location:** `spear_edge/api/ws/live_fft_ws.py`

**Issue:** Large arrays are serialized to JSON every frame.

**Current Optimization:**
- `freqs_hz` is not sent (client computes) ✅
- Only power arrays are sent

**Recommendation:**
- Consider binary WebSocket protocol for high-frequency updates
- Or use compression (gzip) for JSON payloads
- Current approach is acceptable for 15 fps

**Priority:** Nice-to-have (only if performance issues arise)

---

### 6.3 Capture: Memory-Efficient Processing

**Location:** `spear_edge/core/capture/capture_manager.py`

**Status:** ✅ **Already Optimized** - Uses chunked processing for large captures

**Current:**
- `_capture_iq_to_disk()` writes directly to disk
- `compute_spectrogram_chunked()` processes in chunks
- Memory cleanup with `gc.collect()`

**Recommendation:** Keep as-is, excellent implementation

---

## 7. Recommendations Summary

### High Priority (Should Fix)

1. **Remove dead code:**
   - `_capture_iq()` method
   - `_write_operator_spectrogram()` method
   - `_write_spectrogram_pgm()` method

2. **Remove duplicate WebSocket endpoint:**
   - Remove `/ws/tripwire`, keep `/ws/tripwire_link`
   - Update frontend if needed

3. **Fix calibration offset usage:**
   - Use `settings.CALIBRATION_OFFSET_DB` in `scan_task.py` instead of hardcoded 0.0

4. **Migrate to proper logging:**
   - Replace `print()` with `logging` module
   - Add log level configuration

### Medium Priority (Nice to Have)

1. **Update outdated documentation:**
   - Mark `EDGE_HUMPS_NOISE_FLOOR_INVESTIGATION.md` as historical
   - Update `OPTION_A_IMPLEMENTATION.md` to match current code
   - Consolidate duplicate changelogs

2. **Reduce diagnostic logging frequency:**
   - Make FFT diagnostics configurable
   - Reduce SDR health logging frequency

3. **Address TODO comment:**
   - Implement or remove `recent_tripwire_hit` TODO

### Low Priority (Future Work)

1. **Code quality improvements:**
   - Add type hints to public APIs
   - Remove unused imports
   - Standardize error handling

2. **Performance optimizations:**
   - Consider binary WebSocket protocol
   - Review lock contention metrics usage

---

## 8. Files to Modify

### Dead Code Removal

1. `spear_edge/core/capture/capture_manager.py`
   - Remove `_capture_iq()` (lines 729-812)
   - Remove `_write_operator_spectrogram()` (lines 951-993)
   - Remove `_write_spectrogram_pgm()` (lines 1694-1717)

### Duplicate Endpoint Removal

2. `spear_edge/app.py`
   - Remove `/ws/tripwire` endpoint (lines 115-117)
   - Keep `/ws/tripwire_link` endpoint

### Configuration Fix

3. `spear_edge/core/scan/scan_task.py`
   - Use `settings.CALIBRATION_OFFSET_DB` instead of hardcoded 0.0 (line 246)

### Documentation Updates

4. `docs/EDGE_HUMPS_NOISE_FLOOR_INVESTIGATION.md`
   - Mark fixes as implemented
   - Add "Historical" note

5. `docs/OPTION_A_IMPLEMENTATION.md`
   - Verify and update to match current implementation

6. `technical_data_package/CHANGELOG.txt`
   - Remove or mark as duplicate of `docs/CHANGELOG.txt`

---

## 9. Testing Recommendations

After making changes, verify:

1. **Capture functionality:**
   - Manual captures work correctly
   - Armed captures work correctly
   - Spectrogram generation works

2. **WebSocket connections:**
   - Frontend connects to `/ws/tripwire_link` (not `/ws/tripwire`)
   - Live FFT updates work
   - Event notifications work

3. **SDR operations:**
   - FFT processing works correctly
   - Calibration offset (if changed) works as expected

---

## 10. Conclusion

The SPEAR-Edge codebase is generally well-structured with good optimization practices. The main issues are:

1. **Dead code** that can be safely removed
2. **Duplicate endpoints** that should be consolidated
3. **Outdated documentation** that needs updates
4. **Logging** that should be migrated to proper logging module

Most optimizations are already in place (memory-efficient captures, pre-allocated buffers, chunked processing). The recommended changes are primarily cleanup and code quality improvements.

**Estimated Effort:**
- Dead code removal: 1-2 hours
- Duplicate endpoint removal: 1 hour (including frontend check)
- Documentation updates: 2-3 hours
- Logging migration: 4-6 hours (larger task)

**Total:** ~8-12 hours for all recommended changes

---

**Report Generated:** 2025-01-XX  
**Next Review:** After implementing recommended changes
