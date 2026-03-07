# Capture Modes Testing Guide

This guide explains how to test manual and armed (cued) capture modes.

## Quick Start

### Prerequisites
1. SPEAR-Edge server must be running on `http://localhost:8000`
2. Python `requests` library installed: `pip3 install requests`
3. SDR hardware connected (bladeRF) or MockSDR for testing

### Run All Tests
```bash
python3 test_capture_modes.py --all
```

### Run Individual Tests
```bash
# Manual capture only
python3 test_capture_modes.py --manual

# Armed capture only
python3 test_capture_modes.py --armed
```

## Test Details

### Test 1: Manual Capture
- **Endpoint**: `POST /api/capture/start`
- **Purpose**: Verify manual capture works with operator-specified parameters
- **Parameters**:
  - Frequency: 915 MHz (default)
  - Sample Rate: 10 MS/s (from SDR config or default)
  - Duration: 2 seconds
  - Gain Mode: manual
  - Gain: 50 dB
- **Expected Result**: Capture queued and executed successfully

### Test 2: Armed Capture (Tripwire Cued)
- **Endpoint**: `POST /api/tripwire/event`
- **Purpose**: Verify armed mode auto-capture works with Tripwire events
- **Requirements**:
  - Edge mode must be "armed"
  - Event must be actionable (not advisory-only)
  - Event must pass policy checks (confidence >= 0.90, stage="confirmed")
- **Test Event**:
  - Type: `fhss_cluster` (v2.0 actionable)
  - Stage: `confirmed` (v1.1 compatibility)
  - Confidence: 0.95 (above threshold)
  - Frequency: 915 MHz
- **Expected Result**: Capture triggered automatically

### Test 3: Armed Capture Rejection (Advisory Event)
- **Purpose**: Verify advisory-only events are correctly rejected
- **Test Event**:
  - Type: `rf_cue` (advisory only, never actionable)
  - Stage: `cue` (advisory stage)
  - Confidence: 0.95
- **Expected Result**: Event rejected with reason "cue_only" or queued for operator

## Manual Testing via API

### 1. Manual Capture

```bash
curl -X POST http://localhost:8000/api/capture/start \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "manual",
    "freq_hz": 915000000,
    "sample_rate_sps": 10000000,
    "bandwidth_hz": 8000000,
    "gain_mode": "manual",
    "gain_db": 50.0,
    "rx_channel": 0,
    "duration_s": 2.0
  }'
```

**Expected Response**:
```json
{
  "accepted": true,
  "action": "capture_started"
}
```

### 2. Armed Capture (Set Mode First)

```bash
# Set mode to armed
curl -X POST http://localhost:8000/api/live/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "armed"}'

# Send actionable Tripwire event
curl -X POST http://localhost:8000/api/tripwire/event \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "tripwire.event.v2.0",
    "type": "fhss_cluster",
    "stage": "confirmed",
    "node_id": "test_node_001",
    "freq_hz": 915000000,
    "bandwidth_hz": 5000000,
    "confidence": 0.95,
    "scan_plan": "targeted",
    "timestamp": 1234567890.0
  }'
```

**Expected Response** (if policy allows):
```json
{
  "accepted": true,
  "action": "auto_capture_started"
}
```

**Expected Response** (if rejected):
```json
{
  "accepted": true,
  "action": "rejected",
  "reason": "global_cooldown" | "node_cooldown" | "freq_cooldown" | "rate_limited" | "low_confidence" | "cue_only"
}
```

### 3. Advisory Event (Should Not Trigger Capture)

```bash
curl -X POST http://localhost:8000/api/tripwire/event \
  -H "Content-Type: application/json" \
  -d '{
    "type": "rf_cue",
    "stage": "cue",
    "node_id": "test_node_001",
    "freq_hz": 915000000,
    "confidence": 0.95,
    "timestamp": 1234567890.0
  }'
```

**Expected Response**:
```json
{
  "accepted": true,
  "action": "rejected",
  "reason": "cue_only"
}
```

## Verification

### Check Capture Status
```bash
# List recent captures
curl http://localhost:8000/api/live/captures?limit=10
```

### Check Capture Artifacts
Captures are stored in:
```
/home/spear/spear-edgev1_0/data/artifacts/captures/
```

Each capture directory contains:
- `iq/samples.iq`: Raw IQ data
- `iq/samples.sigmf-meta`: SigMF metadata
- `features/spectrogram.npy`: ML-ready spectrogram
- `features/stats.json`: RF statistics
- `thumbnails/spectrogram.png`: Visual spectrogram
- `capture.json`: Complete metadata

### Check Server Logs
Monitor server logs for capture execution:
```bash
# Look for capture-related log messages
tail -f /path/to/server.log | grep -i capture
```

Key log prefixes:
- `[CAPTURE]`: Capture execution
- `[CAPTURE ROUTE]`: Manual capture API
- `[INGEST]`: Tripwire event ingestion
- `[CAPTURE MGR]`: Capture manager operations

## Troubleshooting

### Manual Capture Fails
1. **Check SDR connection**: Verify bladeRF is connected
2. **Check mode**: Ensure not in "tasked" mode
3. **Check queue**: Queue may be full (max 8 captures)
4. **Check logs**: Look for error messages in server logs

### Armed Capture Not Triggering
1. **Check mode**: Must be "armed" (not "manual")
2. **Check event type**: Must be actionable (not `rf_cue`, `heartbeat`, etc.)
3. **Check stage**: Must be "confirmed" (v1.1) or actionable type (v2.0)
4. **Check confidence**: Must be >= 0.90
5. **Check cooldowns**: Wait for cooldown periods:
   - Global: 3.0 seconds
   - Per-node: 2.0 seconds
   - Per-frequency: 8.0 seconds per 100 kHz bin
6. **Check rate limit**: Max 10 captures per minute
7. **Check scan plan**: Must not be "survey_wide" or "wifi_bt_24g"

### Common Issues

**"queue_full"**:
- Capture queue is full (max 8 captures)
- Wait for existing captures to complete

**"global_cooldown"**:
- Too soon after previous capture
- Wait 3+ seconds

**"node_cooldown"**:
- Same node triggered capture recently
- Wait 2+ seconds

**"freq_cooldown"**:
- Same frequency bin captured recently
- Wait 8+ seconds

**"low_confidence"**:
- Confidence < 0.90
- Increase confidence in test event

**"cue_only"**:
- Event type is advisory-only (`rf_cue`)
- Use actionable event type (`fhss_cluster`, `confirmed_event`)

## Test Script Options

```bash
# Test with custom URL
python3 test_capture_modes.py --all --url http://192.168.1.100:8000

# Test manual only
python3 test_capture_modes.py --manual

# Test armed only
python3 test_capture_modes.py --armed
```

## Expected Test Output

```
======================================================================
CAPTURE MODES TEST SUITE
======================================================================
✅ Server is reachable

======================================================================
TEST 1: Manual Capture
======================================================================

📡 Capture Parameters:
   Frequency: 915.000 MHz
   Sample Rate: 10.00 MS/s
   Duration: 2s
   Gain Mode: manual
   Gain: 50 dB

📤 Sending manual capture request...
✅ Response: {
  "accepted": true,
  "action": "capture_started"
}
✅ Manual capture queued successfully!
⏳ Waiting 4.0s for capture to complete...
✅ Manual capture test completed

======================================================================
TEST 2: Armed Capture (Tripwire Cued)
======================================================================

📡 Event Parameters:
   Frequency: 915.000 MHz
   Type: fhss_cluster (v2.0 actionable event)
   Stage: confirmed (v1.1 format)
   Confidence: 0.95 (above 0.90 threshold)

📤 Sending Tripwire event (should trigger capture)...
✅ Response: {
  "accepted": true,
  "action": "auto_capture_started"
}
✅ Armed capture triggered successfully!
⏳ Waiting 7s for capture to complete...
✅ Armed capture test completed

======================================================================
TEST SUMMARY
======================================================================
✅ PASS: Manual Capture
✅ PASS: Armed Capture Rejection
✅ PASS: Armed Capture
======================================================================
✅ All tests passed!
```

## Next Steps

After successful testing:
1. Verify capture artifacts are created correctly
2. Check capture.json metadata is complete
3. Verify spectrogram generation
4. Test with real Tripwire nodes (if available)
5. Monitor capture queue and cooldown behavior
