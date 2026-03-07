# SPEAR-Edge API Reference

## Base URL

All HTTP endpoints are prefixed with `/api` unless otherwise noted. 

**Main Application**: The default server runs on `http://localhost:8080`.

**Ingest Service**: A separate ingest service runs on `http://localhost:8000` for Tripwire event ingestion (see [Ingest Service](INGEST_SERVICE.md) for details).

## Authentication

Currently, no authentication is required. The system is designed for local network use only.

## HTTP Endpoints

### Health and Status

#### `GET /health`
Check server health.

**Response:**
```json
{
  "ok": true
}
```

#### `GET /health/status`
Get system status.

**Response:**
```json
{
  "ok": true,
  "status": {...}
}
```

#### `GET /health/sdr`
Get SDR health metrics.

**Response:**
```json
{
  "status": "active",
  "success_rate_pct": 99.5,
  "throughput_mbps": 15.2,
  "samples_per_sec": 2400000,
  "avg_read_time_ms": 2.1,
  "errors": 0,
  "timeouts": 0,
  "reads": {
    "total": 1000,
    "successful": 995
  },
  "stream": "active",
  "usb_speed": "SuperSpeed"
}
```

### Live Scan Control

#### `POST /live/start`
Start or restart live FFT scanning.

**Request Body:**
```json
{
  "center_freq_hz": 915000000,
  "sample_rate_sps": 2400000,
  "fft_size": 2048,
  "fps": 15.0
}
```

**Response:**
```json
{
  "ok": true,
  "status": {...}
}
```

#### `POST /live/stop`
Stop live FFT scanning.

**Response:**
```json
{
  "ok": true,
  "status": {...}
}
```

### Mode Control

#### `GET /live/mode`
Get current operator mode.

**Response:**
```json
{
  "mode": "manual"
}
```

**Modes:**
- `manual`: Operator control only, no automatic captures
- `armed`: Automatic captures enabled (with policy guards)

#### `POST /live/mode/set`
Set operator mode.

**Request Body:**
```json
{
  "mode": "armed"
}
```

**Response:**
```json
{
  "ok": true,
  "mode": "armed"
}
```

### Auto-Capture Policy

#### `GET /live/auto/policy`
Get auto-capture policy settings.

**Response:**
```json
{
  "enabled": true,
  "min_confidence": 0.90,
  "global_cooldown_s": 3.0,
  "per_node_cooldown_s": 2.0,
  "per_freq_cooldown_s": 8.0,
  "freq_bin_hz": 100000,
  "max_captures_per_min": 10
}
```

#### `POST /live/auto/policy`
Update auto-capture policy.

**Request Body:**
```json
{
  "enabled": true,
  "min_confidence": 0.90,
  "global_cooldown_s": 3.0,
  "per_node_cooldown_s": 2.0,
  "per_freq_cooldown_s": 8.0,
  "freq_bin_hz": 100000,
  "max_captures_per_min": 10
}
```

### SDR Configuration

#### `POST /live/sdr/config`
Configure SDR parameters.

**Request Body:**
```json
{
  "center_freq_hz": 915000000,
  "sample_rate_sps": 2400000,
  "bandwidth_hz": 2400000,
  "gain_mode": "manual",
  "gain_db": 10.0,
  "rx_channel": 0,
  "bt200_enabled": false,
  "dual_channel": false
}
```

**Response:**
```json
{
  "ok": true,
  "config": {...}
}
```

#### `GET /live/sdr/info`
Get SDR information and capabilities.

**Response:**
```json
{
  "driver": "bladerf_native",
  "label": "bladeRF 2.0 micro",
  "rx_channels": 2,
  "supports_agc": true,
  "current_config": {...}
}
```

### Capture Management

#### `POST /api/capture/start`
Start a manual capture.

**Request Body:**
```json
{
  "reason": "manual",
  "freq_hz": 915000000,
  "sample_rate_sps": 10000000,
  "bandwidth_hz": 10000000,
  "gain_mode": "manual",
  "gain_db": 10.0,
  "rx_channel": 0,
  "duration_s": 5.0,
  "source_node": null,
  "scan_plan": null,
  "classification": null
}
```

**Response:**
```json
{
  "accepted": true,
  "action": "capture_started"
}
```

#### `POST /api/capture/label`
Update classification label for a capture.

**Request Body:**
```json
{
  "capture_dir": "20260307_043820_915000000Hz_...",
  "label": "elrs"
}
```

**Response:**
```json
{
  "ok": true,
  "label": "elrs",
  "capture_dir": "20260307_043820_915000000Hz_..."
}
```

#### `GET /live/captures`
Get list of recent captures.

**Response:**
```json
{
  "captures": [
    {
      "ts": 1709856000.0,
      "freq_hz": 915000000,
      "duration_s": 5.0,
      "iq_path": "/path/to/capture.iq",
      "meta": {...}
    }
  ]
}
```

### Tripwire Integration

#### `POST /api/tripwire/event`
Receive Tripwire event (primary endpoint).

**Request Body:**
```json
{
  "type": "rf_cue",
  "stage": "confirmed",
  "node_id": "tripwire-001",
  "freq_hz": 915000000,
  "bandwidth_hz": 2000000,
  "delta_db": 15.5,
  "level_db": -45.2,
  "classification": "elrs",
  "confidence": 0.95,
  "scan_plan": "915mhz_scan",
  "timestamp": 1709856000.0
}
```

**Response:**
```json
{
  "accepted": true,
  "action": "auto_capture_started"
}
```

**Event Types:**
- `rf_cue`: RF detection cue (advisory only)
- `fhss_cluster`: FHSS cluster detection
- `aoa_cone`: Angle of Arrival cone (advisory only)
- `rf_energy_start`: RF energy tracking start (advisory only)
- `rf_energy_end`: RF energy tracking end (advisory only)
- `rf_spike`: RF spike detection (advisory only)

**Stages:**
- `energy`: Energy detection stage
- `cue`: Cue stage (advisory only)
- `confirmed`: Confirmed detection (actionable in armed mode)

#### `POST /api/tripwire/cue`
Legacy endpoint (forwards to `/event`).

#### `GET /api/tripwire/ping`
Ping endpoint for connectivity check.

**Response:**
```json
{
  "ok": true
}
```

#### `GET /api/tripwire/aoa-fusion`
Get AoA fusion data (active AoA cones with node GPS positions).

**Response:**
```json
{
  "cones": [
    {
      "node_id": "tripwire-001",
      "freq_hz": 915000000,
      "bearing_deg": 45.0,
      "cone_width_deg": 30.0,
      "gps": {
        "lat": 40.7128,
        "lon": -74.0060
      },
      "timestamp": 1709856000.0
    }
  ]
}
```

#### `POST /api/tripwire/scan-plan`
Update scan plan for a Tripwire node.

**Request Body:**
```json
{
  "node_id": "tripwire-001",
  "scan_plan": "915mhz_scan"
}
```

### Hub/Network

#### `GET /api/hub/nodes`
Get known Tripwire nodes.

**Response:**
```json
{
  "nodes": {
    "tripwire-001": {
      "node_id": "tripwire-001",
      "connected": true,
      "last_seen": 1709856000.0,
      "remote_ip": "192.168.1.100",
      "last_event": {...}
    }
  }
}
```

#### `GET /api/network/config`
Get network configuration.

**Response:**
```json
{
  "host": "0.0.0.0",
  "port": 8080
}
```

#### `POST /api/network/set`
Update network configuration.

**Request Body:**
```json
{
  "host": "0.0.0.0",
  "port": 8080
}
```

### Edge Mode

#### `GET /api/edge/mode`
Get edge mode status.

**Response:**
```json
{
  "mode": "manual"
}
```

#### `POST /api/edge/mode/{mode}`
Set edge mode.

**Path Parameters:**
- `mode`: `manual` or `armed`

**Response:**
```json
{
  "ok": true,
  "mode": "armed"
}
```

## WebSocket Endpoints

### Live FFT WebSocket

#### `WS /ws/live_fft`
Real-time FFT spectrum data stream.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/live_fft');
```

**Hello Message (Text):**
```json
{
  "type": "hello",
  "proto": 1,
  "binary": true,
  "calibration_offset_db": 0.0,
  "power_units": "dBFS"
}
```

**Data Frames (Binary):**
Binary frames contain:
- Header (32 bytes):
  - Magic: `SPRF` (4 bytes)
  - Version: 1 (1 byte)
  - Flags: bitfield (1 byte)
    - `0x01`: Has instant power array
  - Header length: 32 (2 bytes)
  - FFT size: N (4 bytes)
  - Center frequency: Hz (8 bytes, int64)
  - Sample rate: SPS (4 bytes, uint32)
  - Timestamp: seconds (4 bytes, float32)
  - Noise floor: dBFS (4 bytes, float32)
- Power array (max-hold): N × float32
- Power array (instant): N × float32 (if flag set)

**Frame Fields:**
- `power_dbfs`: Max-hold power spectrum (for FFT line display)
- `power_inst_dbfs`: Instant power spectrum (for waterfall display)
- `freqs_hz`: Frequency axis (derived from center_freq, sample_rate, fft_size)
- `noise_floor_dbfs`: Estimated noise floor

### Events WebSocket

#### `WS /ws/notify`
System events and notifications.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/notify');
```

**Event Types:**

**Tripwire Cue:**
```json
{
  "type": "tripwire_cue",
  "payload": {
    "type": "rf_cue",
    "node_id": "tripwire-001",
    "freq_hz": 915000000,
    ...
  }
}
```

**Edge Mode Change:**
```json
{
  "type": "edge_mode",
  "mode": "armed"
}
```

**Tripwire Nodes Update:**
```json
{
  "type": "tripwire_nodes",
  "nodes": {...}
}
```

**Auto-Capture Rejection:**
```json
{
  "type": "tripwire_auto_reject",
  "cue": {...},
  "reason": "cooldown_active"
}
```

### Tripwire Link WebSocket

#### `WS /ws/tripwire_link`
Tripwire node connection management.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/tripwire_link');
```

**Hello Message:**
```json
{
  "type": "hello",
  "node_id": "tripwire-001",
  "callsign": "TW-001",
  "gps": {
    "lat": 40.7128,
    "lon": -74.0060
  }
}
```

**Heartbeat:**
```json
{
  "type": "heartbeat",
  "timestamp": 1709856000.0
}
```

## Error Responses

All endpoints may return error responses:

```json
{
  "ok": false,
  "error": "error_message"
}
```

Common HTTP status codes:
- `200 OK`: Success
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Server error

## Rate Limiting

- Auto-capture policy enforces rate limits
- Manual captures are not rate-limited
- WebSocket connections are not rate-limited

## Notes

- All frequencies are in Hz (integers)
- All timestamps are Unix epoch seconds (float)
- Power values are in dBFS (decibels relative to full scale)
- Sample rates are in samples per second (SPS)
- Durations are in seconds (float)
