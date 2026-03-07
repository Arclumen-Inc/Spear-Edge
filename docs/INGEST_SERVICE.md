# SPEAR-Edge Ingest Service

## Overview

The Ingest Service is a separate FastAPI application that runs on port 8000 and serves as the authoritative endpoint for receiving Tripwire events. It maintains the node registry and forwards events to the main application.

## Purpose

- **Event Ingestion**: Receives HTTP POST events from Tripwire nodes
- **Node Registry**: Maintains authoritative record of Tripwire nodes
- **Event Forwarding**: Forwards events to main application (port 8080)
- **IP Tracking**: Handles X-Forwarded-For headers for proper client IP tracking

## Architecture

```
Tripwire Node
    │
    │ HTTP POST /api/tripwire/event
    ▼
Ingest Service (Port 8000)
    │
    ├─► Store node registry
    ├─► Update node status
    └─► Forward to main app (port 8080)
            │
            ▼
    Main Application (Port 8080)
            │
            ├─► Process event
            ├─► Trigger capture (if armed)
            └─► Update UI via WebSocket
```

## Running the Service

### Standalone

```bash
uvicorn spear_edge.ingest_app:app --host 0.0.0.0 --port 8000
```

### With Main Application

Run both services:

```bash
# Terminal 1: Main application
uvicorn spear_edge.app:app --host 0.0.0.0 --port 8080

# Terminal 2: Ingest service
uvicorn spear_edge.ingest_app:app --host 0.0.0.0 --port 8000
```

Or use a process manager (systemd, supervisor, etc.) to run both.

## API Endpoint

### `POST /api/tripwire/event`

Receive Tripwire event.

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
  "status": "ok"
}
```

**Behavior:**
1. Extracts node_id from payload
2. Updates node registry with:
   - `node_id`
   - `connected`: true
   - `last_seen`: current timestamp
   - `remote_ip`: client IP (handles X-Forwarded-For)
   - `last_event`: event payload slice
3. Saves node registry to `data/artifacts/tripwire_nodes.json`
4. Forwards event to main app: `http://127.0.0.1:8080/api/tripwire/cue`
5. Returns success response

## Node Registry

The ingest service maintains a node registry file:
- **Location**: `data/artifacts/tripwire_nodes.json`
- **Format**: JSON object with node_id keys
- **Fields**:
  - `node_id`: Node identifier
  - `connected`: Connection status (true if event received recently)
  - `last_seen`: Timestamp of last event
  - `remote_ip`: Client IP address
  - `last_event`: Last event payload (subset)

**Example:**
```json
{
  "nodes": {
    "tripwire-001": {
      "node_id": "tripwire-001",
      "connected": true,
      "last_seen": 1709856000.0,
      "remote_ip": "192.168.1.100",
      "last_event": {
        "classification": "elrs",
        "freq_hz": 915000000,
        "delta_db": 15.5,
        "remarks": null,
        "scan_plan": "915mhz_scan",
        "timestamp": 1709856000.0
      }
    }
  }
}
```

## IP Tracking

The ingest service handles IP tracking for forwarded requests:

1. **Direct Connection**: Uses `request.client.host`
2. **Forwarded Connection**: Uses `X-Forwarded-For` header if present
3. **Localhost Forward**: If request comes from localhost, extracts original IP from header

This ensures proper IP tracking even when requests are forwarded through proxies or load balancers.

## Integration with Main Application

The ingest service forwards events to the main application:

- **Endpoint**: `http://127.0.0.1:8080/api/tripwire/cue`
- **Method**: HTTP POST
- **Headers**: Includes `X-Forwarded-For` with original client IP
- **Behavior**: Best-effort forwarding (errors are silently ignored)

The main application then:
- Processes the event
- Updates orchestrator state
- Triggers captures (if in armed mode)
- Updates UI via WebSocket

## Configuration

No special configuration required. The service uses default FastAPI settings.

**Optional**: Set via environment variables:
- `SPEAR_INGEST_HOST`: Host (default: `0.0.0.0`)
- `SPEAR_INGEST_PORT`: Port (default: `8000`)

## Deployment

### Development

Run both services manually or use a process manager.

### Production

Use a process manager (systemd, supervisor, etc.):

**systemd example:**
```ini
[Unit]
Description=SPEAR-Edge Ingest Service
After=network.target

[Service]
Type=simple
User=spear
WorkingDirectory=/home/spear/spear-edgev1_0
Environment="PATH=/home/spear/spear-edgev1_0/venv/bin"
ExecStart=/home/spear/spear-edgev1_0/venv/bin/uvicorn spear_edge.ingest_app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Service Not Starting

1. Check port 8000 is available: `netstat -tuln | grep 8000`
2. Check permissions: Ensure user has access to data directory
3. Check logs: Review uvicorn output for errors

### Events Not Forwarding

1. Verify main app is running on port 8080
2. Check network connectivity: `curl http://127.0.0.1:8080/health`
3. Review ingest service logs for forwarding errors

### Node Registry Not Updating

1. Check data directory exists: `data/artifacts/`
2. Check write permissions
3. Verify events are being received: Check logs for `[INGEST]` messages

## Notes

- The ingest service is **stateless** (except for node registry file)
- Multiple ingest instances can run (with different ports) for load balancing
- The main application can also receive events directly (via `/api/tripwire/event`)
- The ingest service is **optional** - the main app can function without it

## Relationship to Main Application

The ingest service is a **convenience layer** that:
- Provides a dedicated endpoint for Tripwire nodes
- Maintains node registry separately from main app
- Handles IP tracking and forwarding

The main application can receive events directly, but the ingest service provides:
- Separation of concerns
- Independent scaling
- Dedicated endpoint for external services
