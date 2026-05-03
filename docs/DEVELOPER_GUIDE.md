# SPEAR-Edge Developer Guide

## Codebase Structure

```
spear_edge/
├── app.py                 # Main FastAPI application (port 8080)
├── ingest_app.py          # Ingest service (port 8000) - Tripwire event ingestion
├── settings.py            # Configuration
├── api/                   # HTTP and WebSocket endpoints
│   ├── http/             # REST API routes
│   └── ws/               # WebSocket handlers
├── core/                  # Core functionality
│   ├── orchestrator/     # Main orchestrator
│   ├── sdr/              # SDR drivers
│   ├── scan/             # FFT processing
│   ├── capture/          # Capture management
│   ├── classify/         # ML classification
│   ├── integrate/        # Tripwire/ATAK integration
│   ├── bus/              # Event bus
│   └── gps/              # GPS client
├── ml/                    # ML models and inference
└── ui/                    # Frontend
    └── web/               # HTML/CSS/JS
```

## Development Setup

### Prerequisites

- Python 3.10+
- Virtual environment
- Development dependencies (see [Installation Guide](INSTALLATION.md))

### Setup Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-ml-dev.txt

# Install development tools (optional)
pip install pytest black flake8 mypy
```

### Running in Development

```bash
# Set debug logging
export SPEAR_LOG_LEVEL=DEBUG

# Run main application with auto-reload
uvicorn spear_edge.app:app --host 0.0.0.0 --port 8080 --reload

# Run ingest service (optional, separate terminal)
uvicorn spear_edge.ingest_app:app --host 0.0.0.0 --port 8000 --reload
```

**Note:** The ingest service is optional but recommended for Tripwire integration. See [Ingest Service](INGEST_SERVICE.md) for details.

## Architecture Patterns

### Single Source of Truth

The **Orchestrator** owns SDR hardware exclusively:
- No direct SDR access from routes
- All SDR operations go through orchestrator
- Prevents race conditions and state conflicts

**Example:**
```python
# ✅ Good: Use orchestrator
await orchestrator.start_scan(...)

# ❌ Bad: Direct SDR access
sdr.tune(...)  # Don't do this
```

### Async-First

All I/O operations are async:
- Blocking SDR operations use `asyncio.to_thread()`
- Event-driven communication via EventBus
- Never block the event loop

**Example:**
```python
# ✅ Good: Async with thread
iq = await asyncio.to_thread(self.sdr.read_samples, num_samples)

# ❌ Bad: Blocking in async
iq = self.sdr.read_samples(num_samples)  # Blocks event loop
```

### Event Bus Pattern

Components communicate via EventBus:
- Publisher: `bus.publish_nowait(event_type, data)`
- Subscriber: `bus.subscribe(event_type, callback)`
- Decouples components

**Example:**
```python
# Publish event
orchestrator.bus.publish_nowait("live_spectrum", frame)

# Subscribe to event
q = await orchestrator.bus.subscribe("live_spectrum", maxsize=2)
```

## Key Components

### Orchestrator

**Location:** `core/orchestrator/orchestrator.py`

**Responsibilities:**
- SDR lifecycle management
- Scan control (start/stop)
- Mode management (manual/armed/tasked)
- Tripwire integration
- Auto-capture policy enforcement

**Key Methods:**
- `start_scan()`: Start live scanning
- `stop_scan()`: Stop scanning
- `can_auto_capture()`: Check auto-capture policy
- `record_tripwire_cue()`: Record Tripwire event

### SDR Driver

**Location:** `core/sdr/bladerf_native.py`

**Responsibilities:**
- Native libbladeRF integration
- RF parameter configuration
- Stream management
- Hardware health monitoring

**Critical Constraints:**
1. **Stream Lifecycle Order:**
   - Set sample rate FIRST
   - Set bandwidth
   - Set frequency
   - Enable RX channel
   - Create/activate stream LAST

2. **Read Sizes:**
   - Must be power-of-two (8192, 16384, etc.)
   - Never use arbitrary sizes

3. **Gain Management:**
   - LNA gain automatically optimized
   - BT200 disabled by default (hardware not connected)

### Scan Pipeline

**Components:**
- **RX Task** (`core/scan/rx_task.py`): Continuous IQ reading
- **Ring Buffer** (`core/scan/ring_buffer.py`): Thread-safe circular buffer
- **Scan Task** (`core/scan/scan_task.py`): FFT processing

**Data Flow:**
```
SDR → RX Task → Ring Buffer → Scan Task → Event Bus
```

### Capture System

**Location:** `core/capture/capture_manager.py`

**Responsibilities:**
- Queue-based capture execution
- IQ recording
- Spectrogram generation
- ML classification integration

**Capture Lifecycle:**
1. Request queued
2. Worker dequeues request
3. Pause scan
4. Tune SDR
5. Record IQ samples
6. Generate spectrogram
7. Run ML classification
8. Resume scan
9. Save artifacts

### Wi-Fi Monitor Domain (Kismet-backed)

SPEAR now includes a separate Wi-Fi monitor runtime path independent from bladeRF scan/capture.

Key modules:

- `spear_edge/core/wifi_monitor/models.py`
- `spear_edge/core/wifi_monitor/manager.py`
- `spear_edge/core/wifi_monitor/provider_base.py`
- `spear_edge/core/wifi_monitor/provider_kismet.py`
- `spear_edge/core/wifi_monitor/provider_generic.py`

Lifecycle:

- Created during app startup in `spear_edge/app.py`
- Available as `app.state.wifi_monitor` and `state.wifi_monitor`
- Optional autostart via `SPEAR_WIFI_MONITOR_AUTOSTART`
- Stopped during app shutdown

Event bus topics:

- `rid_update` (RID detections from Wi-Fi monitor path)
- `wifi_intel_update` (channel/device/source/anomaly updates)

WebSocket:

- `spear_edge/api/ws/events_ws.py` forwards both topics via `/ws/notify`.

HTTP API:

- Route module: `spear_edge/api/http/routes_wifi_monitor.py`
- Base: `/api/wifi-monitor`

Control/status endpoints:

- `GET /status`
- `POST /start`
- `POST /stop`
- `POST /config`
- `POST /test-kismet`

Datasource/channel control endpoints:

- `GET /interfaces`
- `GET /datasources`
- `POST /datasource/add`
- `POST /datasource/open`
- `POST /datasource/close`
- `POST /datasource/set-channel`
- `POST /datasource/set-hop`

Alert endpoints:

- `GET /alerts`
- `POST /alerts/presence`

Spear Manager proxy endpoints (for Kismet service control from `/wifi`):

- `GET /manager/kismet/status`
- `POST /manager/kismet/start`
- `POST /manager/kismet/stop`

These call Spear Manager (`SPEAR_WIFI_MANAGER_URL`) and include bearer auth
when `SPEAR_WIFI_MANAGER_TOKEN` is set.

Install / Kismet routes: **`docs/SPEAR_MANAGER.md`**, **`scripts/install_spear_manager.sh`** (unpacks tarball under `/home/spear/spear_manager` and patches `main.py`).

Wi-Fi monitor environment variables:

- `SPEAR_WIFI_MONITOR_AUTOSTART` (`true|false`)
- `SPEAR_WIFI_MONITOR_BACKEND` (`kismet|generic`)
- `SPEAR_WIFI_MONITOR_IFACE`
- `SPEAR_WIFI_MONITOR_CHANNEL_MODE` (`hop|fixed`)
- `SPEAR_WIFI_MONITOR_POLL_INTERVAL_S`
- `SPEAR_WIFI_MONITOR_HOP_CHANNELS` (comma-separated list)
- `SPEAR_WIFI_MONITOR_KISMET_URL` (default `http://127.0.0.1:2501`)
- `SPEAR_WIFI_MONITOR_KISMET_USERNAME`
- `SPEAR_WIFI_MONITOR_KISMET_PASSWORD`
- `SPEAR_WIFI_MONITOR_KISMET_TIMEOUT_S`
- `SPEAR_WIFI_MANAGER_URL` (for Kismet service control proxy)
- `SPEAR_WIFI_MANAGER_TOKEN`

#### Remote ID decode wiring (Phase 1B)

- Producer entrypoint: `RemoteIdDecoder.produce_artifact(...)`
- Consumer entrypoint: `RemoteIdDecoder.decode(...)`
- Standard artifact location: `decode/remote_id.json` under each capture directory
- CaptureManager runs artifact production before protocol fusion.

Environment variables:

- `SPEAR_RID_DECODER_CMD`:
  command called by CaptureManager to attempt decode generation.
  Default service config uses:
  `/home/spear/spear-edgev1_0/venv/bin/python /home/spear/spear-edgev1_0/scripts/rid_decode_wrapper.py`
- `SPEAR_RID_BACKEND_CMD`:
  optional backend decoder command consumed by the wrapper.
  For local integration testing, you can point this at:
  `/home/spear/spear-edgev1_0/scripts/rid_backend_stub.py`
- `SPEAR_DJI_DECODER_CMD`:
  command called by CaptureManager to generate DJI decode artifacts.
  Default service config uses:
  `/home/spear/spear-edgev1_0/venv/bin/python /home/spear/spear-edgev1_0/scripts/dji_decode_wrapper.py`
- `SPEAR_DJI_BACKEND_CMD`:
  optional backend command consumed by DJI wrapper.
  For local integration testing, you can point this at:
  `/home/spear/spear-edgev1_0/scripts/dji_backend_stub.py`

Expected backend CLI contract:

- accepts: `--iq-path --center-freq-hz --sample-rate-sps --output-json`
- writes JSON output matching protocol result schema (or wrapper falls back to `decode_error` / `no_decode`)

RID wrapper normalization behavior:

- Accepts canonical keys directly:
  - `protocol`, `status`, `confidence`, `validation`, `decoded_fields`, `evidence`
- Also accepts common minimal backend forms and normalizes:
  - `fields` or `data` as alternatives to `decoded_fields`
  - top-level `crc_pass`, `frame_count`
  - field aliases: `id/serial` -> `uas_id`, `lat/lon` -> `latitude/longitude`
- Status inference if omitted:
  - `decoded_verified` when `crc_pass=true` and `frame_count>0`
  - `decoded_partial` when decoded fields exist without full validation
  - `no_decode` otherwise

DJI wrapper normalization behavior matches RID wrapper behavior
(same accepted aliases and status inference rules), but with `protocol=dji_droneid`.

#### Live field backends (optional)

- **DJI DroneID from IQ:** `scripts/dji_samples2droneid_backend.py` shells out to
  [samples2djidroneid](https://github.com/anarkiwi/samples2djidroneid) (native or Docker).
  - `SPEAR_DJI_SAMPLES2_CMD` — native decoder argv0 / path (default `samples2djidroneid`).
  - `SPEAR_DJI_SAMPLES2_DOCKER_IMAGE` — if set, run `docker run ... <image> <iq>` instead of native.
  - `SPEAR_DJI_SAMPLES2_TIMEOUT_S` — subprocess timeout seconds (default `180`).
  - Requires capture sample rate **15360000** or **30720000** sps.
- **ASTM Remote ID from PCAP sidecar:** `scripts/rid_pcap_sidecar_backend.py` looks for
  `remote_rid.pcap(ng)` or `rid_sidecar.pcap(ng)` beside `iq/samples.iq`, then runs:
  - `SPEAR_RID_PCAP_DECODER_CMD` — must contain both `{PCAP}` and `{OUT}` placeholders; command
    is `shlex.split` after substitution; decoder should write JSON to `{OUT}` or print JSON on stdout.

### ML Classification

**Location:** `ml/`, `core/classify/`

**Fallback Chain:**
1. PyTorch (GPU-accelerated)
2. ONNX (CPU fallback)
3. Stub (no-op)

**Model Requirements:**
- Input: 512×512 float32 spectrogram
- Output: Classification with confidence
- Labels: `class_labels.json`

## Adding New Features

### Adding a New API Endpoint

1. Create route in `api/http/`:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/myfeature", tags=["myfeature"])

@router.get("/status")
async def get_status():
    return {"status": "ok"}
```

2. Register in `app.py`:
```python
from spear_edge.api.http.routes_myfeature import router as myfeature_router
app.include_router(myfeature_router)
```

### Adding a New WebSocket Handler

1. Create handler in `api/ws/`:
```python
async def my_ws(websocket: WebSocket, orchestrator):
    await websocket.accept()
    # ... handler logic
```

2. Register in `app.py`:
```python
@app.websocket("/ws/myfeature")
async def ws_myfeature(websocket: WebSocket):
    await my_ws(websocket, app.state.orchestrator)
```

### Adding a New Event Type

1. Define event in `core/bus/models.py`:
```python
@dataclass
class MyEvent:
    field1: str
    field2: int
```

2. Publish event:
```python
orchestrator.bus.publish_nowait("my_event", MyEvent(...))
```

3. Subscribe to event:
```python
q = await orchestrator.bus.subscribe("my_event", maxsize=10)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_specific.py

# Run with coverage
pytest --cov=spear_edge
```

### Test Structure

```
tests/
├── test_sdr.py          # SDR driver tests
├── test_orchestrator.py # Orchestrator tests
├── test_capture.py      # Capture tests
└── test_api.py         # API endpoint tests
```

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Use dataclasses for data models
- Async functions clearly named

**Example:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MyModel:
    field1: str
    field2: Optional[int] = None

async def my_async_function(param: str) -> MyModel:
    """Async function with type hints."""
    return MyModel(field1=param)
```

### JavaScript

- Use `const` for constants
- Use `let` for variables
- Use arrow functions for callbacks
- Cache DOM references

**Example:**
```javascript
const CACHE_SIZE = 100;
let currentIndex = 0;

const processData = (data) => {
    // Process data
};
```

## Debugging

### Logging

Use descriptive prefixes:
```python
print("[COMPONENT] Message")
```

**Prefixes:**
- `[ORCH]`: Orchestrator
- `[SCAN]`: Scan task
- `[RX]`: RX task
- `[CAPTURE]`: Capture manager
- `[SDR]`: SDR driver
- `[FFT WS]`: FFT WebSocket
- `[API]`: API routes

### Debug Mode

Set logging level:
```bash
export SPEAR_LOG_LEVEL=DEBUG
```

### Common Issues

**Stream returns 0 samples:**
- Stream not activated
- Wrong read size (not power-of-two)
- Stream deactivated

**Sample rate reverts:**
- Stream activated before RF parameters set
- Fix: Set RF parameters before stream activation

**UI freezing:**
- Blocking operations in event loop
- Fix: Use `asyncio.to_thread()` for blocking ops

**Waterfall not scrolling:**
- Canvas coordinate system mismatch
- Check canvas width/height

## Performance Optimization

### Jetson Orin Nano Considerations

- Use NumPy operations (vectorized)
- Minimize Python object allocations
- Use `np.complex64` (not complex128)
- Ring buffer is thread-safe

### Frontend Optimization

- Use offscreen canvas for waterfall
- Cache grid/axis drawing
- Drop frames if rendering falls behind
- Throttle logging

## Contributing

### Code Review Checklist

- [ ] Follows architecture patterns
- [ ] Uses async correctly
- [ ] No direct SDR access (use orchestrator)
- [ ] Proper error handling
- [ ] Logging with prefixes
- [ ] Type hints included
- [ ] Tests added/updated

### Commit Messages

Format: `[COMPONENT] Brief description`

Examples:
- `[ORCH] Add auto-capture policy validation`
- `[SDR] Fix stream lifecycle order`
- `[API] Add capture label endpoint`

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [libbladeRF Documentation](https://github.com/Nuand/bladeRF/wiki)
- [NumPy Documentation](https://numpy.org/doc/)

## Getting Help

1. Check logs for error messages
2. Review [Technical Overview](TECHNICAL_OVERVIEW.md)
3. Check [API Reference](API_REFERENCE.md)
4. Review code comments and docstrings
