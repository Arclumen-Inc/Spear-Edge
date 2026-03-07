# SPEAR-Edge Technical Overview

## System Description

SPEAR-Edge is a Software Defined Radio (SDR) based RF monitoring and capture system designed for the NVIDIA Jetson Orin Nano. It provides real-time spectrum analysis, automated signal capture, and integration with Tripwire nodes for distributed RF monitoring.

## Core Capabilities

- **Real-time Spectrum Analysis**: Live FFT visualization with waterfall display
- **Automated Signal Capture**: Triggered captures from Tripwire node events
- **ML-based Classification**: RF signal classification using PyTorch/ONNX models
- **Tripwire Integration**: Distributed monitoring with multiple Tripwire nodes
- **ATAK Integration**: CoT (Cursor on Target) messaging for tactical awareness

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│              Ingest Application (Port 8000)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Tripwire Event Ingest Endpoint                      │  │
│  │  - Receives events from Tripwire nodes               │  │
│  │  - Stores node registry                              │  │
│  │  - Forwards to main app (port 8080)                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (HTTP forward)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Main Application (Port 8080)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   HTTP API   │  │  WebSocket    │  │   Static UI  │     │
│  │   Routes     │  │   Handlers    │  │   (HTML/JS) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Scan Task   │  │   Capture    │  │  Classifier  │     │
│  │  (FFT Loop)  │  │   Manager    │  │   Pipeline   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   RX Task    │  │  Event Bus   │  │  Tripwire    │     │
│  │  (IQ Read)   │  │  (Pub/Sub)   │  │  Registry    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              BladeRF Native Driver                           │
│         (libbladeRF via ctypes)                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    bladeRF 2.0 Hardware
```

### Component Responsibilities

#### Ingest Application (`ingest_app.py`)
- **Separate FastAPI service** running on port 8000
- Receives Tripwire events via HTTP POST
- Maintains authoritative node registry
- Forwards events to main application (port 8080)
- Handles X-Forwarded-For headers for proper IP tracking

#### Orchestrator (`core/orchestrator/orchestrator.py`)
- **Single source of truth** for SDR hardware ownership
- Manages scan lifecycle (start/stop)
- Coordinates capture operations
- Maintains operator mode (manual/armed/tasked)
- Handles Tripwire integration and auto-capture policy

#### SDR Driver (`core/sdr/bladerf_native.py`)
- Native libbladeRF integration via ctypes
- RF parameter configuration (frequency, sample rate, bandwidth, gain)
- Stream management with power-of-two read sizes
- Hardware health monitoring

#### Scan Pipeline (`core/scan/`)
- **RX Task**: Continuous IQ sample reading from SDR into ring buffer
- **Ring Buffer**: Thread-safe circular buffer for IQ samples
- **Scan Task**: FFT processing, noise floor estimation, frame generation

#### Capture System (`core/capture/`)
- **Capture Manager**: Queue-based capture execution
- **Recorder**: IQ data recording to disk
- **Spectrogram**: Spectrogram generation and thumbnail creation
- **Metadata**: Capture metadata management

#### ML Classification (`ml/`, `core/classify/`)
- **PyTorch Classifier**: GPU-accelerated inference (primary)
- **ONNX Classifier**: CPU inference fallback
- **Stub Classifier**: No-op fallback when ML unavailable
- Hierarchical classification (device + protocol identification)

#### Integration (`core/integrate/`)
- **Tripwire Registry**: Node connection tracking
- **CoT Broadcaster**: ATAK messaging
- **Event Handling**: Tripwire event processing

## Data Flow

### Live Scan Flow

```
SDR Hardware
    │
    ▼
RX Task (async) ──► Ring Buffer ──► Scan Task (FFT)
    │                                    │
    │                                    ▼
    │                            Event Bus ──► WebSocket ──► UI
    │
    └─────────────────────────────────────────────────────────┘
```

### Capture Flow

```
Tripwire Event / Manual Request
    │
    ▼
Capture Manager Queue
    │
    ▼
Capture Worker (async)
    │
    ├─► Pause Scan
    ├─► Tune SDR to Capture Frequency
    ├─► Record IQ Samples
    ├─► Generate Spectrogram
    ├─► ML Classification
    ├─► Resume Scan
    └─► Save Artifacts
```

## Operating Modes

### Manual Mode
- Operator has full control
- No automatic captures
- All captures are manual

### Armed Mode
- Automatic captures enabled (with policy guards)
- Tripwire confirmed events trigger captures
- Rate limiting and cooldown protection
- Operator can still perform manual captures

### Tasked Mode (Internal)
- Transient mode for scheduled tasks
- Not directly settable by operator

## Key Design Principles

### 1. Single Source of Truth
- Orchestrator owns SDR hardware exclusively
- No direct SDR access from routes or other components
- All SDR operations go through orchestrator

### 2. Async-First Architecture
- All I/O operations are async
- Blocking SDR operations use `asyncio.to_thread()`
- Event-driven communication via EventBus

### 3. Hardware Safety
- bladeRF stream lifecycle strictly enforced
- RF parameters set in correct order
- Power-of-two read sizes required
- Gain protection to prevent clipping

### 4. Graceful Degradation
- ML classification falls back (PyTorch → ONNX → Stub)
- WebSocket disconnects handled gracefully
- Capture failures don't crash runtime

## Technology Stack

### Backend
- **Python 3.10+**: Core language
- **FastAPI**: HTTP/WebSocket framework
- **asyncio**: Async concurrency
- **NumPy**: Signal processing
- **PyTorch/ONNX**: ML inference

### Frontend
- **Vanilla JavaScript**: No frameworks
- **Canvas API**: FFT/waterfall rendering
- **WebSocket**: Real-time data streaming

### Hardware
- **Jetson Orin Nano**: Compute platform
- **bladeRF 2.0 micro**: SDR hardware
- **libbladeRF**: Native driver library

## Performance Characteristics

### Sample Rate Limits
- Maximum: ~30 MS/s (Jetson Orin Nano)
- Default: 2.4 MS/s
- Ring buffer: 0.5 seconds of buffering

### FFT Processing
- Default FFT size: 2048
- Default FPS: 15
- Window: Hanning (for sidelobe suppression)
- Normalization: Window energy-based

### Capture Performance
- Queue size: 8 captures
- Cooldown: 1.5 seconds between captures
- Priority: Armed mode captures (60) > Manual (50)

## File Organization

```
spear_edge/
├── app.py                 # Main FastAPI application (port 8080)
├── ingest_app.py          # Ingest application (port 8000)
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
│   └── models/           # Model files
└── ui/                    # Frontend
    └── web/               # HTML/CSS/JS

data/
├── artifacts/
│   └── captures/         # Capture artifacts
└── dataset/              # Training datasets

docs/                     # Documentation
tests/                    # Test files
scripts/                  # Utility scripts
```

## Configuration

### Environment Variables

- `SPEAR_HOST`: Server host (default: `0.0.0.0`)
- `SPEAR_PORT`: Server port (default: `8080`)
- `SPEAR_LOG_LEVEL`: Logging level (default: `WARNING`)
- `SPEAR_CENTER_FREQ_HZ`: Default center frequency (default: `915000000`)
- `SPEAR_SAMPLE_RATE_SPS`: Default sample rate (default: `2400000`)
- `SPEAR_FFT_SIZE`: Default FFT size (default: `2048`)
- `SPEAR_FPS`: Default FPS (default: `15.0`)
- `SPEAR_CALIBRATION_OFFSET_DB`: RF calibration offset (default: `0.0`)
- `SPEAR_IQ_SCALING_MODE`: IQ scaling mode (default: `int16`)

## Security Considerations

- CORS enabled for all origins (development)
- No authentication (local network only)
- File system access limited to data directories
- No external network dependencies (except Tripwire nodes)

## Future Enhancements

- Multi-SDR support
- Advanced scheduling and tasking
- Enhanced ML model training pipeline
- Real-time AoA (Angle of Arrival) processing
- Extended protocol support
