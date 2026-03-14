# SPEAR-Edge Project Writeup

**Version**: 1.0  
**Date**: 2025-03-07  
**Platform**: NVIDIA Jetson Orin Nano  
**SDR Hardware**: bladeRF 2.0 micro (xA4/xA9)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Core Features](#core-features)
4. [System Architecture](#system-architecture)
5. [Technical Components](#technical-components)
6. [Workflows](#workflows)
7. [Integration Points](#integration-points)
8. [Hardware Requirements](#hardware-requirements)
9. [Key Technical Details](#key-technical-details)
10. [Performance Characteristics](#performance-characteristics)

---

## Executive Summary

SPEAR-Edge is a comprehensive Software Defined Radio (SDR) based RF monitoring and capture system designed for the NVIDIA Jetson Orin Nano platform. It provides real-time spectrum analysis, automated signal capture, machine learning-based signal classification, and integration with distributed Tripwire sensor nodes and the Android Team Awareness Kit (ATAK).

The system operates in three modes:
- **Manual Mode**: Operator-controlled RF monitoring and capture
- **Armed Mode**: Automatic capture triggered by Tripwire node events
- **Tasked Mode**: Internal transient state during active captures

Key capabilities include:
- Real-time FFT/waterfall visualization at 30 FPS
- Memory-efficient IQ capture pipeline (streams directly to disk)
- ML-ready signal classification (PyTorch GPU-accelerated)
- Tripwire node integration via WebSocket and HTTP
- ATAK integration via CoT (Cursor on Target) protocol
- Web-based user interface for monitoring and control

---

## Project Overview

### Purpose

SPEAR-Edge serves as a tactical RF monitoring and analysis system that can operate standalone or as part of a distributed sensor network. It captures RF signals, classifies them using machine learning, and provides situational awareness through integration with ATAK.

### Design Philosophy

The system follows an event-driven, asynchronous architecture with clear separation of concerns:

- **Hardware Abstraction**: SDR drivers implement a common interface (`SDRBase`)
- **Event-Driven Communication**: Pub/sub event bus for loose coupling
- **Memory Efficiency**: Streaming IQ capture to disk, chunked processing
- **Performance Optimization**: GPU-accelerated ML, vectorized NumPy operations
- **Reliability**: Never crash runtime, graceful error handling

### Target Use Cases

1. **Standalone RF Monitoring**: Manual spectrum analysis and signal capture
2. **Distributed Sensor Network**: Integration with Tripwire nodes for coordinated monitoring
3. **Tactical Operations**: ATAK integration for real-time situational awareness
4. **ML Training Data Collection**: Automated capture and labeling workflow

---

## Core Features

### 1. Real-Time Spectrum Analysis

**FFT Processing Pipeline**:
- Hanning window for frequency domain analysis
- Configurable FFT size (default: 2048 points)
- Frame rate control (default: 15-30 FPS)
- Adaptive noise floor estimation (2nd/10th percentile)
- Spectrum smoothing (EMA, configurable alpha)
- Edge bin handling (zero first/last 2.5% to remove window artifacts)

**Waterfall Visualization**:
- Time-frequency representation with color-coded power levels
- Dynamic range: noise floor ± 50 dB
- Multiple color palettes (classic, custom)
- Brightness/contrast/gamma controls
- Smooth scrolling with offscreen canvas optimization

**Display Modes**:
- Instant mode: Fast attack, slow decay (default)
- Peak hold mode: Maximum power tracking
- Max-hold: 0.35s reset window for FHSS visibility

### 2. Automated Signal Capture

**Capture Types**:
- **Manual Captures**: Operator-initiated via UI or API
- **Armed Captures**: Automatic triggers from Tripwire events
- **Tasked Captures**: Internal captures during active jobs

**Capture Pipeline**:
1. Pause live scan
2. Tune SDR to capture frequency
3. Apply bandwidth and gain settings
4. Stream IQ samples directly to disk (memory-efficient)
5. Generate spectrogram (chunked processing, ≤512x512)
6. Compute signal statistics (SNR, occupied bandwidth, duty cycle)
7. Classify signal using ML model
8. Generate artifacts (IQ file, spectrogram PNG, metadata JSON)
9. Resume live scan

**Artifact Structure**:
```
capture_<timestamp>_<freq>Hz_<rate>sps_<reason>/
├── iq/
│   ├── samples.iq (raw complex64)
│   └── samples.sigmf-meta (SigMF metadata)
├── features/
│   ├── spectrogram.npy (ML-ready tensor, ≤512x512)
│   ├── psd.npy (power spectral density)
│   └── stats.json (SNR, bandwidth, duty cycle, etc.)
├── thumbnails/
│   └── spectrogram.png (visualization)
├── capture.json (comprehensive metadata)
└── interchange/
    └── vita49.vrt (optional VITA-49 format)
```

**Quality Metrics**:
- Signal presence detection
- SNR estimation
- Clipping detection
- DC offset warnings
- Partial capture detection

### 3. Machine Learning Classification

**ML Pipeline**:
- **Input**: Spectrogram tensor (≤512x512, float32, noise-floor normalized)
- **Model**: PyTorch CNN (GPU-accelerated on Jetson)
- **Output**: Device/protocol classification with confidence scores
- **Fallback Chain**: PyTorch → ONNX Runtime → Stub classifier

**Classification Features**:
- 23+ device/protocol classes (ELRS, DJI, analog video, etc.)
- Hierarchical classification (device name + signal type)
- Top-k predictions with confidence scores
- GPU acceleration for real-time inference
- Extensible through fine-tuning workflow

**Training Workflow**:
- Dataset preparation from captures
- Fine-tuning scripts for new classes
- Model export (PyTorch .pth, ONNX .onnx)
- Model management via ML dashboard

### 4. Tripwire Integration

**Event Types** (Tripwire v1.1/v2.0 alignment):
- **Cues** (`rf_cue`, `aoa_cone`): Advisory only, never actionable
- **Confirmed Events** (`confirmed_event`, `fhss_cluster`): Actionable for auto-capture
- **System Events**: Calibration, metrics (not actionable)

**Auto-Capture Policy** (Armed Mode):
- Minimum confidence threshold (default: 0.90)
- Global cooldown (default: 3.0s)
- Per-node cooldown (default: 2.0s)
- Per-frequency cooldown (default: 8.0s, 100 kHz bins)
- Rate limiting (default: 10 captures/minute)

**Tripwire Registry**:
- Tracks up to 3 connected nodes
- Node health monitoring (5s timeout)
- GPS location tracking
- WebSocket link management

**AoA (Angle of Arrival) Support**:
- Cone event tracking (20 cone buffer)
- TAI (Triangulated Area of Interest) calculation
- Bearing visualization
- ATAK marker generation
- **AoA fusion API** (`GET /api/tripwire/aoa-fusion`): Returns active cones from up to 3 nodes with node GPS for triangulation; consumed by the main UI’s AoA fusion visualization.

### 5. ATAK Integration

**CoT (Cursor on Target) Protocol**:
- Chat messages for capture notifications
- Detection markers with classification results
- Status messages (online/offline, tripwire count)
- TAI (Triangulated Area of Interest) polygons

**Message Types**:
- Capture start notifications
- Classification results (confirmed events only)
- Edge status (armed/online with tripwire count)
- AoA-derived TAI polygons

### 6. Web-Based User Interface

**Main Dashboard** (`index.html`):
- Real-time FFT/waterfall display
- SDR controls (frequency, sample rate, bandwidth, gain)
- Capture controls (manual capture, armed mode toggle)
- Tripwire node status (from hub + WebSocket), scan plan dropdown and “Send plan” per node
- AoA fusion visualization (cones + triangulation)
- Capture history
- System health metrics
- **Network configuration**: View/set interface IP addresses via `GET/POST /api/network/*` (UI in header/settings)

**Hub**:
- Read-only node registry (`GET /api/hub/nodes`) backed by persisted file; live updates via WebSocket `tripwire_nodes`.

**ML Dashboard** (`ml.html`, `GET /ml`):
- Capture browser with thumbnails, filters (label, source)
- Single and batch label edit; batch delete
- Model management: list, load, activate, export (ZIP), import (ZIP), test on capture
- **Quick train**: Fine-tune on 1–2 captures (classification head), progress and cancel
- Class labels and capture/dataset stats
- Training dataset export

**UI Technologies**:
- Vanilla JavaScript (no frameworks)
- Canvas API for FFT/waterfall rendering
- WebSocket for real-time updates
- Responsive design for mobile/desktop

---

## System Architecture

### High-Level Architecture

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
│              Main Application (Port 8080)                   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Orchestrator │  │ Capture Mgr  │  │ Classifier    │   │
│  │              │  │              │  │              │   │
│  │ - SDR owner  │  │ - Queue mgmt │  │ - PyTorch/ONNX│   │
│  │ - Mode ctrl  │  │ - Artifacts  │  │ - GPU accel   │   │
│  │ - Event bus  │  │ - ML features│  │              │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                  │                  │           │
│  ┌──────▼──────────────────▼──────────────────▼───────┐  │
│  │              Event Bus (Pub/Sub)                     │  │
│  │  - live_spectrum, capture_start, capture_result     │  │
│  │  - tripwire_nodes, edge_mode, classification_result │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Scan Task    │  │ RX Task      │  │ Ring Buffer  │   │
│  │              │  │              │  │              │   │
│  │ - FFT proc   │  │ - SDR reads  │  │ - Thread-safe│   │
│  │ - Frame gen  │  │ - Line rate  │  │ - IQ samples │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                  │                  │           │
│  ┌──────▼──────────────────▼──────────────────▼───────┐  │
│  │         SDR Driver (BladeRFNativeDevice)              │  │
│  │  - Native libbladeRF.so.2 bindings                   │  │
│  │  - Stream lifecycle management                        │  │
│  │  - RF parameter configuration                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ HTTP API     │  │ WebSocket    │  │ CoT          │   │
│  │              │  │              │  │              │   │
│  │ - REST routes│  │ - Live FFT  │  │ - ATAK msgs  │   │
│  │ - Capture    │  │ - Events    │  │ - Markers     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**Orchestrator** (`orchestrator.py`):
- Single authoritative owner of SDR hardware
- Mode management (manual/armed/tasked)
- Scan lifecycle (start/stop)
- Tripwire cue recording (advisory only)
- Auto-capture policy enforcement
- ATAK status messaging

**CaptureManager** (`capture_manager.py`):
- Capture queue management (max 8 concurrent)
- IQ capture to disk (memory-efficient)
- Spectrogram generation (chunked processing)
- ML feature extraction
- Artifact writing (IQ, spectrogram, metadata)
- Classification integration

**ScanTask** (`scan_task.py`):
- FFT processing from ring buffer
- Frame rate control (FPS limiting)
- Spectrum smoothing (EMA)
- Noise floor estimation
- Frequency axis calculation

**RxTask** (`rx_task.py`):
- Continuous SDR sample reading
- Line-rate IQ capture
- Ring buffer population
- Stream health monitoring

**BladeRFNativeDevice** (`bladerf_native.py`):
- Native libbladeRF.so.2 bindings (ctypes)
- RF parameter configuration (critical order)
- Stream lifecycle management
- Gain control (manual/AGC)
- BT200 external LNA support (optional)

**EventBus** (`event_bus.py`):
- Pub/sub messaging system
- Async queue-based delivery
- Topic-based subscriptions
- Latest-wins semantics for UI updates

---

## Technical Components

### SDR Driver Architecture

**Base Interface** (`base.py`):
```python
class SDRBase:
    def open(self)
    def close(self)
    def apply_config(self, cfg: SdrConfig)
    def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None)
    def set_gain(self, gain_db: float)
    def set_gain_mode(self, mode: GainMode)
    def read_samples(self, num_samples)
    def get_info(self) -> dict
    def get_health(self) -> dict
```

**BladeRF Implementation** (`bladerf_native.py`):
- Native C library bindings via ctypes
- Critical configuration order:
  1. Set sample rate FIRST
  2. Set bandwidth
  3. Set frequency
  4. Enable RX channel
  5. Configure and activate stream
  6. Set gain (AFTER stream active)

**Key Constraints**:
- Read sizes MUST be power-of-two (8192, 16384, etc.)
- Stream must be activated AFTER all RF parameters
- Gain must be set AFTER stream activation
- USB buffer configuration (stay under 16MB Linux limit)

### FFT Processing Pipeline

**Window Function**:
- Type: Hanning (Hann) window
- Rationale: Lower sidelobes than rectangular, better resolution than Blackman
- Coherent gain: ≈0.5 (normalized by window sum)

**Processing Steps**:
1. Extract IQ samples from ring buffer
2. Ensure complex64 dtype
3. Optional DC offset removal (configurable, disabled for wideband)
4. Apply Hanning window
5. Compute FFT with fftshift (center frequency)
6. Calculate magnitude, normalize by window sum
7. Convert to dBFS: `20 * log10(magnitude + eps)`
8. Apply smoothing (EMA) for display
9. Estimate noise floor (adaptive percentile)
10. Zero edge bins (first/last 2.5%)

**Frequency Axis**:
```python
freqs_hz = (
    np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_sps))
    + center_freq_hz
)
```

**Noise Floor Estimation**:
- Exclude edge bins (first/last 5%)
- Detect wideband signals (>20% of bins above threshold)
- Adaptive percentile:
  - Wideband: 2nd percentile (excludes signal energy)
  - Narrowband: 10th percentile (standard method)

### Capture System

**Memory-Efficient Design**:
- IQ samples streamed directly to disk (no in-memory buffering)
- Spectrogram computed in chunks (5M samples = ~40 MB per chunk)
- Final spectrogram downsampled to ≤512x512 for ML
- Aggressive memory cleanup after processing

**Capture Workflow**:
1. Receive `CaptureRequest` (manual or Tripwire)
2. Check cooldown/rate limits
3. Pause live scan (snapshot parameters)
4. Tune SDR to capture frequency
5. Apply bandwidth/gain from request metadata
6. Prime stream (dummy reads to warm up)
7. Capture IQ to disk (chunked writes, power-of-two reads)
8. Compute spectrogram (chunked processing)
9. Extract ML features (spectrogram, PSD, stats)
10. Classify signal (if SNR > threshold)
11. Write artifacts (IQ, spectrogram PNG, metadata JSON)
12. Publish capture result event
13. Resume live scan

**Artifact Metadata** (`capture.json`):
- Request provenance (reason, source_node, scan_plan)
- RF configuration (freq, sample_rate, bandwidth, gain)
- Timing information (timestamp, duration, sample count)
- Derived stats (SNR, occupied BW, duty cycle, burstiness)
- Quality metrics (clipping, DC offset, partial capture)
- Classification results (label, confidence, top-k)
- File references (IQ path, spectrogram path, etc.)

### Machine Learning Pipeline

**Input Format**:
- Spectrogram tensor: ≤512x512, float32, noise-floor normalized
- Time axis: rows, Frequency axis: columns
- Power units: dB relative to noise floor

**Model Architecture**:
- PyTorch CNN (ResNet-based or custom)
- GPU acceleration (CUDA on Jetson)
- 23+ output classes (device/protocol types)

**Classification Output**:
```python
{
    "label": "elrs",
    "confidence": 0.95,
    "topk": [
        {"label": "elrs", "p": 0.95},
        {"label": "fhss_control", "p": 0.03},
        {"label": "unknown", "p": 0.02}
    ],
    "device_name": "ELRS",
    "signal_type": "FHSS Control"
}
```

**Fallback Chain**:
1. PyTorch GPU classifier (primary)
2. ONNX Runtime classifier (fallback)
3. Orchestrator classifier (legacy)
4. Stub classifier (no-op, always returns "unknown")

### WebSocket Protocols

**Live FFT WebSocket** (`/ws/live_fft`):
- Binary frame format (32-byte header + float32 arrays)
- Frame structure:
  - Magic: "SPRF" (4 bytes)
  - Version: 1 (1 byte)
  - Flags: has_instant_spectrum (1 byte)
  - Header length: 32 (2 bytes)
  - FFT size: uint32 (4 bytes)
  - Center frequency: int64 (8 bytes)
  - Sample rate: uint32 (4 bytes)
  - Timestamp: float32 (4 bytes)
  - Noise floor: float32 (4 bytes)
  - Smoothed spectrum: float32[fft_size]
  - Instant spectrum (optional): float32[fft_size]

**Events WebSocket** (`/ws/notify`):
- JSON messages for UI updates
- Event types:
  - `tripwire_cue`: New Tripwire detection
  - `edge_mode`: Mode change (manual/armed)
  - `tripwire_nodes`: Node registry update
  - `capture_start`: Capture initiated
  - `capture_result`: Capture completed
  - `classification_result`: ML classification result

**Tripwire Link WebSocket** (`/ws/tripwire_link`):
- Bidirectional communication with Tripwire nodes
- Event forwarding (Edge → Tripwire)
- State synchronization (Edge → Tripwire)
- Remote scan plan setting: operator can send a scan plan to a connected node via UI (dropdown + "Send plan" button) or `POST /api/tripwire/scan-plan`; Edge forwards `set_scan_plan` over the node's WebSocket.

---

## Workflows

### Manual Capture Workflow

1. Operator sets SDR parameters (freq, sample_rate, bandwidth, gain)
2. Operator clicks "Capture" button in UI
3. UI sends POST `/api/capture/start` with parameters
4. CaptureManager queues `CaptureRequest`
5. Capture worker executes capture (see Capture System above)
6. Classification runs automatically
7. Results displayed in UI and logged to capture history

### Armed Mode Auto-Capture Workflow

1. Operator enables "Armed Mode" in UI
2. Orchestrator sets mode to "armed"
3. Tripwire node sends event to ingest endpoint (`/api/tripwire/event`)
4. Ingest endpoint validates event and forwards to main app
5. Orchestrator checks auto-capture eligibility:
   - Event type must be actionable (confirmed_event, fhss_cluster)
   - Confidence ≥ min_confidence (0.90)
   - Cooldowns satisfied (global, per-node, per-freq)
   - Rate limit not exceeded
6. If eligible, create `CaptureRequest` with priority 60
7. CaptureManager queues request
8. Capture executes (same as manual workflow)
9. Classification runs
10. If classification confidence ≥ threshold, send to ATAK
11. Results logged and displayed

### Tripwire Integration Workflow

1. Tripwire node connects via WebSocket (`/ws/tripwire_link`)
2. Edge registers node in TripwireRegistry
3. Node sends periodic heartbeats
4. Node sends detection events:
   - Cues: Advisory only, stored in UI but not actionable
   - Confirmed events: Eligible for auto-capture (if armed)
5. Edge processes events:
   - Record cues for UI display
   - Evaluate confirmed events for auto-capture
   - Update node health status
6. Edge sends status updates to node (mode, capture results)

### ATAK Integration Workflow

1. CoTBroadcaster initialized at startup
2. On mode change to "armed":
   - Send status: "SPEAR-Edge Online connected to N tripwires ARMED"
3. On capture start:
   - Send chat: "SPEAR-Edge capturing X.XXX MHz (Tripwire cue)"
4. On capture completion (confirmed events only):
   - Read classification from capture.json
   - Send chat: "DEVICE_NAME detected @ X.XXX MHz (confidence Y.YY)"
   - Send detection marker (CoT XML)
5. On AoA cone events (2+ cones with GPS):
   - Calculate TAI (Triangulated Area of Interest)
   - Send TAI polygon to ATAK

---

## Integration Points

### Tripwire Event Format

**v1.1 Format** (legacy):
```json
{
    "type": "rf_cue" | "confirmed_event",
    "stage": "cue" | "confirmed",
    "node_id": "tripwire-001",
    "freq_hz": 915000000,
    "bandwidth_hz": 2000000,
    "confidence": 0.95,
    "scan_plan": "survey_narrow",
    "classification": "elrs",
    "timestamp": 1234567890.123
}
```

**v2.0 Format** (current):
```json
{
    "type": "rf_cue" | "fhss_cluster" | "aoa_cone" | "rf_energy_start",
    "node_id": "tripwire-001",
    "freq_hz": 915000000,
    "bandwidth_hz": 2000000,
    "confidence": 0.95,
    "scan_plan": "survey_narrow",
    "bearing_deg": 45.0,  // For AoA events
    "cone_width_deg": 15.0,  // For AoA events
    "timestamp": 1234567890.123
}
```

### API Endpoints

**Health** (prefix `/health`):
- `GET /health`: System health check
- `GET /health/status`: Orchestrator status (mode, scan state, cues)
- `GET /health/sdr`: SDR health metrics

**Live / Tasking** (prefix `/live` — main UI backend):
- `POST /live/start`: Start live scan (center_freq_hz, sample_rate_sps, fft_size, fps)
- `POST /live/stop`: Stop live scan
- `POST /live/smoothing`: Set FFT smoothing alpha (body: `alpha`)
- `GET /live/mode`: Get current mode (manual/armed)
- `POST /live/mode/set`: Set mode (body: `mode`)
- `GET /live/auto/policy`: Get auto-capture policy
- `POST /live/auto/policy`: Set auto-capture policy
- `POST /live/sdr/config`: Set SDR config (frequency, sample rate, gain, etc.)
- `GET /live/sdr/info`: SDR device information and current config
- `GET /live/captures`: List recent captures (query: `limit`)

**Capture**:
- `POST /api/capture/start`: Manual capture request
- `POST /api/capture/label`: Update capture classification label

**Hub**:
- `GET /api/hub/nodes`: Read-only list of known Tripwire nodes (from persisted registry). Live node list is also pushed via WebSocket `tripwire_nodes`.

**Tripwire**:
- `POST /api/tripwire/event`: Ingest Tripwire event (also reachable via ingest on port 8000)
- `POST /api/tripwire/cue`: Legacy alias for event (backward compatibility)
- `GET /api/tripwire/ping`: Liveness check
- `GET /api/tripwire/aoa-fusion`: AoA fusion data (active cones + node GPS for triangulation, up to 3 nodes)
- `POST /api/tripwire/auto_reject`: Configure auto-reject policy
- `POST /api/tripwire/scan-plan`: Send scan plan to a connected Tripwire node (body: `node_id`, `scan_plan`); forwarded via WebSocket as `set_scan_plan`. UI: scan plan dropdown and "Send plan" button on each Tripwire node card.

**Network**:
- `GET /api/network/config`: Get network interface configuration (interfaces and addresses)
- `POST /api/network/set`: Set IP address for an interface (body: `interface`, `address`). Used by UI for network setup.

**Edge Mode**:
- `GET /api/edge/mode`: Get current mode
- `POST /api/edge/mode/{mode}`: Set mode (manual/armed)

**ML** (`/api/ml` and `/ml`):
- `GET /api/ml/models`: List available ML models
- `GET /api/ml/models/current`: Current active model info
- `POST /api/ml/models/load`: Load ML model
- `POST /api/ml/models/activate`: Activate a model by name
- `POST /api/ml/models/export`: Export current model as ZIP
- `POST /api/ml/models/import`: Import model from uploaded ZIP
- `POST /api/ml/models/test`: Test model on a capture
- `GET /api/ml/captures`: List captures (for ML dashboard)
- `GET /api/ml/captures/{capture_dir}`: Capture metadata
- `GET /api/ml/captures/{capture_dir}/thumbnail`: Spectrogram thumbnail
- `POST /api/ml/captures/{capture_dir}/label`: Update capture label
- `POST /api/ml/captures/batch-label`: Batch label update
- `POST /api/ml/captures/{capture_dir}/delete`, `POST /api/ml/captures/batch-delete`: Delete captures
- `GET /api/ml/class-labels`: List class labels
- `GET /api/ml/stats`: Dataset/capture stats
- `POST /api/ml/train/quick`: Start quick fine-tune (1–2 captures, classification head)
- `GET /api/ml/train/status/{job_id}`: Training job status
- `POST /api/ml/train/cancel/{job_id}`: Cancel training job
- `GET /ml`: ML dashboard (HTML page)

---

## Hardware Requirements

### Primary Platform

**NVIDIA Jetson Orin Nano**:
- CPU: 6-core ARM Cortex-A78AE
- GPU: 1024-core NVIDIA Ampere architecture
- RAM: 8 GB LPDDR5
- Storage: NVMe SSD recommended (for IQ capture)
- OS: Jetson Linux (Ubuntu-based)

### SDR Hardware

**bladeRF 2.0 micro** (xA4/xA9):
- Frequency range: 47 MHz - 6 GHz
- Sample rate: Up to 61.44 MS/s (USB 3.0)
- Practical limit on Jetson: ~30-40 MS/s
- Channels: 1 RX (single channel) or 2 RX (dual channel)
- Gain range: 0-60 dB (automatic LNA optimization)
- Format: SC16_Q11 (interleaved int16 I/Q)

**Optional Hardware**:
- BT200 External LNA: +16-20 dB gain (bias-tee controlled)
- GPS module: For time synchronization (via gpsd)

### USB Configuration

**USB 3.0 Required**:
- High sample rates (>20 MS/s) require USB 3.0
- USB 2.0 limited to ~4 MS/s
- Check USB speed: `lsusb -t` (should show "480M" for USB 2.0, "5000M" for USB 3.0)

**Linux USB Memory Pool**:
- Default limit: 16 MB
- BladeRF buffer configuration must stay under limit
- Calculation: `buffer_size * 4 bytes * num_buffers < 16 MB`

---

## Key Technical Details

### bladeRF Stream Lifecycle

**CRITICAL ORDER** (must be followed exactly):
1. Set sample rate FIRST
2. Set bandwidth
3. Set frequency
4. Enable RX channel
5. Configure stream (bladerf_sync_config)
6. Activate stream (bladerf_enable_module)
7. Set gain (AFTER stream is active)

**Why This Order Matters**:
- Stream activation before RF parameters causes sample rate to revert to 4 MHz
- Gain set before stream activation gets clamped to 60 dB
- Wrong order causes buffer timeouts and stream failures

### Sample Rate Limits

**Jetson Orin Nano Practical Limits**:
- Maximum: ~30-40 MS/s (with optimization)
- Recommended: 2.4-20 MS/s for general use
- High performance: 20-40 MS/s (requires USB 3.0, careful tuning)

**Buffer Sizing**:
- Ring buffer: `int(sample_rate_sps * 0.5)` (0.5 seconds)
- Read chunk: Adaptive based on sample rate:
  - >40 MS/s: 192K samples (75% of 256K buffer)
  - 20-40 MS/s: 96K samples (75% of 128K buffer)
  - 10-20 MS/s: 48K samples (75% of 64K buffer)
  - <10 MS/s: 24K samples (75% of 32K buffer)

**Read Size Requirements**:
- MUST be power-of-two (8192, 16384, 32768, etc.)
- Arbitrary sizes cause buffer timeouts
- Use `BLADE_RF_READ_SAMPLES = 8192` constant for consistency

### IQ Format and Scaling

**bladeRF Format**: SC16_Q11
- Interleaved int16 I/Q samples
- Range: [-2048, 2047] (11-bit signed)
- Normalization: Divide by 2048.0 to get [-1.0, 1.0) range
- Alternative scaling: Divide by 32768.0 for int16 range (matches SDR++)

**Storage Format**: CF32_LE
- Complex float32 (8 bytes per sample)
- Little-endian byte order
- Compatible with SigMF standard

### FFT Processing Details

**Window Function**:
- Type: Hanning (Hann)
- Coherent gain normalization: Divide by `sum(window)`
- Edge bin handling: Zero first/last 2.5% (removes window sidelobes)

**Noise Floor Estimation**:
- Exclude edge bins (first/last 5%)
- Adaptive percentile:
  - Wideband signals (>20% bins active): 2nd percentile
  - Narrowband signals: 10th percentile
- Smoothing: Asymmetrical attack/release for FFT display

**Calibration**:
- Default: 0.0 dBFS (true Q11 scaling)
- Alternative: -24.08 dBFS (SDR++-style 16-bit scaling)
- Configurable via `SPEAR_CALIBRATION_OFFSET_DB` env var
- Display-only offset (backend uses true Q11 internally)

### Memory Management

**Capture Memory Efficiency**:
- IQ samples streamed directly to disk (no buffering)
- Spectrogram computed in chunks (5M samples = ~40 MB)
- Final spectrogram downsampled to ≤512x512
- Aggressive cleanup after processing (gc.collect())

**Ring Buffer**:
- Thread-safe (threading.Lock)
- Fixed size: `int(sample_rate_sps * duration)`
- Duration: 0.3s (high rates) or 0.5s (normal rates)
- Memory: `size * 8 bytes` (complex64)

**ML Model Loading**:
- GPU cache cleared before loading (torch.cuda.empty_cache())
- Models loaded lazily (on first classification)
- Fallback chain prevents memory issues

### Error Handling Philosophy

**Never Crash Runtime**:
- Live FFT processing must never crash (catch all exceptions)
- SDR operations fail gracefully (return error codes)
- WebSocket handlers handle disconnects cleanly
- Subscriber failures isolated (don't break event bus)

**Logging Strategy**:
- Descriptive prefixes: `[CAPTURE]`, `[ORCH]`, `[SCAN]`, `[RX]`, `[FFT WS]`
- Log important state changes and errors
- Avoid excessive logging in tight loops
- Throttle logging (every 2 seconds for frame keys)

---

## Performance Characteristics

### Real-Time Performance

**FFT Frame Rate**:
- Target: 30 FPS (configurable)
- Actual: Depends on sample rate and FFT size
- Frame drops: UI can drop old frames (latest-wins)

**Waterfall Rendering**:
- Uses `requestAnimationFrame` for smooth updates
- Offscreen canvas for scrolling optimization
- Device pixel ratio handling for high-DPI displays

**SDR Throughput**:
- Line rate: Up to 30-40 MS/s (Jetson limit)
- USB overhead: ~10-15% (USB 3.0)
- Ring buffer: Prevents sample loss during FFT processing

### Capture Performance

**Capture Duration**:
- Default: 5 seconds
- Configurable: 1-60 seconds
- Actual duration: May be shorter if stream issues occur

**Processing Time**:
- IQ capture: Real-time (streaming to disk)
- Spectrogram: ~1-2 seconds for 5-second capture
- Classification: ~100-500 ms (GPU-accelerated)
- Total: ~6-8 seconds for 5-second capture

**Memory Usage**:
- Peak during capture: ~200-300 MB (chunked processing)
- After capture: ~50-100 MB (cleanup)
- Disk I/O: ~40-80 MB/s (depends on sample rate)

### ML Inference Performance

**GPU Acceleration**:
- PyTorch CUDA: ~100-500 ms per spectrogram
- ONNX Runtime: ~200-1000 ms per spectrogram
- CPU fallback: ~2-5 seconds per spectrogram

**Model Size**:
- PyTorch model: ~10-50 MB
- ONNX model: ~5-20 MB
- Memory footprint: ~100-200 MB (GPU)

### Network Performance

**WebSocket Throughput**:
- Live FFT: ~1-2 MB/s (30 FPS, 2048-point FFT)
- Binary format: More efficient than JSON
- Frame dropping: UI drops old frames if slow

**HTTP API**:
- Response time: <100 ms (most endpoints)
- Capture start: <50 ms (queue insertion)
- Status endpoints: <10 ms

---

## Conclusion

SPEAR-Edge is a comprehensive RF monitoring and capture system that combines real-time spectrum analysis, automated signal capture, machine learning classification, and tactical integration. Its event-driven architecture, memory-efficient design, and hardware abstraction make it suitable for both standalone operation and distributed sensor networks.

The system's key strengths are:
- **Performance**: GPU-accelerated ML, optimized for Jetson Orin Nano
- **Reliability**: Never-crash philosophy, graceful error handling
- **Extensibility**: Modular design, easy to add new SDR drivers or ML models
- **Integration**: Tripwire and ATAK integration for tactical operations
- **Usability**: Web-based UI, comprehensive API, detailed logging

For detailed technical documentation, see:
- [Technical Overview](TECHNICAL_OVERVIEW.md)
- [API Reference](API_REFERENCE.md)
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Hardware Guide](HARDWARE_GUIDE.md)
- [FFT/Waterfall Guide](FFT_WATERFALL_BLADERF_GUIDE.md)
