# SPEAR-Edge User Guide

## Overview

SPEAR-Edge provides a web-based interface for RF spectrum monitoring, signal capture, and integration with Tripwire nodes. This guide covers basic operations and workflows.

## Accessing the Interface

1. Start the application (see [Installation Guide](INSTALLATION.md))
2. Open a web browser
3. Navigate to: `http://localhost:8080` (or your configured host/port)

## Main Interface

### FFT Display

The main FFT display shows:
- **FFT Line**: Max-hold power spectrum (green line)
- **Waterfall**: Time-frequency spectrogram (scrolling display)
- **Frequency Axis**: Horizontal axis showing frequency range
- **Power Axis**: Vertical axis showing power in dBFS

### Controls Panel

#### Scan Controls
- **Start/Stop**: Start or stop live scanning
- **Frequency**: Center frequency (MHz)
- **Sample Rate**: Sample rate (MS/s)
- **FFT Size**: FFT size (power of 2, typically 2048)
- **FPS**: Frames per second (default: 15)

#### SDR Controls
- **Gain**: Manual gain control (dB)
- **Bandwidth**: RF bandwidth (MHz)
- **Channel**: RX channel (0 or 1)
- **BT200**: External LNA enable (bias-tee, if connected)

#### Mode Controls
- **Mode**: Operator mode selection
  - **Manual**: Operator control only
  - **Armed**: Automatic captures enabled

## Basic Operations

### Starting a Live Scan

1. Set desired parameters:
   - Center frequency (e.g., 915 MHz)
   - Sample rate (e.g., 2.4 MS/s)
   - FFT size (e.g., 2048)
   - FPS (e.g., 15)

2. Click **Start** button

3. Observe FFT display updating in real-time

### Stopping a Live Scan

Click **Stop** button. The scan will stop cleanly, preventing buffer timeouts.

### Adjusting Gain

1. Use the **Gain** slider in the SDR controls
2. Monitor the FFT display for:
   - **Too high**: Clipping (samples max out, flat top)
   - **Too low**: Low signal levels, poor SNR
   - **Optimal**: Good dynamic range, noise floor around -70 to -95 dBFS

**Note:** Gain changes take effect immediately during active scan.

### Manual Capture

1. Ensure scan is running
2. Click **Capture** button (or use API endpoint)
3. Set capture parameters:
   - Frequency
   - Duration (default: 5 seconds)
   - Sample rate
   - Bandwidth
   - Gain

4. Capture executes:
   - Scan pauses
   - SDR tunes to capture frequency
   - IQ samples recorded
   - Spectrogram generated
   - Scan resumes

5. Capture artifacts saved to `data/artifacts/captures/`

## Operating Modes

### Manual Mode

- Operator has full control
- No automatic captures
- All captures are manual
- Tripwire cues appear for review (advisory only)

**Use when:**
- Testing or debugging
- Manual signal analysis
- Controlled capture scenarios

### Armed Mode

- Automatic captures enabled
- Tripwire confirmed events trigger captures
- Rate limiting and cooldown protection
- Operator can still perform manual captures

**Use when:**
- Monitoring with Tripwire nodes
- Automated signal collection
- Long-duration surveillance

**Auto-Capture Policy:**
- Minimum confidence: 0.90 (90%)
- Global cooldown: 3.0 seconds
- Per-node cooldown: 2.0 seconds
- Per-frequency cooldown: 8.0 seconds
- Maximum captures per minute: 10

## Tripwire Integration

### Connecting Tripwire Nodes

Tripwire nodes connect via WebSocket (`/ws/tripwire_link`):
1. Node sends hello message with node_id, GPS, callsign
2. Node sends periodic heartbeats
3. Edge tracks connection status (connected if heartbeat within 5 seconds)

### Tripwire Events

Tripwire nodes send events via HTTP (`/api/tripwire/event`):

**Event Types:**
- **RF Cue**: RF detection cue (advisory only)
- **FHSS Cluster**: FHSS cluster detection
- **AoA Cone**: Angle of Arrival cone (advisory only)
- **RF Energy**: Energy tracking events (advisory only)
- **RF Spike**: Spike detection (advisory only)

**Event Stages:**
- **Energy**: Energy detection stage
- **Cue**: Cue stage (advisory only)
- **Confirmed**: Confirmed detection (actionable in armed mode)

### Auto-Capture Behavior

In **Armed Mode**, confirmed events trigger automatic captures:
1. Event received from Tripwire node
2. Policy check (confidence, cooldown, rate limit)
3. If allowed, capture queued and executed
4. Capture artifacts saved with metadata

**Rejection Reasons:**
- Confidence too low
- Global cooldown active
- Per-node cooldown active
- Per-frequency cooldown active
- Rate limit exceeded

## Capture Artifacts

### Capture Directory Structure

Each capture creates a directory:
```
data/artifacts/captures/YYYYMMDD_HHMMSS_FREQHz_<hash>/
├── capture.iq          # IQ samples (complex64)
├── capture.json         # Metadata
├── spectrogram.png      # Spectrogram image
└── thumbnail.png        # Thumbnail (if generated)
```

### Capture Metadata

`capture.json` contains:
- Timestamp
- Frequency, sample rate, bandwidth
- Gain settings
- Duration
- Source node (if from Tripwire)
- Classification (if available)
- GPS coordinates (if available)

### Viewing Captures

1. Navigate to capture directory
2. View `spectrogram.png` for visual analysis
3. Load `capture.iq` in analysis tools (e.g., GNU Radio, Python)
4. Review `capture.json` for metadata

## ML Classification

### Automatic Classification

Captures are automatically classified (if ML models available):
1. Spectrogram generated
2. ML model inference
3. Classification result stored in `capture.json`

### Manual Labeling

Update classification label via API:
```bash
curl -X POST http://localhost:8080/api/capture/label \
  -H "Content-Type: application/json" \
  -d '{
    "capture_dir": "20260307_043820_915000000Hz_...",
    "label": "elrs"
  }'
```

## Troubleshooting

### FFT Not Updating

1. Check scan is running (Start button active)
2. Verify WebSocket connection (check browser console)
3. Check SDR health: `GET /health/sdr`
4. Review logs for errors

### No Signal Visible

1. **CRITICAL: Verify frequency is correct** (if using version before fix)
   - Check backend logs for "RX0 configured: req_fc=... act_fc=..."
   - If `act_fc` differs significantly from `req_fc`, frequency tuning may be incorrect
   - Update to latest version to fix frequency tuning bug for signals above 4.29 GHz
2. Check gain settings (may be too low)
3. Verify frequency range (signal may be outside range)
4. Check antenna connection
5. Verify sample rate (may be too low for signal bandwidth)
6. For wideband signals (VTX): Check antenna polarization (RHCP vs. linear)

### Capture Fails

1. Check capture queue not full
2. Verify SDR is not in use by another process
3. Check disk space
4. Review logs for errors

### Tripwire Not Connecting

1. Verify network connectivity
2. Check WebSocket endpoint: `ws://edge-ip:8080/ws/tripwire_link`
3. Verify node sends hello message
4. Check firewall settings

### High CPU Usage

1. Reduce sample rate
2. Reduce FPS
3. Reduce FFT size
4. Check for background processes

## Best Practices

### Gain Management

- Start with low gain (0-10 dB)
- Increase gradually until signal visible
- Avoid clipping (samples max out)
- Target noise floor: -70 to -95 dBFS

### Sample Rate Selection

- Use minimum sample rate for signal bandwidth
- Higher rates = more CPU usage
- Jetson Orin Nano limit: ~30 MS/s
- Default: 2.4 MS/s (good balance)

### Frequency Planning

- Set center frequency to signal of interest
- Sample rate determines frequency range
- Range = center ± (sample_rate / 2)

### Capture Duration

- Default: 5 seconds
- Longer = more data, more disk space
- Shorter = less context, faster processing
- Adjust based on signal characteristics

### Mode Selection

- Use **Manual** for testing and debugging
- Use **Armed** for automated monitoring
- Monitor auto-capture rate in Armed mode
- Adjust policy if needed

## Advanced Features

### AoA Fusion

View AoA (Angle of Arrival) data from multiple Tripwire nodes:
- `GET /api/tripwire/aoa-fusion`
- Returns active AoA cones with GPS positions
- Useful for triangulation and bearing visualization

### Scan Plans

Update scan plan for Tripwire nodes:
- `POST /api/tripwire/scan-plan`
- Configure node scanning behavior
- Coordinate multi-node scanning

### Network Configuration

Configure server settings:
- `GET /api/network/config`
- `POST /api/network/set`
- Change host/port (requires restart)

## Keyboard Shortcuts

(If implemented in UI)
- `Space`: Start/Stop scan
- `C`: Capture
- `+/-`: Adjust gain
- `[/]`: Adjust frequency

## Support

For technical issues:
1. Check logs (console output)
2. Review [API Reference](API_REFERENCE.md)
3. Check [Developer Guide](DEVELOPER_GUIDE.md)
4. Review [Hardware Guide](HARDWARE_GUIDE.md)
