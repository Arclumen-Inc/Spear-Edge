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

## Protocol Decode (Remote ID)

### Recommended operational path (current)

SPEAR-Edge now supports a **separate Wi-Fi monitor workflow** for RID/Wi-Fi intelligence:

- Main RF workflow remains on `/` (bladeRF FFT/waterfall + capture + ML).
- Dedicated Wi-Fi operations page is `/wifi` (Kismet-backed monitor controls and intel).
- Main UI still shows compact Wi-Fi/RID alerts and protocol detections for operator awareness.

This is the preferred path for field RID work because ASTM RID is typically visible as
Wi-Fi/BLE packet-layer data, not from wideband SDR IQ alone.

### Wi-Fi Radio page (`/wifi`)

The `/wifi` page includes:

- **Wi-Fi Monitor Service** controls: save config, test Kismet connection, start/stop monitor.
- **Kismet Service controls** (via Spear Manager): status/start/stop.
- **RID Detections (Wi-Fi)** cards.
- **Channel Activity**, **Top Emitters**, **Data Sources**, **Device Details**, **Anomalies**.
- **Alert Controls** for Kismet presence-alert operations.

### Kismet requirements

Kismet is a **separate package** and is not bundled with SPEAR-Edge. If it is not installed yet, see **`docs/KISMET_INSTALL_JETSON.md`** or run **`sudo bash scripts/install_kismet_jetson.sh`** (Ubuntu LTS codenames only; otherwise use the official packages page).

After installation, Kismet must run on the same host (or a host reachable at `SPEAR_WIFI_MONITOR_KISMET_URL`):

```bash
sudo systemctl enable --now kismet.service
sudo systemctl status kismet.service --no-pager
curl -I http://127.0.0.1:2501
```

Expected:

- `kismet.service` is active/running.
- HTTP endpoint responds on `:2501`.

### Spear Manager integration for Kismet service control

SPEAR `/wifi` calls **Spear Manager** (separate service, default port **8081**) for `GET/POST …/kismet/status|start|stop`.

**Install Spear Manager** from the bundle under `/home/spear/spear_manager/` using the SPEAR-Edge script (adds Kismet routes to the manager API):

```bash
cd /home/spear/spear-edgev1_0   # or your clone path
bash scripts/install_spear_manager.sh
```

Full details: **`docs/SPEAR_MANAGER.md`**.

Set in `spear-edge.service`:

```ini
Environment=SPEAR_WIFI_MANAGER_URL=http://127.0.0.1:8081
Environment=SPEAR_WIFI_MANAGER_TOKEN=
```

If your Spear Manager uses a token, set the same value in `SPEAR_WIFI_MANAGER_TOKEN`.
If manager auth is disabled, leave token empty.

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart spear-edge
```

### Wi-Fi monitor configuration defaults (service env)

Optional environment variables in `spear-edge.service`:

```ini
Environment=SPEAR_WIFI_MONITOR_BACKEND=kismet
Environment=SPEAR_WIFI_MONITOR_IFACE=wlP1p1s0
Environment=SPEAR_WIFI_MONITOR_CHANNEL_MODE=hop
Environment=SPEAR_WIFI_MONITOR_POLL_INTERVAL_S=2.0
Environment=SPEAR_WIFI_MONITOR_HOP_CHANNELS=1,6,11,36,44,149
Environment=SPEAR_WIFI_MONITOR_KISMET_URL=http://127.0.0.1:2501
Environment=SPEAR_WIFI_MONITOR_KISMET_USERNAME=
Environment=SPEAR_WIFI_MONITOR_KISMET_PASSWORD=
Environment=SPEAR_WIFI_MONITOR_KISMET_TIMEOUT_S=3.0
Environment=SPEAR_WIFI_MONITOR_AUTOSTART=false
```

### What can be validated before monitor-capable adapter arrives

- Kismet service lifecycle and endpoint reachability.
- `/wifi` API and UI wiring.
- manager-based Kismet service control from `/wifi`.

### What requires monitor-capable Wi-Fi hardware (ex: AX210)

- Real packet intake in monitor mode.
- Source/channel hopping performance.
- Live RID-over-Wi-Fi detection quality.

---

SPEAR-Edge can emit protocol decode artifacts for Remote ID captures:

- Artifact path per capture: `decode/remote_id.json`
- UI panel: **Protocol Detections** (right column)

By default, the wrapper runs in safe fallback mode (writes `no_decode` if no backend decoder is configured).

To enable a real decoder backend:

1. Edit `scripts/spear-edge.service`
2. Set:
   - `SPEAR_RID_DECODER_CMD` (already set to wrapper script)
   - `SPEAR_RID_BACKEND_CMD` to your decoder command
3. Reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart spear-edge
```

Verify in logs:

```bash
journalctl -u spear-edge -f | rg "CAPTURE|remote_id|decode"
```

### Quick end-to-end UI validation with stub backend

If you want to validate protocol UI/fusion flow before integrating a real RID decoder:

1. Edit `scripts/spear-edge.service`
2. Set:

```ini
Environment=SPEAR_RID_BACKEND_CMD=/home/spear/spear-edgev1_0/venv/bin/python /home/spear/spear-edgev1_0/scripts/rid_backend_stub.py
```

3. Reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart spear-edge
```

4. Trigger a manual capture in a RID candidate band (2.4 GHz or 5.8 GHz).
5. Confirm:
   - `decode/remote_id.json` exists in capture directory
   - **Protocol Detections** panel shows `REMOTE_ID · DECODED_VERIFIED`
   - ML classification still appears as shadow/provenance in the ML panel

### DJI stub validation

You can do the same for DJI protocol flow:

```ini
Environment=SPEAR_DJI_BACKEND_CMD=/home/spear/spear-edgev1_0/venv/bin/python /home/spear/spear-edgev1_0/scripts/dji_backend_stub.py
```

After `daemon-reload` and restart, run a capture in 2.4/5.8 GHz and confirm:

- `decode/dji_droneid.json` appears in capture directory
- **Protocol Detections** shows `DJI_DRONEID · DECODED_VERIFIED`

### Live DJI DroneID from IQ (samples2djidroneid)

SPEAR stores IQ as **interleaved float32 I/Q** (`numpy.complex64`) in `iq/samples.iq`, which matches the **GNU Radio complex float32 (“fc32”)** layout expected by the community decoder:

- Upstream project: [anarkiwi/samples2djidroneid](https://github.com/anarkiwi/samples2djidroneid)
- Supported capture rates for that decoder: **15.36 Msps** or **30.72 Msps** (other rates will produce a `decode_error` artifact with a clear reason).

**Steps**

1. Build the Docker image or native `samples2djidroneid` binary from the upstream repository.
2. In your systemd unit (or shell for dev), point the DJI backend at `scripts/dji_samples2droneid_backend.py` and choose **native** or **Docker**:
   - **Docker** (recommended on Jetson if you do not want a local build):
     - `Environment=SPEAR_DJI_SAMPLES2_DOCKER_IMAGE=samples2djidroneid` (use the image name you built)
     - `Environment=SPEAR_DJI_BACKEND_CMD=/home/spear/spear-edgev1_0/venv/bin/python /home/spear/spear-edgev1_0/scripts/dji_samples2droneid_backend.py`
   - **Native binary**:
     - `Environment=SPEAR_DJI_SAMPLES2_CMD=/usr/local/bin/samples2djidroneid` (or a basename on `PATH`)
     - Same `SPEAR_DJI_BACKEND_CMD=...dji_samples2droneid_backend.py` as above.
3. Optional: `SPEAR_DJI_SAMPLES2_TIMEOUT_S` (default `180`) if decoding is slow on busy spectrum.
4. `daemon-reload`, restart `spear-edge`, run a manual capture on a DJI/Occusync-relevant frequency with **15.36 or 30.72 Msps**.
5. Inspect `decode/dji_droneid.json` and the **Protocol Detections** card.

**systemd note:** If any `Environment=` value contains spaces, wrap the entire assignment in double quotes (see [Environment=](https://www.freedesktop.org/software/systemd/man/systemd.exec.html#Environment=)).

### Live ASTM Remote ID (PCAP sidecar path)

ASTM Remote ID is normally received as **Wi‑Fi (beacon / NAN) or Bluetooth LE** frames, not as “automatic decode” from a raw wideband IQ file alone. SPEAR therefore supports an optional **PCAP sidecar** placed next to `samples.iq` in the same `iq/` folder:

| File name (pick one) |
|----------------------|
| `remote_rid.pcapng` |
| `remote_rid.pcap` |
| `rid_sidecar.pcapng` |
| `rid_sidecar.pcap` |

Your workflow is:

1. While the drone is transmitting Remote ID, record a monitor-mode PCAP (external tool: `tcpdump`, Wireshark, `droneid-go -pcap`, vendor sniffer, etc.) and save it under one of the names above **inside the capture’s `iq/` directory** (same folder as `samples.iq`).
2. Set `SPEAR_RID_BACKEND_CMD` to `scripts/rid_pcap_sidecar_backend.py`.
3. Set `SPEAR_RID_PCAP_DECODER_CMD` to a shell token list (parsed with Python `shlex.split`) that **must include both placeholders** `{PCAP}` and `{OUT}`:
   - `{PCAP}` expands to the absolute path of the sidecar file.
   - `{OUT}` expands to the absolute path of `decode/remote_id.json` for this capture (your decoder should write **SPEAR-shaped JSON** there, or print one JSON object on stdout).
4. Restart the service and run a capture in a RID candidate band so the decoder stage runs.

If no sidecar exists, the artifact will be `no_decode` with reason `rid_pcap_sidecar_not_found`. If `SPEAR_RID_PCAP_DECODER_CMD` is unset, you get `rid_pcap_decoder_cmd_not_configured`.

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
