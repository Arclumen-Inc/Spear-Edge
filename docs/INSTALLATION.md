# SPEAR-Edge Installation Guide

## System Requirements

### Hardware
- **NVIDIA Jetson Orin Nano** (or compatible Jetson device)
- **bladeRF 2.0 micro** (xA4 or xA9)
- USB 3.0 connection between Jetson and bladeRF
- GPS receiver (optional, for ATAK integration)

### Software
- **JetPack 5.x** (Ubuntu 22.04 based)
- **Python 3.10+**
- **libbladeRF** (native driver library)
- **CUDA** (for ML inference acceleration, optional)

## Dependencies

### System Packages

```bash
sudo apt update
sudo apt install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    libbladerf2 \
    libbladerf-dev \
    gpsd \
    gpsd-clients \
    build-essential
```

### Python Dependencies

Core dependencies (required):
```bash
pip install -r requirements.txt
```

ML development dependencies (for training, optional):
```bash
pip install -r requirements-ml-dev.txt
```

**Note:** PyTorch must be installed separately based on your CUDA version:
- CUDA 11.8: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- CPU-only: `pip install torch torchvision torchaudio`

## Installation Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd spear-edgev1_0
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify libbladeRF Installation

```bash
bladeRF-cli --version
bladeRF-cli --probe
```

Expected output should show your bladeRF device.

### 5. Verify Hardware Connection

```bash
lsusb | grep -i blade
```

Should show: `Nuand bladeRF 2.0 micro`

### 6. Test GPS (Optional)

If using GPS for ATAK integration:

```bash
sudo systemctl start gpsd
gpsmon
```

### 7. Create Data Directories

```bash
mkdir -p data/artifacts/captures
mkdir -p data/dataset_raw
mkdir -p data/dataset
```

### 8. Verify Installation

Run the main application:

```bash
python -m spear_edge.app
```

Or using uvicorn directly:

```bash
uvicorn spear_edge.app:app --host 0.0.0.0 --port 8080
```

**Optional:** Run the ingest application (for Tripwire event ingestion):

```bash
uvicorn spear_edge.ingest_app:app --host 0.0.0.0 --port 8000
```

Access the UI at: `http://localhost:8080`

## Configuration

### Environment Variables

Create a `.env` file (optional) or set environment variables:

```bash
# Server configuration
SPEAR_HOST=0.0.0.0
SPEAR_PORT=8080
SPEAR_LOG_LEVEL=WARNING

# Default scan parameters
SPEAR_CENTER_FREQ_HZ=915000000
SPEAR_SAMPLE_RATE_SPS=2400000
SPEAR_FFT_SIZE=2048
SPEAR_FPS=15.0

# RF calibration
SPEAR_CALIBRATION_OFFSET_DB=0.0
SPEAR_IQ_SCALING_MODE=int16

# GPSD connection (USB or GPIO/UART-backed gpsd)
SPEAR_GPSD_HOST=127.0.0.1
SPEAR_GPSD_PORT=2947
SPEAR_GPS_POLL_INTERVAL_S=1.0
```

### Settings File

Default settings are in `spear_edge/settings.py`. Environment variables override defaults.

## ML Model Setup (Optional)

### Place Model Files

Place trained model files in `spear_edge/ml/models/`:

- PyTorch model: `rf_classifier.pth`
- ONNX model: `rf_classifier.onnx`
- Class labels: `class_labels.json`

### Verify ML Models

The system will automatically detect and use available models:
1. PyTorch (GPU-accelerated, if CUDA available)
2. ONNX (CPU fallback)
3. Stub (no-op, if no models available)

## Troubleshooting

### bladeRF Not Detected

1. Check USB connection: `lsusb | grep blade`
2. Verify libbladeRF: `bladeRF-cli --probe`
3. Check permissions: `sudo usermod -a -G plugdev $USER` (logout/login)
4. Verify USB 3.0 connection (required for high sample rates)

### Permission Errors

```bash
sudo usermod -a -G plugdev,dialout $USER
# Logout and login again
```

### CUDA/PyTorch Issues

1. Verify CUDA installation: `nvcc --version`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Install matching PyTorch version for your CUDA version

### GPS Issues

1. Check GPS device: `ls /dev/tty* | grep -i gps`
2. Start gpsd: `sudo systemctl start gpsd`
3. Configure gpsd: `sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock`

For Jetson GPIO/UART GPS (for example `/dev/ttyTHS1`), point gpsd at the UART device:

```bash
sudo systemctl stop gpsd.socket gpsd
sudo gpsd /dev/ttyTHS1 -n -F /var/run/gpsd.sock
gpsmon
```

If you need this persistent across reboot, set the device in gpsd defaults/systemd drop-in and restart gpsd.

### Port Already in Use

Change port via environment variable:
```bash
SPEAR_PORT=8081 python -m spear_edge.app
```

## Post-Installation

### Verify Installation

1. Start the application
2. Access UI at `http://localhost:8080`
3. Start a live scan
4. Verify FFT display updates
5. Test a manual capture

### Next Steps

- Read the [User Guide](USER_GUIDE.md) for operating the system
- Review the [Technical Overview](TECHNICAL_OVERVIEW.md) for architecture details
- Check the [API Reference](API_REFERENCE.md) for integration

## Uninstallation

To remove SPEAR-Edge:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove data (optional)
rm -rf data/artifacts
```

**Note:** System packages (libbladeRF, gpsd) are not removed automatically.
