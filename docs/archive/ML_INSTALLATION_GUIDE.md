# ML Installation Guide

This guide covers installing all dependencies needed for ML model training (development machine) and inference (Jetson Orin Nano).

## Overview

There are two separate environments with different requirements:

1. **Development Machine** (for training): Needs PyTorch, torchsig, and training libraries
2. **Jetson Orin Nano** (for inference): Needs ONNX Runtime with GPU support

---

## Part 1: Development Machine Setup (Training)

### Prerequisites

- Python 3.10+ (recommended: 3.10 or 3.11)
- CUDA-capable GPU (optional but recommended for faster training)
- At least 16GB RAM (32GB recommended)
- 50GB+ free disk space for datasets

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv_ml

# Activate virtual environment
# On Linux/Mac:
source venv_ml/bin/activate
# On Windows:
venv_ml\Scripts\activate
```

### Step 2: Install PyTorch

**For CUDA-capable GPU (recommended):**

```bash
# Check your CUDA version first
nvidia-smi

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (slower but works):
pip install torch torchvision torchaudio
```

**Verify installation:**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Install TorchSig

```bash
# Install torchsig (RF signal datasets and utilities)
pip install torchsig

# Verify installation
python -c "import torchsig; print(f'TorchSig version: {torchsig.__version__}')"
```

### Step 4: Install Training Dependencies

```bash
# Core ML libraries
pip install numpy>=1.23,<2.0
pip install scikit-learn  # For metrics (accuracy, F1-score, confusion matrix)
pip install tensorboard    # For training visualization

# ONNX export (for converting trained models)
pip install onnx
pip install onnxruntime  # For testing ONNX export locally

# Data handling
pip install matplotlib   # For visualization (optional but useful)
pip install tqdm         # Progress bars
```

### Step 5: Install Additional Utilities

```bash
# For dataset preparation and utilities
pip install pandas       # For data manipulation
pip install pillow       # For image processing (if needed)
```

### Step 6: Verify Installation

Create a test script `test_ml_setup.py`:

```python
#!/usr/bin/env python3
"""Test ML setup installation."""

print("Testing ML setup...")

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available (CPU-only mode)")
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")

# Test TorchSig
try:
    import torchsig
    print(f"✅ TorchSig: {torchsig.__version__}")
except ImportError as e:
    print(f"❌ TorchSig not installed: {e}")

# Test other dependencies
deps = {
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "tensorboard": "tensorboard",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
}

for name, module in deps.items():
    try:
        __import__(module)
        print(f"✅ {name}: installed")
    except ImportError:
        print(f"❌ {name}: not installed")

print("\nSetup complete!")
```

Run it:
```bash
python test_ml_setup.py
```

### Step 7: Install SPEAR-Edge Base Dependencies (Optional)

If you want to test data preparation with actual SPEAR-Edge captures:

```bash
# Install base SPEAR-Edge dependencies (for reading capture artifacts)
pip install -r requirements.txt
```

---

## Part 2: Jetson Orin Nano Setup (Inference)

### Prerequisites

- Jetson Orin Nano with JetPack 5.x installed
- Python 3.10+ (should come with JetPack)
- CUDA and cuDNN (included in JetPack)

### Step 1: Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### Step 2: Install ONNX Runtime for Jetson

**Option A: Install Jetson-specific ONNX Runtime (Recommended)**

Jetson requires a special build of ONNX Runtime with TensorRT support. The easiest way is to use the pre-built wheels from NVIDIA or build from source.

**Using pre-built wheel (if available):**

```bash
# Check your JetPack version
cat /etc/nv_tegra_release

# Install ONNX Runtime GPU (Jetson-specific)
# Note: Version may vary based on JetPack version
pip3 install onnxruntime-gpu
```

**Option B: Build from source (if pre-built not available):**

```bash
# Install build dependencies
sudo apt install -y build-essential cmake python3-dev python3-pip

# Install ONNX Runtime dependencies
pip3 install numpy

# Clone and build ONNX Runtime (this takes a while)
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
```

**Option C: Use CPU-only ONNX Runtime (Fallback)**

If GPU support isn't critical initially:

```bash
pip3 install onnxruntime
```

### Step 3: Verify ONNX Runtime Installation

```python
#!/usr/bin/env python3
"""Test ONNX Runtime on Jetson."""

import onnxruntime as ort

print("Testing ONNX Runtime...")
print(f"Version: {ort.__version__}")

# Check available providers
providers = ort.get_available_providers()
print(f"Available providers: {providers}")

if 'TensorrtExecutionProvider' in providers:
    print("✅ TensorRT provider available (best performance)")
elif 'CUDAExecutionProvider' in providers:
    print("✅ CUDA provider available (good performance)")
elif 'CPUExecutionProvider' in providers:
    print("⚠️  Only CPU provider available (slower)")
```

### Step 4: Install Base Dependencies

```bash
# Install SPEAR-Edge requirements (if not already installed)
pip3 install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Test that ONNX Runtime can load models
python3 -c "import onnxruntime as ort; print('ONNX Runtime OK')"

# Test that inference works
python3 -c "import numpy as np; import onnxruntime as ort; sess = ort.InferenceSession('spear_edge/ml/models/spear_dummy.onnx'); print('Model loading OK')"
```

---

## Part 3: Quick Installation Scripts

### Development Machine: `install_ml_dev.sh`

```bash
#!/bin/bash
# install_ml_dev.sh - Install ML dependencies for development machine

set -e

echo "Creating virtual environment..."
python3 -m venv venv_ml
source venv_ml/bin/activate

echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing TorchSig..."
pip install torchsig

echo "Installing training dependencies..."
pip install numpy>=1.23,<2.0
pip install scikit-learn tensorboard onnx onnxruntime
pip install matplotlib tqdm pandas pillow

echo "Installing SPEAR-Edge base dependencies..."
pip install -r requirements.txt

echo "✅ Installation complete!"
echo "Activate with: source venv_ml/bin/activate"
```

### Jetson: `install_ml_jetson.sh`

```bash
#!/bin/bash
# install_ml_jetson.sh - Install ML dependencies for Jetson

set -e

echo "Updating system..."
sudo apt update
sudo apt upgrade -y

echo "Installing ONNX Runtime..."
# Try GPU version first, fallback to CPU
pip3 install onnxruntime-gpu || pip3 install onnxruntime

echo "Installing base dependencies..."
pip3 install -r requirements.txt

echo "✅ Installation complete!"
```

---

## Part 4: Troubleshooting

### Development Machine Issues

**Problem: PyTorch CUDA not available**
- Solution: Verify CUDA installation: `nvidia-smi`
- Install correct PyTorch version for your CUDA version
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

**Problem: TorchSig import errors**
- Solution: Ensure PyTorch is installed first
- Try: `pip install --upgrade torchsig`

**Problem: Out of memory during training**
- Solution: Reduce batch size in training script
- Use gradient accumulation for effective larger batches

### Jetson Issues

**Problem: ONNX Runtime TensorRT provider not available**
- Solution: This is normal if using CPU-only ONNX Runtime
- For TensorRT, you need Jetson-specific build or build from source
- CPU provider will work but is slower

**Problem: Model loading fails**
- Solution: Verify model file exists and is readable
- Check ONNX model validity: `python -c "import onnx; onnx.checker.check_model('model.onnx')"`

**Problem: Inference is slow**
- Solution: Ensure TensorRT or CUDA provider is available
- Check active provider: `session.get_providers()`
- Consider model quantization (INT8) for faster inference

---

## Part 5: Verification Checklist

### Development Machine
- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed and CUDA available (if GPU)
- [ ] TorchSig installed
- [ ] Training dependencies installed (scikit-learn, tensorboard, onnx)
- [ ] Test script runs successfully

### Jetson
- [ ] JetPack 5.x installed
- [ ] ONNX Runtime installed (GPU version preferred)
- [ ] TensorRT or CUDA provider available
- [ ] Base dependencies installed
- [ ] Can load dummy ONNX model

---

## Next Steps

After installation:

1. **Development Machine**: Proceed with data preparation and model training
2. **Jetson**: Wait for trained model, then deploy to `spear_edge/ml/models/rf_classifier.onnx`

See the main ML implementation plan for next steps.
