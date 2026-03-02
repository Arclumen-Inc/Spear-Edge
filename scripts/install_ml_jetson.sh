#!/bin/bash
# install_ml_jetson.sh - Install ML dependencies for Jetson Orin Nano
# Usage: ./scripts/install_ml_jetson.sh

set -e

echo "=========================================="
echo "SPEAR-Edge ML Jetson Setup"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: This script is designed for Jetson devices"
    echo "   /etc/nv_tegra_release not found"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Jetson device detected:"
    cat /etc/nv_tegra_release
    echo ""
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Update system
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install build dependencies (if needed)
echo ""
echo "Installing build dependencies..."
sudo apt install -y build-essential cmake python3-dev python3-pip || true

# Install ONNX Runtime
echo ""
echo "Installing ONNX Runtime..."
echo "Attempting to install GPU version (with TensorRT support)..."

# Try GPU version first
if pip3 install onnxruntime-gpu; then
    echo "✅ ONNX Runtime GPU installed"
else
    echo "⚠️  GPU version failed, trying CPU version..."
    pip3 install onnxruntime
    echo "✅ ONNX Runtime CPU installed"
fi

# Verify ONNX Runtime
echo ""
echo "Verifying ONNX Runtime installation..."
python3 << 'EOF'
import onnxruntime as ort
print(f"✅ ONNX Runtime: {ort.__version__}")
providers = ort.get_available_providers()
print(f"Available providers: {providers}")

if 'TensorrtExecutionProvider' in providers:
    print("✅ TensorRT provider available (best performance)")
elif 'CUDAExecutionProvider' in providers:
    print("✅ CUDA provider available (good performance)")
elif 'CPUExecutionProvider' in providers:
    print("⚠️  Only CPU provider available (slower, but works)")
EOF

# Install base dependencies
echo ""
echo "Installing SPEAR-Edge base dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing minimal dependencies..."
    pip3 install numpy>=1.23,<2.0
fi

# Test model loading (if dummy model exists)
echo ""
if [ -f "spear_edge/ml/models/spear_dummy.onnx" ]; then
    echo "Testing model loading with dummy model..."
    python3 << 'EOF'
import onnxruntime as ort
import numpy as np
try:
    sess = ort.InferenceSession('spear_edge/ml/models/spear_dummy.onnx')
    print("✅ Model loading successful")
    print(f"   Input: {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}")
    print(f"   Output: {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")
except Exception as e:
    print(f"⚠️  Model loading test failed: {e}")
EOF
else
    echo "⚠️  Dummy model not found, skipping model test"
fi

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "To verify installation:"
echo "  python3 -c \"import onnxruntime as ort; print(ort.__version__)\""
echo ""
