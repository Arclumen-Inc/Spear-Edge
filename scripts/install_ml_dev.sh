#!/bin/bash
# install_ml_dev.sh - Install ML dependencies for development machine
# Usage: ./scripts/install_ml_dev.sh

set -e

echo "=========================================="
echo "SPEAR-Edge ML Development Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $(python3 --version)"

if [ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.10" ]; then
    echo "⚠️  Warning: Python 3.10+ recommended (found $PYTHON_VERSION)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
if [ ! -d "venv_ml" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_ml
else
    echo "Virtual environment already exists, skipping creation..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_ml/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Check for CUDA
echo ""
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "✅ CUDA detected: Version $CUDA_VERSION"
    echo "Installing PyTorch with CUDA support..."
    
    # Determine CUDA version and install appropriate PyTorch
    if [ "$(printf '%s\n' "11.8" "$CUDA_VERSION" | sort -V | head -n1)" = "11.8" ]; then
        echo "Installing PyTorch for CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [ "$(printf '%s\n' "12.1" "$CUDA_VERSION" | sort -V | head -n1)" = "12.1" ]; then
        echo "Installing PyTorch for CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "Installing PyTorch for CUDA $CUDA_VERSION (default)..."
        pip install torch torchvision torchaudio
    fi
else
    echo "⚠️  No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "❌ PyTorch installation failed"
    exit 1
}

# Install TorchSig
echo ""
echo "Installing TorchSig..."
pip install torchsig

# Install training dependencies
echo ""
echo "Installing training dependencies..."
pip install "numpy>=1.23,<2.0"
pip install scikit-learn tensorboard onnx onnxruntime
pip install matplotlib tqdm pandas pillow

# Install SPEAR-Edge base dependencies (optional, for testing with captures)
echo ""
echo "Installing SPEAR-Edge base dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, skipping..."
fi

# Create test script
echo ""
echo "Creating test script..."
cat > test_ml_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test ML setup installation."""

print("Testing ML setup...")
print("")

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

print("")
print("Setup complete!")
EOF

chmod +x test_ml_setup.py

# Run test
echo ""
echo "Running installation test..."
python3 test_ml_setup.py

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv_ml/bin/activate"
echo ""
echo "To test the setup:"
echo "  python3 test_ml_setup.py"
echo ""
