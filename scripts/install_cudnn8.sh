#!/bin/bash
# Automated cuDNN 8 installation script for Jetson Orin Nano
# This script helps install cuDNN 8 after manual download

set -e

echo "=========================================="
echo "cuDNN 8 Installation for Jetson Orin Nano"
echo "=========================================="
echo ""

# Check if running as root for certain operations
if [ "$EUID" -ne 0 ]; then 
    echo "Note: Some operations require sudo privileges"
fi

# Check if CUDA is installed
if [ ! -d "/usr/local/cuda" ]; then
    echo "ERROR: CUDA not found at /usr/local/cuda"
    echo "Please install CUDA first"
    exit 1
fi

echo "CUDA found at: /usr/local/cuda"
echo "CUDA version: $(cat /usr/local/cuda/version.txt 2>/dev/null || echo 'Unknown')"
echo ""

# Prompt for cuDNN archive location
echo "Please provide the path to the extracted cuDNN 8 archive directory"
echo "Example: /home/spear/Downloads/cudnn-linux-aarch64-8.x.x.x_cuda12.x-archive"
echo ""
read -p "Enter cuDNN 8 archive directory path: " CUDNN_DIR

if [ ! -d "$CUDNN_DIR" ]; then
    echo "ERROR: Directory not found: $CUDNN_DIR"
    echo ""
    echo "If you haven't downloaded cuDNN 8 yet:"
    echo "1. Visit: https://developer.nvidia.com/cudnn"
    echo "2. Download cuDNN 8.x for CUDA 12.x (aarch64/ARM64)"
    echo "3. Extract the tar file"
    echo "4. Run this script again with the extracted directory path"
    exit 1
fi

# Verify it's a cuDNN directory
if [ ! -d "$CUDNN_DIR/include" ] || [ ! -d "$CUDNN_DIR/lib" ]; then
    echo "ERROR: Invalid cuDNN archive directory"
    echo "Expected structure: $CUDNN_DIR/{include,lib}/"
    exit 1
fi

echo ""
echo "Installing cuDNN 8 from: $CUDNN_DIR"
echo ""

# Copy include files
echo "Copying header files..."
sudo cp $CUDNN_DIR/include/cudnn*.h /usr/local/cuda/include/
echo "✓ Headers installed"

# Copy library files
echo "Copying library files..."
sudo cp $CUDNN_DIR/lib/libcudnn* /usr/local/cuda/lib64/
echo "✓ Libraries installed"

# Set permissions
echo "Setting permissions..."
sudo chmod a+r /usr/local/cuda/include/cudnn*.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
echo "✓ Permissions set"

# Create symlink for system library path
echo "Creating system symlink..."
CUDNN8_LIB=$(find /usr/local/cuda/lib64 -name "libcudnn.so.8*" -type f | head -1)
if [ -z "$CUDNN8_LIB" ]; then
    echo "WARNING: libcudnn.so.8 not found in /usr/local/cuda/lib64"
    echo "Checking available versions..."
    ls -la /usr/local/cuda/lib64/libcudnn.so* || echo "No cuDNN libraries found"
else
    sudo ln -sf $CUDNN8_LIB /usr/lib/aarch64-linux-gnu/libcudnn.so.8
    echo "✓ Symlink created: /usr/lib/aarch64-linux-gnu/libcudnn.so.8 -> $CUDNN8_LIB"
fi

# Update library cache
echo "Updating library cache..."
sudo ldconfig
echo "✓ Library cache updated"

echo ""
echo "=========================================="
echo "cuDNN 8 Installation Complete!"
echo "=========================================="
echo ""
echo "Verification:"
ldconfig -p | grep cudnn | head -3
echo ""
echo "Next steps:"
echo "1. Install PyTorch:"
echo "   python3 -m pip install --no-cache-dir \\"
echo "     https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl"
echo ""
echo "2. Test PyTorch CUDA:"
echo "   python3 -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
echo ""
