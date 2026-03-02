#!/bin/bash
# Install cuDNN 8 from extracted archive
# Run with: sudo ./install_cudnn8_now.sh

set -e

CUDNN_DIR="/home/spear/spear-edgev1_0/cudnn-linux-aarch64-8.9.5.30_cuda12-archive"
CUDA_DIR="/usr/local/cuda"

echo "=========================================="
echo "Installing cuDNN 8.9.5.30"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: This script must be run with sudo"
    echo "Usage: sudo ./install_cudnn8_now.sh"
    exit 1
fi

# Verify extracted directory exists
if [ ! -d "$CUDNN_DIR" ]; then
    echo "ERROR: cuDNN directory not found: $CUDNN_DIR"
    exit 1
fi

# Verify CUDA directory exists
if [ ! -d "$CUDA_DIR" ]; then
    echo "ERROR: CUDA directory not found: $CUDA_DIR"
    exit 1
fi

echo "Source: $CUDNN_DIR"
echo "Destination: $CUDA_DIR"
echo ""

# Copy include files
echo "Step 1: Copying header files..."
cp $CUDNN_DIR/include/cudnn*.h $CUDA_DIR/include/
echo "✓ Headers installed"

# Copy library files
echo "Step 2: Copying library files..."
cp $CUDNN_DIR/lib/libcudnn* $CUDA_DIR/lib64/
echo "✓ Libraries installed"

# Set permissions
echo "Step 3: Setting permissions..."
chmod a+r $CUDA_DIR/include/cudnn*.h
chmod a+r $CUDA_DIR/lib64/libcudnn*
echo "✓ Permissions set"

# Find and create symlink for libcudnn.so.8
echo "Step 4: Creating system symlink..."
CUDNN8_LIB=$(find $CUDA_DIR/lib64 -name "libcudnn.so.8*" -type f | head -1)
if [ -z "$CUDNN8_LIB" ]; then
    echo "WARNING: libcudnn.so.8 not found, checking available versions..."
    ls -la $CUDA_DIR/lib64/libcudnn.so* || echo "No cuDNN libraries found"
    exit 1
else
    # Remove old symlink if it exists
    rm -f /usr/lib/aarch64-linux-gnu/libcudnn.so.8
    # Create new symlink to actual cuDNN 8 library
    ln -sf $CUDNN8_LIB /usr/lib/aarch64-linux-gnu/libcudnn.so.8
    echo "✓ Symlink created: /usr/lib/aarch64-linux-gnu/libcudnn.so.8 -> $CUDNN8_LIB"
fi

# Update library cache
echo "Step 5: Updating library cache..."
ldconfig
echo "✓ Library cache updated"

echo ""
echo "=========================================="
echo "cuDNN 8 Installation Complete!"
echo "=========================================="
echo ""
echo "Verification:"
ldconfig -p | grep "libcudnn.so.8" | head -3
echo ""
echo "Next: Install PyTorch with CUDA support"
