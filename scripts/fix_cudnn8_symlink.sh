#!/bin/bash
# Fix cuDNN 8 symlink - point to actual cuDNN 8 library instead of cuDNN 9

echo "Fixing cuDNN 8 symlink..."
sudo rm -f /usr/lib/aarch64-linux-gnu/libcudnn.so.8
sudo ln -sf /usr/local/cuda/lib64/libcudnn.so.8.9.5 /usr/lib/aarch64-linux-gnu/libcudnn.so.8
sudo ldconfig
echo "✓ Symlink fixed to point to cuDNN 8"
echo ""
echo "Verification:"
ls -la /usr/lib/aarch64-linux-gnu/libcudnn.so.8
