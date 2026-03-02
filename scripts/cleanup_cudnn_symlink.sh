#!/bin/bash
# Cleanup script to remove cuDNN 8 symlink created during PyTorch installation attempts

echo "Removing cuDNN 8 symlink..."
sudo rm -f /usr/lib/aarch64-linux-gnu/libcudnn.so.8
sudo ldconfig
echo "Cleanup complete!"
