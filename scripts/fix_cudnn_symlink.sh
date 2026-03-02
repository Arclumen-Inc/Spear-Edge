#!/bin/bash
# Script to create cuDNN 8 symlink for PyTorch compatibility
# This is needed because PyTorch was built against cuDNN 8, but the system has cuDNN 9

echo "Creating cuDNN 8 symlink for PyTorch compatibility..."
sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 /usr/lib/aarch64-linux-gnu/libcudnn.so.8
sudo ldconfig
echo "Symlink created! Testing PyTorch..."
python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PYEOF
