# Option 1: Manual cuDNN 8 Installation Guide

## Prerequisites
- NVIDIA Developer account (free) - https://developer.nvidia.com/
- CUDA 12.6 installed (✅ Already installed)
- sudo access (✅ Available)

## Step-by-Step Installation

### Step 1: Download cuDNN 8

1. Visit: https://developer.nvidia.com/cudnn
2. Sign in with your NVIDIA Developer account (or create one - it's free)
3. Navigate to "Download cuDNN"
4. Select:
   - **Version**: cuDNN 8.x (latest 8.x version available)
   - **Platform**: Linux
   - **Architecture**: aarch64 (ARM64) - **IMPORTANT for Jetson**
   - **CUDA Version**: 12.x (to match your CUDA 12.6)
5. Download the tar file (e.g., `cudnn-linux-aarch64-8.x.x.x_cuda12.x-archive.tar.xz`)

### Step 2: Extract and Install cuDNN 8

```bash
# Navigate to download directory
cd ~/Downloads  # or wherever you downloaded the file

# Extract the archive
tar -xvf cudnn-linux-aarch64-8.x.x.x_cuda12.x-archive.tar.xz

# Copy include files
sudo cp cudnn-linux-aarch64-8.x.x.x_cuda12.x-archive/include/cudnn*.h /usr/local/cuda/include

# Copy library files
sudo cp cudnn-linux-aarch64-8.x.x.x_cuda12.x-archive/lib/libcudnn* /usr/local/cuda/lib64

# Set proper permissions
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Update library cache
sudo ldconfig
```

### Step 3: Create System Symlink

```bash
# Create symlink for PyTorch compatibility
sudo ln -sf /usr/local/cuda/lib64/libcudnn.so.8 /usr/lib/aarch64-linux-gnu/libcudnn.so.8

# Update library cache again
sudo ldconfig
```

### Step 4: Verify cuDNN 8 Installation

```bash
# Check if cuDNN 8 libraries are available
ldconfig -p | grep cudnn

# Verify symlink
ls -la /usr/lib/aarch64-linux-gnu/libcudnn.so.8
```

### Step 5: Install PyTorch

```bash
# Install PyTorch wheel for JetPack 6
python3 -m pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
```

### Step 6: Verify PyTorch CUDA Support

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("GPU:", torch.cuda.get_device_name(0))
```

## Troubleshooting

### Issue: "libcudnn.so.8: version not found"
- **Solution**: Make sure you installed actual cuDNN 8, not just a symlink to cuDNN 9
- Check: `readelf -d /usr/local/cuda/lib64/libcudnn.so.8 | grep SONAME`

### Issue: "Permission denied"
- **Solution**: Use `sudo` for system library operations

### Issue: "CUDA not available" in PyTorch
- **Solution**: Verify CUDA libraries are in PATH:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```

## Notes

- cuDNN 8 and cuDNN 9 can coexist - they're in different locations
- PyTorch will use cuDNN 8 from `/usr/local/cuda/lib64/`
- System cuDNN 9 remains in `/usr/lib/aarch64-linux-gnu/` for other applications
- The symlink is needed because PyTorch looks in the system library path
