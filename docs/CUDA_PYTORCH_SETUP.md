# PyTorch CUDA Setup for Jetson Orin Nano - cuDNN Compatibility Issue

## Problem
PyTorch wheels for JetPack 6 are built against cuDNN 8, but your system has cuDNN 9.3.0 installed. A simple symlink won't work because the version symbols don't match.

## Current Status
- ✅ CUDA Toolkit 12.6 - Installed
- ✅ NVIDIA Driver 540.4.0 - Installed  
- ✅ cuDNN 9.3.0 - Installed (system)
- ✅ TensorRT 10.3.0 - Installed
- ✅ PyTorch 2.4.0 (with CUDA) - Installed but can't load due to cuDNN mismatch

## Solutions

### Option 1: Install cuDNN 8 Manually (Recommended if you need native PyTorch)

1. Download cuDNN 8 for CUDA 12.x from NVIDIA Developer site:
   - Visit: https://developer.nvidia.com/cudnn
   - Download cuDNN 8.x for CUDA 12.x (Linux ARM64)
   - Extract and install the libraries

2. Install cuDNN 8 libraries:
```bash
# After downloading and extracting cuDNN 8
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig
```

3. Create symlink:
```bash
sudo ln -sf /usr/local/cuda/lib64/libcudnn.so.8 /usr/lib/aarch64-linux-gnu/libcudnn.so.8
sudo ldconfig
```

4. Test PyTorch:
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Option 2: Use Docker Container (Easiest, Most Reliable)

NVIDIA provides pre-built containers with all dependencies configured:

```bash
# Pull the container
docker pull dustynv/l4t-pytorch:r36.3.0-cu124

# Run with GPU access
docker run --runtime nvidia -it --rm \
  -v /home/spear/spear-edgev1_0:/workspace \
  dustynv/l4t-pytorch:r36.3.0-cu124

# Inside container, test:
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Option 3: Wait for cuDNN 9-Compatible PyTorch Build

Monitor NVIDIA's PyTorch releases for JetPack 6. New builds may support cuDNN 9.

Check: https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/

### Option 4: Build PyTorch from Source (Advanced)

Build PyTorch specifically for your environment with cuDNN 9 support. This is complex and time-consuming.

## Verification Script

After applying a solution, run:

```python
import torch
print("=" * 50)
print("PyTorch CUDA Verification")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU computation test: SUCCESS (result shape: {z.shape})")
else:
    print("❌ CUDA is NOT available")
print("=" * 50)
```

## Notes

- The pip-installed `nvidia-cudnn-cu12` package only provides cuDNN 9, not cuDNN 8
- System cuDNN 9 cannot be easily downgraded without affecting other JetPack components
- Docker solution is the most reliable for development/testing
- For production, Option 1 (manual cuDNN 8 install) may be preferred
